"""
test1_telecontrol.py — Gesture control, wired to the REAL Thor arm
--------------------------------------------------------------------
Same pinch-clutch VELOCITY scheme as fingertip_clutch_velocity.py (1.4.py):

  PINCH thumb+index            → ENGAGE. The spot where you pinched (position
                                 AND hand size) becomes the neutral "center".
  Move hand AWAY from center   → drift at a SPEED proportional to offset:
                                   left/right            → J1 base rotation (θ)
                                   up/down               → height (z)
                                   toward/away from cam  → FORWARD/BACK along the
                                                           direction the gripper
                                                           is pointing (tool axis)
  Release pinch                → STOP DEAD instantly.

  GRIPPER:  PINCH thumb+PINKY  → TOGGLES the gripper.
  FIST (hold ~1s)              → toggle FREEZE (overrides everything).

Full gesture description: see fingertip_clutch_velocity.py.

WHAT'S NEW HERE — real hardware output over serial to a CUSTOM Arduino
firmware (thor_teleop_firmware/thor_teleop_firmware.ino), IN ADDITION to the
existing thor_simulator.html websocket twin (both run at once; the simulator
is just a visual reference). This branch does NOT talk to Grbl at all.

  Why not Grbl: Grbl 0.9j queues G-code moves and can't cancel one for a
  fresher one — it fully finishes before accepting the next correction,
  which caps how often the arm can change direction, well below camera
  frame rate. On top of that, Grbl's G1 is POSITION control — it assumes
  the arm's mechanics reliably reach where it's told, which this robot's
  don't always do (it lags/slips mechanically sometimes). A position model
  lets that slip quietly compound into a growing, eventually-wrong belief
  about where a joint actually is.

  So this branch does TRUE VELOCITY CONTROL instead, matching what the
  simulator already gets: every frame we send a signed deg/s for the base
  (A) and upper-elbow (B/C) axes — never a position. The firmware has no
  absolute-position concept at all, so one frame's mechanical slip never
  carries into the next command; it only ever matters for as long as that
  one command is in effect. The firmware also force-stops itself if it
  stops hearing from Python (see CMD_TIMEOUT_MS in the .ino) — losing the
  connection stops the robot, it does not keep spinning at its last speed.

  Real-hardware output starts DISARMED. The gesture loop, freeze/fist, and
  simulator all work exactly as before regardless of arm state.
    H  → ARM / DISARM the real arm (only takes effect once connected and not
         faulted). Nothing to "re-zero" with velocity control — arming just
         starts sending whatever the current commanded speed already is.
    X  → clear a latched fault (e.g. a limit switch trip reported by the
         firmware). Does NOT re-arm; press H again once it's safe.

  Speed is intentionally conservative:
    - every driven axis is rate-limited to HW_MAX_SPEED deg/s
    - the upper elbow (B/C) runs away fast for a given command, so it gets an
      EXTRA 10x slowdown on top of that (HW_UPPER_ELBOW_SCALE)
    - the elbow (D) is not wired up in the firmware AT ALL on this branch —
      there is no code path that can move it, not even a disabled one

  IMPORTANT: the Arduino side (stepsPerDegA/B, direction signs, limit switch
  polarity, gripper pin) is UNCALIBRATED placeholder guesswork — see the
  bring-up checklist at the top of thor_teleop_firmware.ino. Do that before
  arming anything near people.

Requirements: pip install opencv-python "mediapipe==0.10.9" numpy websockets pyserial
Run: python test1_telecontrol.py
     Optionally open thor_simulator.html → Connect, for the visual twin.

Controls: Q=quit  R=reset pose  Space=freeze  [ ]=max speed
          G=toggle grip  O=open  P=close   (keyboard mirrors the gesture)
          H=arm/disarm real arm   X=clear a latched fault
"""

import cv2, mediapipe as mp, numpy as np
import math, json, asyncio, threading, websockets, time, queue
import serial
from serial.tools import list_ports

# ── Robot geometry (real Thor mm) ─────────────────────────────────────────────
L1, L2, L3, L4 = 202.0, 160.0, 80.0, 163.0
A, B = L3, L4
SHOULDER_H = L1 + L2
REACH_MIN = abs(A-B)+15
REACH_MAX = A+B-10
LIMITS = {"J1":(-165,165),"J3":(-90,90),"J4":(-180,180),"J5":(-150,150),"J6":(-180,180)}

# Keep the arm slightly bent near the top so it never reaches the fully-straight
# vertical SINGULARITY (where the tip locks up). R_FLOOR is the minimum radial
# play we insist on; HEIGHT_CEIL is the corresponding height cap.
R_FLOOR     = 55.0
HEIGHT_CEIL = SHOULDER_H + math.sqrt(max(REACH_MAX*REACH_MAX - R_FLOOR*R_FLOOR, 0.0))
HEIGHT_FLOOR = SHOULDER_H - 160.0

# Workspace clamps
WS_HEIGHT = (SHOULDER_H-260, SHOULDER_H+250)
WS_REACH  = (REACH_MIN+5, REACH_MAX-5)

# ── Velocity tuning ───────────────────────────────────────────────────────────
MAX_J1_SPEED     = 80.0     # deg/s at full hand offset (left/right → J1)
MAX_HEIGHT_SPEED = 160.0    # mm/s   (up/down → height z)
MAX_REACH_SPEED  = 140.0    # mm/s   (forward/back along the tool axis)
DEADZONE         = 0.04
CURVE_POW        = 2.0
REACH_GAIN       = 8.0
INPUT_SMOOTH     = 0.65
SCALE_SMOOTH     = 0.18

PINCH_ON  = 0.38
PINCH_OFF = 0.55

J3_SHARE = 0.55

# ── State ─────────────────────────────────────────────────────────────────────
cmd_j1     = 0.0
cmd_height = SHOULDER_H + 10.0
cmd_R      = 160.0

engaged   = False
frozen    = True
anchor    = None

GRIP_CLOSED = 0.0
GRIP_OPEN   = 90.0
gripper_pos = GRIP_OPEN

GRIP_PINCH_ON  = 0.45
GRIP_PINCH_OFF = 0.70
GRIP_SMOOTH    = 0.30
grip_closing   = False
grip_pinched   = False

FIST_HOLD_S     = 1.0
FIST_COOLDOWN   = 0.6
WAKE_BOB_S      = 2.0
WAKE_BOB_MM     = 28.0
SLEEP_BOB_S     = 1.4
SLEEP_BOB_MM    = 34.0
fist_held       = False
fist_start      = None
fist_lock_until = 0.0
anim_until      = 0.0
anim_start      = 0.0
anim_kind       = "wake"

OUT_SMOOTH = 0.40
sm_j1 = sm_j3 = sm_j5 = None

sm_hx = sm_hy = sm_scale = None

# ── WebSocket ─────────────────────────────────────────────────────────────────
_clients=set(); _lock=threading.Lock()
_state={"angles":[0,0,0,0,0,0],"gripper":0.0,"mode":"TELECONTROL"}
def set_state(j1,j3,j4,j5,j6=0.0):
    with _lock:
        _state["angles"]=[round(j1,2),0.0,round(j3,2),round(j4,2),round(j5,2),round(j6,2)]
def get_state():
    with _lock: return dict(_state)
async def ws_handler(ws, path=None):
    global _clients; _clients.add(ws); print(f"[ws] connected ({len(_clients)})")
    try: await ws.wait_closed()
    finally: _clients.discard(ws)
async def broadcaster():
    global _clients
    while True:
        if _clients:
            msg=json.dumps(get_state()); dead=set()
            for c in list(_clients):
                try: await c.send(msg)
                except: dead.add(c)
            _clients-=dead
        await asyncio.sleep(1/40)
async def ws_main():
    async with websockets.serve(ws_handler,"localhost",8765):
        print("[ws] ready -> ws://localhost:8765"); await broadcaster()
threading.Thread(target=lambda: asyncio.run(ws_main()), daemon=True).start()

# ── Math ──────────────────────────────────────────────────────────────────────
def clamp(v,lo,hi): return max(lo,min(hi,v))
def mapv(v,a,b,c,d): return c+clamp((v-a)/(b-a),0,1)*(d-c)

def response(x):
    s=abs(x)
    if s<DEADZONE: return 0.0
    s=(s-DEADZONE)/(1-DEADZONE)
    return math.copysign(clamp(s,0,1)**CURVE_POW, x)

def _ik_planar(r, y, elbow):
    d = clamp(math.hypot(r, y), abs(A-B)+1, A+B-1)
    cos_el = clamp((A*A+B*B-d*d)/(2*A*B), -1, 1)
    el = math.acos(cos_el)
    j5 = -(math.pi-el)*elbow
    phi = math.atan2(r, y)
    cos_psi = clamp((A*A+d*d-B*B)/(2*A*d), -1, 1)
    psi = math.acos(cos_psi)
    j3 = phi + psi*elbow
    j3 = math.degrees(j3); j5 = math.degrees(j5)
    while j3>180: j3-=360
    while j3<-180: j3+=360
    return j3, j5

def solve_reach_ik(height, reach):
    r = reach
    y = height - SHOULDER_H
    best=None
    for e in (+1,-1):
        j3,j5=_ik_planar(r,y,e)
        pen = (abs(j3)-90)*10 if abs(j3)>90 else 0
        score = pen + abs(j3)
        if best is None or score<best[0]:
            best=(score,j3,j5)
    j3,j5=best[1],best[2]
    return clamp(j3,*LIMITS["J3"]), clamp(j5,*LIMITS["J5"])

def clamp_reach(height, R):
    dy = height - SHOULDER_H
    hi_sq = REACH_MAX*REACH_MAX - dy*dy
    lo_sq = REACH_MIN*REACH_MIN - dy*dy
    R_hi = math.sqrt(hi_sq) if hi_sq > 0 else 0.0
    R_lo = math.sqrt(lo_sq) if lo_sq > 0 else 0.0
    R_lo = max(R_lo, R_FLOOR)
    if R_hi < R_lo: R_hi = R_lo
    return clamp(R, R_lo, R_hi)

def pinch_distance(lm):
    return math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y)

def hand_scale(lm):
    return math.hypot(lm[0].x-lm[9].x, lm[0].y-lm[9].y)

def pinch_ratio(lm):
    s = hand_scale(lm)
    return pinch_distance(lm)/s if s > 1e-6 else 1.0

def thumb_pinky_distance(lm):
    return math.hypot(lm[4].x-lm[20].x, lm[4].y-lm[20].y)

def is_fist(lm):
    def d(i): return math.hypot(lm[i].x-lm[0].x, lm[i].y-lm[0].y)
    fingers = [(8,6),(12,10),(16,14),(20,18)]
    curled = sum(1 for tip,pip in fingers if d(tip) < d(pip)*0.92)
    return curled >= 4

# ── Hardware (real Thor arm over the CUSTOM firmware, no Grbl) ────────────────
# Starts DISARMED every run. Press H once connected to start relaying motion;
# X clears a latched fault (e.g. a limit switch trip) without re-arming.
#
# TRUE VELOCITY CONTROL: every frame we send a signed deg/s for A and B, not
# a position. The firmware has no absolute-position concept at all — this
# is deliberate, because the arm's mechanics lag/slip sometimes, and a
# position-tracking model would let that slip compound into a growing,
# eventually-wrong destination. With velocity control, each frame just says
# "how fast, which way, right now" — one frame's slip never carries into the
# next command. The firmware also force-stops if it stops hearing from us
# (CMD_TIMEOUT_MS in the .ino), so silence stops the robot rather than
# leaving it spinning at its last speed.
HW_BAUD              = 115200
HW_BOOT_DELAY_S       = 2.0   # Arduino resets when the serial port opens; wait for setup()
HW_MIRROR_C          = True   # upper-elbow's 2nd motor (C) is mounted opposed to B —
                              # also mirrored independently in the firmware; kept here
                              # too only for the on-screen readout, not sent over wire

HW_MAX_SPEED         = 8.0    # deg/s ceiling for the base (A) axis
HW_UPPER_ELBOW_SCALE = 0.1    # upper elbow (B/C) additionally cut to 10% of that (0.8 deg/s)
# Elbow (D) is not part of this protocol at all — the firmware doesn't even
# wire that axis up, so there is no code path here that could move it.

hw_link  = None    # FirmwareLink once connected
hw_port  = None
hw_armed = False
hw_fault = False

hw_prev_j3 = None   # previous frame's J3 target, to derive its effective deg/s
hw_last_send_t = 0.0   # last time we computed dt (kept fresh every frame, even
                       # while disarmed, so re-arming never sees a stale/huge dt)

class FirmwareLink:
    """Thin serial link to thor_teleop_firmware.ino — no protocol beyond
    plain text lines out, and whatever the firmware prints back in."""

    def __init__(self):
        self.ser = None
        self.rx = queue.Queue()
        self._stop = threading.Event()
        self._thread = None

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    def open(self, port):
        self.ser = serial.Serial(port, HW_BAUD, timeout=0.2)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.ser is not None:
            try: self.ser.close()
            except Exception: pass
            self.ser = None

    def _reader(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as exc:
                self.rx.put(f"read failed: {exc}")
                return
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                text = raw.decode("ascii", "replace").strip()
                if text:
                    self.rx.put(text)

    def send_line(self, text):
        if not self.is_open:
            return False
        try:
            self.ser.write((text + "\n").encode("ascii"))
            return True
        except Exception as exc:
            self.rx.put(f"write failed: {exc}")
            return False

def choose_serial_port():
    ports = [p.device for p in list_ports.comports()]
    if not ports:
        print("[hw] no serial ports found — running SIM-ONLY")
        return None
    if len(ports) == 1:
        print(f"[hw] auto-selected {ports[0]}")
        return ports[0]
    print("[hw] multiple serial ports found:")
    for i, p in enumerate(ports):
        print(f"  {i}: {p}")
    while True:
        choice = input(f"[hw] port index or name (Enter for {ports[0]}, "
                       f"S for sim-only): ").strip()
        if not choice:
            return ports[0]
        if choice.lower() == "s":
            print("[hw] running SIM-ONLY")
            return None
        if choice.isdigit() and 0 <= int(choice) < len(ports):
            return ports[int(choice)]
        for p in ports:
            if p.lower() == choice.lower():
                return p
        print(f"[hw] '{choice}' isn't a valid index or port name — try again "
              f"(or S for sim-only)")

def hw_process_rx():
    """Drain buffered firmware lines, print them, and latch a fault on a
    reported limit-switch trip."""
    global hw_fault, hw_armed
    if hw_link is None:
        return
    while True:
        try:
            line = hw_link.rx.get_nowait()
        except queue.Empty:
            break
        print(f"[fw] {line}")
        if "LIMIT HIT" in line:
            hw_fault = True; hw_armed = False
            print("[hw] limit switch fault latched — clear with X, arm again with H")

def hw_tick(now, v_j1, j3_now, grip_target_deg):
    """Send the CURRENT velocity for A and B every frame — no position, no
    waiting. v_j1 is already a deg/s hand-velocity (same signal used for the
    simulator). J3 has no direct velocity signal of its own (it's IK-solved
    from height/reach, not driven straight from a hand axis), so its
    effective deg/s is derived as the discrete change in its target position
    over dt. Both get clamped to their hardware speed ceilings right before
    sending. The dt/derivative bookkeeping runs every call regardless of arm
    state, so hw_prev_j3 and the dt baseline never go stale while disarmed —
    only the actual send is gated on hw_armed."""
    global hw_last_send_t, hw_prev_j3

    dt = (now - hw_last_send_t) if hw_last_send_t else 0.0
    hw_last_send_t = now
    if hw_prev_j3 is None:
        hw_prev_j3 = j3_now
    v_j3 = (j3_now - hw_prev_j3) / dt if dt > 1e-6 else 0.0
    hw_prev_j3 = j3_now

    if not (hw_armed and hw_link is not None and hw_link.is_open and not hw_fault):
        return

    v1 = clamp(v_j1, -HW_MAX_SPEED, HW_MAX_SPEED)
    v3 = clamp(v_j3, -HW_MAX_SPEED*HW_UPPER_ELBOW_SCALE, HW_MAX_SPEED*HW_UPPER_ELBOW_SCALE)

    hw_link.send_line(f"A{v1:.3f} B{v3:.3f} G{grip_target_deg:.1f}")

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands=mp.solutions.hands; mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                     min_detection_confidence=0.7,min_tracking_confidence=0.6)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

# ── Hardware bring-up ──────────────────────────────────────────────────────────
hw_port = choose_serial_port()
if hw_port:
    try:
        hw_link = FirmwareLink()
        hw_link.open(hw_port)
        print(f"[hw] opened {hw_port} — waiting {HW_BOOT_DELAY_S:.1f}s for the "
              f"Arduino to reset and run setup()...")
        time.sleep(HW_BOOT_DELAY_S)
        hw_link.ser.reset_input_buffer()
        hw_link.ser.reset_output_buffer()
        print("[hw] ready. Press H (with the preview window focused) to arm.")
    except Exception as exc:
        print(f"[hw] could not open {hw_port}: {exc} — running SIM-ONLY")

print("test1_telecontrol.py — open thor_simulator.html -> Connect (optional twin)")
print("PINCH thumb+index to engage. Push hand from center = speed. Release = stop.")
print("PINCH thumb+pinky to close the gripper. Separate to open.")
print("H = arm/disarm the real arm.  X = clear a latched fault.")

last_t=time.time()

while True:
    ok,frame=cap.read()
    if not ok: break
    frame=cv2.flip(frame,1); h,w=frame.shape[:2]; cw=w-255; cam=frame[:,:cw]
    now=time.time(); dt=clamp(now-last_t,0,0.1); last_t=now
    hw_process_rx()

    res=hands.process(cv2.cvtColor(cam,cv2.COLOR_BGR2RGB))
    v_j1=v_h=v_r=0.0
    hand_pos=None; grip_d=None

    fist_now=False
    if res.multi_hand_landmarks:
        lm=res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(cam,res.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(35,45,55),thickness=2,circle_radius=2),
            mp_draw.DrawingSpec(color=(55,65,80),thickness=1))

        fist_now=is_fist(lm)
        if fist_now and not fist_held:
            if fist_start is None and now>=fist_lock_until:
                fist_start=now
            if fist_start is not None and now-fist_start>=FIST_HOLD_S:
                fist_held=True
                fist_start=None
                frozen=not frozen
                fist_lock_until=now+FIST_COOLDOWN
                engaged=False; anchor=None
                anim_start=now
                if frozen:
                    anim_kind="sleep"; anim_until=now+SLEEP_BOB_S
                else:
                    anim_kind="wake";  anim_until=now+WAKE_BOB_S
                print(f"[fist] {'OFF (frozen)' if frozen else 'ON (listening)'}")
        elif not fist_now:
            fist_held=False; fist_start=None

        ctrl_ok = (not frozen) and (not fist_now) and (now>=anim_until)
        if not ctrl_ok:
            sm_hx=sm_hy=sm_scale=None
        if ctrl_ok:
            ratio=pinch_ratio(lm)
            raw_hx,raw_hy=(lm[4].x+lm[8].x)/2,(lm[4].y+lm[8].y)/2
            raw_scale=hand_scale(lm)
            if sm_hx is None: sm_hx,sm_hy,sm_scale=raw_hx,raw_hy,raw_scale
            sm_hx   += INPUT_SMOOTH*(raw_hx-sm_hx)
            sm_hy   += INPUT_SMOOTH*(raw_hy-sm_hy)
            sm_scale+= SCALE_SMOOTH*(raw_scale-sm_scale)
            hand_pos=(sm_hx,sm_hy); scale=sm_scale

            if not engaged and ratio < PINCH_ON:
                engaged=True
                anchor=(hand_pos[0],hand_pos[1],scale)
            elif engaged and ratio > PINCH_OFF:
                engaged=False; anchor=None

            if engaged and anchor is not None:
                ox=(hand_pos[0]-anchor[0])*2.0
                oy=(anchor[1]-hand_pos[1])*2.0
                v_j1=response(clamp(ox,-1,1))*MAX_J1_SPEED
                v_h =response(clamp(oy,-1,1))*MAX_HEIGHT_SPEED
                sc=(scale-anchor[2])*REACH_GAIN
                v_r=response(clamp(sc,-1,1))*MAX_REACH_SPEED

                j3c,j5c=solve_reach_ik(cmd_height, clamp_reach(cmd_height, cmd_R))
                phi=math.radians(j3c+j5c)
                step=v_r*dt

                cmd_j1   = cmd_j1 + v_j1*dt
                cmd_height = clamp(cmd_height + v_h*dt + step*math.cos(phi),
                                   HEIGHT_FLOOR, HEIGHT_CEIL)
                cmd_R = clamp_reach(cmd_height, cmd_R + step*math.sin(phi))

            grip_d = thumb_pinky_distance(lm)/max(scale,1e-6)
            if not grip_pinched and grip_d < GRIP_PINCH_ON:
                grip_pinched = True
                grip_closing = not grip_closing
                print(f"[gripper] {'CLOSE' if grip_closing else 'OPEN'}")
            elif grip_pinched and grip_d > GRIP_PINCH_OFF:
                grip_pinched = False
    else:
        engaged=False; anchor=None
        sm_hx=sm_hy=sm_scale=None

    grip_target = GRIP_CLOSED if grip_closing else GRIP_OPEN
    gripper_pos += GRIP_SMOOTH*(grip_target - gripper_pos)
    if abs(gripper_pos-grip_target) < 0.5:
        gripper_pos = grip_target

    anim_offset = 0.0
    if now < anim_until:
        if anim_kind == "wake":
            phase = (now - anim_start) / WAKE_BOB_S
            anim_offset = math.sin(phase * 2*math.pi) * WAKE_BOB_MM
        else:
            phase = (now - anim_start) / SLEEP_BOB_S
            decay = (1.0 - phase)
            anim_offset = math.sin(phase * 3*math.pi) * SLEEP_BOB_MM * decay

    eff_h = cmd_height + anim_offset
    tgt_h, tgt_r = eff_h, clamp_reach(eff_h, cmd_R)
    j3_raw,j5_raw = solve_reach_ik(tgt_h, tgt_r)

    if sm_j1 is None:
        sm_j1,sm_j3,sm_j5 = cmd_j1,j3_raw,j5_raw
    sm_j1 += OUT_SMOOTH*(cmd_j1 - sm_j1)
    sm_j3 += OUT_SMOOTH*(j3_raw - sm_j3)
    sm_j5 += OUT_SMOOTH*(j5_raw - sm_j5)
    set_state(sm_j1,sm_j3,0.0,sm_j5,gripper_pos)

    hw_tick(now, v_j1, sm_j3, gripper_pos)

    # ── Visualisation overlays ──────────────────────────────────────────────────
    animating = now < anim_until
    if res.multi_hand_landmarks:
        wx,wy=int(lm[0].x*cw),int(lm[0].y*h)
        if fist_now:
            cv2.circle(cam,(wx,wy),30,(40,48,70),2,cv2.LINE_AA)
            if fist_start is not None:
                frac=clamp((now-fist_start)/FIST_HOLD_S,0.0,1.0)
                cv2.ellipse(cam,(wx,wy),(30,30),-90,0,360*frac,(80,180,255),3,cv2.LINE_AA)
                label="hold..." if frac<1 else "RELEASE"
            else:
                label="fist"
            cv2.putText(cam,label,(wx-26,wy-38),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(80,180,255),2,cv2.LINE_AA)
        elif not frozen and not animating and hand_pos is not None:
            hp=(int(hand_pos[0]*cw),int(hand_pos[1]*h))
            if engaged and anchor is not None:
                ap=(int(anchor[0]*cw),int(anchor[1]*h))
                dz=int(DEADZONE*(h/2))
                cv2.circle(cam,ap,dz,(60,70,90),1,cv2.LINE_AA)
                cv2.circle(cam,ap,int(h/2),(40,48,64),1,cv2.LINE_AA)
                moving=(abs(v_j1)+abs(v_h)+abs(v_r))>1
                col=(80,255,120) if moving else (90,100,120)
                cv2.line(cam,ap,hp,col,2,cv2.LINE_AA)
                cv2.circle(cam,hp,9,col,2,cv2.LINE_AA)
                cv2.circle(cam,ap,4,(120,130,150),-1,cv2.LINE_AA)
                cv2.putText(cam,"ENGAGED",(ap[0]-30,ap[1]-int(h/2)-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,255,120),2,cv2.LINE_AA)
                bx=hp[0]+22
                cv2.rectangle(cam,(bx,hp[1]-40),(bx+8,hp[1]+40),(50,58,76),1,cv2.LINE_AA)
                fillv=int(clamp(v_r/MAX_REACH_SPEED,-1,1)*40)
                rc=(80,200,255) if abs(v_r)>1 else (90,100,120)
                cv2.rectangle(cam,(bx,hp[1]),(bx+8,hp[1]-fillv),rc,-1)
                cv2.putText(cam,"FWD" if v_r>1 else ("BACK" if v_r<-1 else "tool"),
                            (bx-6,hp[1]-46),cv2.FONT_HERSHEY_SIMPLEX,0.4,rc,1,cv2.LINE_AA)
            else:
                cv2.circle(cam,hp,9,(120,130,150),2,cv2.LINE_AA)
                cv2.putText(cam,"pinch to engage",(hp[0]-50,hp[1]-16),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(120,130,150),1,cv2.LINE_AA)

            tp=(int(lm[4].x*cw),int(lm[4].y*h)); pk=(int(lm[20].x*cw),int(lm[20].y*h))
            gcol=(80,200,120) if grip_closing else (90,100,120)
            cv2.line(cam,tp,pk,gcol,1,cv2.LINE_AA)
            cv2.circle(cam,pk,6,gcol,2,cv2.LINE_AA)
            cv2.putText(cam,"GRIP" if grip_closing else "grip",
                        (pk[0]+8,pk[1]+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,gcol,1,cv2.LINE_AA)
    else:
        pass

    if animating and anim_kind=="wake":
        cv2.putText(cam,"LISTENING",(cw//2-110,54),
                    cv2.FONT_HERSHEY_SIMPLEX,0.95,(80,220,120),2,cv2.LINE_AA)
    elif animating and anim_kind=="sleep":
        cv2.putText(cam,"POWERING DOWN",(cw//2-160,54),
                    cv2.FONT_HERSHEY_SIMPLEX,0.85,(60,80,230),2,cv2.LINE_AA)
    elif frozen:
        cv2.putText(cam,"FROZEN  -  hold fist to resume",(cw//2-210,54),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(60,80,230),2,cv2.LINE_AA)

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    px=cw; cv2.rectangle(frame,(px,0),(w,h),(16,18,22),-1); cv2.line(frame,(px,0),(px,h),(32,38,52),1)
    def L(t,x,y,c=(150,155,165),s=0.40): cv2.putText(frame,t,(x,y),cv2.FONT_HERSHEY_SIMPLEX,s,c,1,cv2.LINE_AA)
    L("TELECONTROL",px+10,26,(80,180,255),0.46)
    if frozen:                       st,sc="FROZEN",(60,80,230)
    elif animating and anim_kind=="wake":  st,sc="LISTENING",(80,220,120)
    elif engaged:                    st,sc="ENGAGED",(80,220,120)
    else:                            st,sc="active",(90,95,110)
    L(st,px+10,50,sc,0.5)
    if now < fist_lock_until:
        L(f"fist lock {fist_lock_until-now:0.1f}s",px+120,50,(90,90,120),0.34)

    L("Velocity",px+10,84,(55,70,100),0.42)
    L(f"J1   : {v_j1:+6.1f} d/s",px+14,104,(80,255,120) if abs(v_j1)>1 else (90,95,110),0.34)
    L(f"hgt  : {v_h:+6.1f} mm/s",px+14,124,(80,255,120) if abs(v_h)>1 else (90,95,110),0.34)
    L(f"fwd  : {v_r:+6.1f} mm/s",px+14,144,(80,255,120) if abs(v_r)>1 else (90,95,110),0.34)

    L("Commanded pose",px+10,178,(55,70,100),0.42)
    turns = int(cmd_j1 // 360)
    wrapped = cmd_j1 % 360.0
    L(f"J1 : {cmd_j1:+7.1f}",px+14,198,(120,180,230),0.36)
    L(f"  = {turns:+d} turns {wrapped:5.1f}",px+14,214,(90,110,150),0.32)
    L(f"hgt: {cmd_height:5.0f}mm",px+14,232,(120,180,230),0.36)
    L(f"R  : {cmd_R:5.0f}mm",px+14,250,(120,180,230),0.36)

    L("Solved joints",px+10,282,(55,70,100),0.42)
    j3v,j5v=solve_reach_ik(cmd_height, clamp_reach(cmd_height, cmd_R))
    for i,(n,vv) in enumerate([("J1",cmd_j1),("J3",j3v),("J5",j5v)]):
        L(f"{n}: {vv:+6.1f}",px+14,302+i*20,(120,180,230),0.36)

    L("Hardware",px+10,376,(55,70,100),0.42)
    hw_connected = hw_link is not None and hw_link.is_open
    hcol = (230,80,80) if hw_fault else ((80,220,120) if hw_connected else (90,95,110))
    L(f"{hw_port or 'no port'} : {'connected' if hw_connected else 'disconnected'}",
      px+14,396,hcol,0.34)
    acol=(80,220,120) if hw_armed else (90,95,110)
    L("ARMED" if hw_armed else "sim-only (H to arm)",px+14,414,acol,0.36)
    L("elbow (D) not wired up",px+14,432,(90,95,110),0.30)

    L(f"max J1 {MAX_J1_SPEED:.0f} d/s",px+10,h-44,(70,80,100),0.32)
    gtxt="OPEN" if gripper_pos>=GRIP_OPEN-1 else ("CLOSED" if gripper_pos<=GRIP_CLOSED+1 else f"{gripper_pos:.0f}")
    L(f"Gripper: {gtxt}",px+10,h-62,(80,200,120) if gripper_pos>GRIP_CLOSED+1 else (200,120,80),0.40)
    L("hold fist=freeze  thumb+pinky=grip",px+10,h-26,(60,70,90),0.28)
    L("Space=toggle Q=quit  G/O/P grip  H=arm X=clear fault",px+10,h-10,(60,70,90),0.30)

    cv2.imshow("Telecontrol — Thor",frame)
    k=cv2.waitKey(1)&0xFF
    if   k==ord('q'): break
    elif k==ord('r'):
        cmd_j1=0.0; cmd_height=SHOULDER_H+10.0; cmd_R=160.0
        engaged=False; anchor=None; grip_closing=False; grip_pinched=False
        fist_held=False; fist_start=None; fist_lock_until=0.0; anim_until=0.0
        print("[reset]")
    elif k==ord(' '): frozen=not frozen; print(f"[freeze]{frozen}")
    elif k==ord('['): MAX_J1_SPEED=max(20,MAX_J1_SPEED-10); print(f"[spd]{MAX_J1_SPEED}")
    elif k==ord(']'): MAX_J1_SPEED=min(250,MAX_J1_SPEED+10); print(f"[spd]{MAX_J1_SPEED}")
    elif k==ord('g'):
        grip_closing = not grip_closing
        print(f"[gripper] {'CLOSING' if grip_closing else 'OPENING'}")
    elif k==ord('o'): grip_closing=False; print("[gripper] OPEN")
    elif k==ord('p'): grip_closing=True;  print("[gripper] CLOSE")
    elif k==ord('h'):
        if hw_link is None or not hw_link.is_open:
            print("[hw] no serial connection — nothing to arm")
        elif hw_fault:
            print("[hw] fault latched — clear with X before arming")
        else:
            hw_armed = not hw_armed
            if hw_armed:
                # Nothing to re-zero: velocity control has no position/backlog
                # to catch up on — arming just starts sending whatever the
                # current commanded speed already is (often 0, if not engaged).
                print("[hw] ARMED — elbow (D) is not wired up in this firmware at all")
            else:
                print("[hw] disarmed")
    elif k==ord('x'):
        if hw_fault:
            hw_fault = False
            print("[hw] fault cleared — arm again with H")
        else:
            print("[hw] no fault to clear")

cap.release(); cv2.destroyAllWindows()
if hw_link is not None:
    hw_link.close()
print("Done.")
