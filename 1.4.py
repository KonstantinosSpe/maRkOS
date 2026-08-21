"""
fingertip_clutch_velocity.py — Pinch-clutch VELOCITY control
-------------------------------------------------------------
Combines reliable pinch engagement with velocity (speed) control:

  PINCH thumb+index            → ENGAGE. The spot where you pinched (position
                                 AND hand size) becomes the neutral "center".
  Move hand AWAY from center   → drift at a SPEED proportional to offset:
                                   left/right            → J1 base rotation (θ)
                                   up/down               → height (z)
                                   toward/away from cam  → FORWARD/BACK along the
                                                           direction the gripper
                                                           is pointing (tool axis)
  Small offset = slow creep (fine).  Large offset = fast travel.
  Push hand TOWARD camera = tip moves the way the gripper aims; pull back =
  retreats along that same line. θ and z stay cylindrical as before. (The push
  is detected from the hand's apparent size — a true 3rd axis.)
  Release pinch                → STOP DEAD instantly.
  Engage uses a SCALE-NORMALISED pinch, so it works near or far from the camera.

  GRIPPER:
  PINCH thumb+PINKY            → TOGGLES the gripper. 1st pinch closes,
                                 2nd pinch opens, 3rd closes... Releasing
                                 does nothing; only a fresh pinch flips it.
  This is independent of the thumb+index clutch above.

  FIST (v1.2):
  HOLD A FIST (~1s)            → toggle FREEZE. A ring fills around your hand
                                 while you hold so you know when it commits;
                                 hold again to resume. The fist overrides every
                                 other gesture. Each toggle plays a short
                                 automated motion (control locked) so you can
                                 SEE the change: an up/down NOD when it turns ON
                                 (listening) and a damped POWER-DOWN wobble that
                                 settles when it turns OFF (frozen).

Why this is better:
  - Velocity control: rest your hand near center = no motion, no jitter.
  - Pinch engagement is the most reliable thing vision tracks.
  - You never run out of hand space — it's speed, not position.

Controls: Q=quit  R=reset pose  Space=freeze  [ ]=max speed
          G=toggle grip  O=open  P=close   (keyboard mirrors the gesture)

Requirements: pip install opencv-python "mediapipe==0.10.9" numpy websockets
Run: python fingertip_clutch_velocity.py  then open thor_simulator.html → Connect.
"""

import cv2, mediapipe as mp, numpy as np
import math, json, asyncio, threading, websockets, time

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
# J1 is a STEPPER — continuous rotation, no angle limit (multi-turn allowed)
# Full vertical envelope: from low (arm folded down) to high (arm up)
WS_HEIGHT = (SHOULDER_H-260, SHOULDER_H+250)
# Full reach: from nearly folded to nearly straight
WS_REACH  = (REACH_MIN+5, REACH_MAX-5)

# ── Velocity tuning ───────────────────────────────────────────────────────────
MAX_J1_SPEED     = 80.0     # deg/s at full hand offset (left/right → J1)
MAX_HEIGHT_SPEED = 160.0    # mm/s   (up/down → height z)
MAX_REACH_SPEED  = 140.0    # mm/s   (forward/back along the tool axis)
DEADZONE         = 0.04     # hand offset (normalised) ignored around center
CURVE_POW        = 2.0      # response curve (higher = finer near center)
REACH_GAIN       = 8.0      # how strongly hand-scale change maps to fwd velocity
INPUT_SMOOTH     = 0.40     # EMA on hand x/y inputs (lower = smoother, more lag)
SCALE_SMOOTH     = 0.18     # heavier EMA on the depth axis (it's the noisiest)

# Pinch engage thresholds — now a RATIO of (thumb-index gap / hand size), so
# engaging works the same whether your hand is near or far from the camera.
PINCH_ON  = 0.38   # ratio below this → engage
PINCH_OFF = 0.55   # ratio above this → release (hysteresis gap between them)

# IK elbow share
J3_SHARE = 0.55

# ── State ─────────────────────────────────────────────────────────────────────
cmd_j1     = 0.0
cmd_height = SHOULDER_H + 10.0   # height z (mm, absolute)
cmd_R      = 160.0               # radial distance from base axis (mm)

engaged   = False
frozen    = True     # start DEACTIVATED — fist once to activate (then it nods)
anchor    = None     # (hx, hy, hand_scale) captured at engage

# J6 = GRIPPER SERVO (not a spin). 0 = closed, 90 = open (tune GRIP_OPEN later)
GRIP_CLOSED = 0.0
GRIP_OPEN   = 90.0
gripper_pos = GRIP_OPEN        # current servo angle (start open)

# ── Gripper gesture: thumb (4) + pinky (20) pinch ─────────────────────────────
# Pinch them together → toggle. Hysteresis stops flicker; smoothing animates the
# jaws. Thresholds are a RATIO of (thumb-pinky gap / hand size) so they work the
# same whether the hand is near or far from the camera.
GRIP_PINCH_ON  = 0.45     # ratio below this → toggle
GRIP_PINCH_OFF = 0.70     # ratio above this → re-arm
GRIP_SMOOTH    = 0.30     # 0..1, how fast the jaws move toward the target
grip_closing   = False    # latched target: True = jaws closed, False = open
grip_pinched   = False    # is thumb+pinky CURRENTLY touching (edge detector)

# ── Fist = freeze toggle (HOLD to confirm) ────────────────────────────────────
# HOLD a fist for FIST_HOLD_S seconds → toggle FROZEN. Hold again to resume.
# The fist OVERRIDES every other gesture. A short debounce stops double-fires.
# Each toggle plays a brief automated motion (control locked) so you can SEE it:
#   turning ON  → an up/down "nod"        (it's listening)
#   turning OFF → a damped "power-down" wobble that settles (it stopped)
FIST_HOLD_S     = 1.0     # s, how long the fist must be held to toggle
FIST_COOLDOWN   = 0.6     # s, short debounce so one hold = one toggle
WAKE_BOB_S      = 2.0     # s, duration of the wake-up nod
WAKE_BOB_MM     = 28.0    # mm, amplitude of the wake nod
SLEEP_BOB_S     = 1.4     # s, duration of the power-down wobble
SLEEP_BOB_MM    = 34.0    # mm, starting amplitude of the power-down wobble
fist_held       = False   # latched: this fist already counted (wait for release)
fist_start      = None    # time.time() the current continuous fist began
fist_lock_until = 0.0     # time.time() until which fist toggling is disabled
anim_until      = 0.0     # time.time() until which the toggle animation runs
anim_start      = 0.0
anim_kind       = "wake"  # "wake" or "sleep" — which motion is playing

# Output smoothing (low-pass) on solved joint angles to keep motion fluid
OUT_SMOOTH = 0.18    # 0..1, lower = smoother/slower, higher = snappier
sm_j1 = sm_j3 = sm_j5 = None

# Input smoothing (low-pass) on the raw hand signals — kills tracking jitter,
# especially on the noisy hand-scale (depth) axis, before it drives anything.
sm_hx = sm_hy = sm_scale = None

# ── WebSocket ─────────────────────────────────────────────────────────────────
_clients=set(); _lock=threading.Lock()
_state={"angles":[0,0,0,0,0,0],"gripper":0.0,"mode":"CLUTCH-VEL"}
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
    """Solve j3,j5 (deg, angles from VERTICAL) to match the SIMULATOR geometry:
         r = A*sin(j3) + B*sin(j3+j5)
         y = A*cos(j3) + B*cos(j3+j5)
       This is the actual forward kinematics the 3D arm uses."""
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
    """height = gripper height (mm, absolute). reach = horizontal distance (mm).
       Picks the elbow solution that best keeps J3 within limits."""
    r = reach
    y = height - SHOULDER_H          # vertical relative to shoulder
    best=None
    for e in (+1,-1):
        j3,j5=_ik_planar(r,y,e)
        # penalise out-of-limit J3 heavily, then prefer small |J3|
        pen = (abs(j3)-90)*10 if abs(j3)>90 else 0
        score = pen + abs(j3)
        if best is None or score<best[0]:
            best=(score,j3,j5)
    j3,j5=best[1],best[2]
    return clamp(j3,*LIMITS["J3"]), clamp(j5,*LIMITS["J5"])

def clamp_reach(height, R):
    """Clamp a radial distance R (mm from the base axis) to what's actually
    reachable at this height. The straight-line distance from the shoulder to the
    tip, d = hypot(R, dy), must lie within the arm's span [REACH_MIN, REACH_MAX].
    We also keep R >= R_FLOOR so the arm never fully straightens into the vertical
    singularity (where tool-axis 'forward' would lose all radial authority)."""
    dy = height - SHOULDER_H
    hi_sq = REACH_MAX*REACH_MAX - dy*dy
    lo_sq = REACH_MIN*REACH_MIN - dy*dy
    R_hi = math.sqrt(hi_sq) if hi_sq > 0 else 0.0
    R_lo = math.sqrt(lo_sq) if lo_sq > 0 else 0.0
    R_lo = max(R_lo, R_FLOOR)          # never collapse to vertical
    if R_hi < R_lo: R_hi = R_lo
    return clamp(R, R_lo, R_hi)

def pinch_distance(lm):
    return math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y)

def hand_scale(lm):
    """Apparent hand size: wrist (0) to middle-finger knuckle (9). Grows as the
    hand nears the camera and shrinks as it moves away — our depth proxy, used
    both to normalise the pinch and to drive the radial (in/out) axis."""
    return math.hypot(lm[0].x-lm[9].x, lm[0].y-lm[9].y)

def pinch_ratio(lm):
    """Thumb-index gap relative to hand size. Scale-invariant, so a pinch reads
    the same whether the hand is close to or far from the camera."""
    s = hand_scale(lm)
    return pinch_distance(lm)/s if s > 1e-6 else 1.0

def thumb_pinky_distance(lm):
    """Distance between thumb tip (4) and pinky tip (20). Used for the gripper."""
    return math.hypot(lm[4].x-lm[20].x, lm[4].y-lm[20].y)

def is_fist(lm):
    """True when the four fingers are curled into the palm. A finger counts as
    curled when its TIP sits closer to the wrist (0) than its PIP joint does —
    orientation-independent, so it works at any hand angle. Requires all four
    (index/middle/ring/pinky) curled so a clutch/grip pinch never reads as a fist."""
    def d(i): return math.hypot(lm[i].x-lm[0].x, lm[i].y-lm[0].y)
    fingers = [(8,6),(12,10),(16,14),(20,18)]   # (tip, pip) per finger
    curled = sum(1 for tip,pip in fingers if d(tip) < d(pip)*0.92)
    return curled >= 4

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands=mp.solutions.hands; mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                     min_detection_confidence=0.7,min_tracking_confidence=0.6)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

print("fingertip_clutch_velocity.py — open thor_simulator.html -> Connect")
print("PINCH thumb+index to engage. Push hand from center = speed. Release = stop.")
print("PINCH thumb+pinky to close the gripper. Separate to open.")

last_t=time.time()

while True:
    ok,frame=cap.read()
    if not ok: break
    frame=cv2.flip(frame,1); h,w=frame.shape[:2]; cw=w-255; cam=frame[:,:cw]
    now=time.time(); dt=clamp(now-last_t,0,0.1); last_t=now

    res=hands.process(cv2.cvtColor(cam,cv2.COLOR_BGR2RGB))
    v_j1=v_h=v_r=0.0
    hand_pos=None; grip_d=None

    fist_now=False
    if res.multi_hand_landmarks:
        lm=res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(cam,res.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(35,45,55),thickness=2,circle_radius=2),
            mp_draw.DrawingSpec(color=(55,65,80),thickness=1))

        # ── FIST = freeze toggle. HOLD it for FIST_HOLD_S to flip the state.
        #    Runs even while frozen (so you can wake it) and overrides every
        #    other gesture. Open the hand to re-arm for the next hold.
        fist_now=is_fist(lm)
        if fist_now and not fist_held:
            if fist_start is None and now>=fist_lock_until:
                fist_start=now                        # begin charging the 1s hold
            if fist_start is not None and now-fist_start>=FIST_HOLD_S:
                fist_held=True                        # commit: one hold = one toggle
                fist_start=None
                frozen=not frozen
                fist_lock_until=now+FIST_COOLDOWN
                engaged=False; anchor=None            # drop any clutch in progress
                anim_start=now
                if frozen:                            # turning OFF → power-down wobble
                    anim_kind="sleep"; anim_until=now+SLEEP_BOB_S
                else:                                 # turning ON → wake nod
                    anim_kind="wake";  anim_until=now+WAKE_BOB_S
                print(f"[fist] {'OFF (frozen)' if frozen else 'ON (listening)'}")
        elif not fist_now:
            fist_held=False; fist_start=None          # re-arm once the hand opens

        # Control gestures only when awake, not mid-fist, and past the toggle anim.
        ctrl_ok = (not frozen) and (not fist_now) and (now>=anim_until)
        if not ctrl_ok:
            sm_hx=sm_hy=sm_scale=None        # re-init cleanly when control resumes
        if ctrl_ok:
            ratio=pinch_ratio(lm)                       # engage uses raw (responsive)
            # Smooth the raw hand signals before they drive motion.
            raw_hx,raw_hy=(lm[4].x+lm[8].x)/2,(lm[4].y+lm[8].y)/2
            raw_scale=hand_scale(lm)
            if sm_hx is None: sm_hx,sm_hy,sm_scale=raw_hx,raw_hy,raw_scale
            sm_hx   += INPUT_SMOOTH*(raw_hx-sm_hx)
            sm_hy   += INPUT_SMOOTH*(raw_hy-sm_hy)
            sm_scale+= SCALE_SMOOTH*(raw_scale-sm_scale)   # depth axis: smooth harder
            hand_pos=(sm_hx,sm_hy); scale=sm_scale

            if not engaged and ratio < PINCH_ON:
                engaged=True
                anchor=(hand_pos[0],hand_pos[1],scale)   # capture hand size too
            elif engaged and ratio > PINCH_OFF:
                engaged=False; anchor=None

            if engaged and anchor is not None:
                ox=(hand_pos[0]-anchor[0])*2.0     # left/right offset → J1 (θ)
                oy=(anchor[1]-hand_pos[1])*2.0     # up/down offset    → height (z)
                v_j1=response(clamp(ox,-1,1))*MAX_J1_SPEED
                v_h =response(clamp(oy,-1,1))*MAX_HEIGHT_SPEED
                # hand toward camera = push FORWARD; away = pull BACK — but along
                # the direction the gripper points, not straight out from the base.
                sc=(scale-anchor[2])*REACH_GAIN
                v_r=response(clamp(sc,-1,1))*MAX_REACH_SPEED

                # Pointer direction in the vertical plane: phi = (J3+J5) from
                # vertical. The tip advances along (sin phi, cos phi), so "forward"
                # follows wherever the tool is aimed (a tool-axis jog). θ and z
                # stay cylindrical exactly as before.
                j3c,j5c=solve_reach_ik(cmd_height, clamp_reach(cmd_height, cmd_R))
                phi=math.radians(j3c+j5c)
                step=v_r*dt

                cmd_j1   = cmd_j1 + v_j1*dt           # stepper: continuous
                # Height z gets the up/down command PLUS the vertical part of the
                # forward push; clamped between floor and the singularity-safe ceiling.
                cmd_height = clamp(cmd_height + v_h*dt + step*math.cos(phi),
                                   HEIGHT_FLOOR, HEIGHT_CEIL)
                # Radial R gets the radial part of the forward push, then clamped
                # to what's reachable at this height.
                cmd_R = clamp_reach(cmd_height, cmd_R + step*math.sin(phi))

            # ── Gripper gesture (independent of the clutch above) ─────────────
            # Each fresh thumb+pinky pinch TOGGLES the gripper: close, then open,
            # then close... Releasing does nothing — only a new pinch flips it.
            grip_d = thumb_pinky_distance(lm)/max(scale,1e-6)
            if not grip_pinched and grip_d < GRIP_PINCH_ON:
                grip_pinched = True
                grip_closing = not grip_closing      # toggle on the rising edge
                print(f"[gripper] {'CLOSE' if grip_closing else 'OPEN'}")
            elif grip_pinched and grip_d > GRIP_PINCH_OFF:
                grip_pinched = False                 # armed for the next pinch
    else:
        # Hand left the frame — drop the clutch and re-init the smoother so we
        # don't glide from a stale position when it comes back.
        engaged=False; anchor=None
        sm_hx=sm_hy=sm_scale=None

    # Drive the gripper toward its target every frame (smooth jaw motion).
    grip_target = GRIP_CLOSED if grip_closing else GRIP_OPEN
    gripper_pos += GRIP_SMOOTH*(grip_target - gripper_pos)
    if abs(gripper_pos-grip_target) < 0.5:
        gripper_pos = grip_target

    # ── Toggle animation: a brief automated motion so you can SEE the state flip.
    # Applied as a temporary offset to the solved height — cmd_height itself is
    # untouched, so the arm returns to its real commanded pose when it ends.
    #   wake  → one smooth up/down nod
    #   sleep → a damped wobble that settles to a stop (looks like powering off)
    anim_offset = 0.0
    if now < anim_until:
        if anim_kind == "wake":
            phase = (now - anim_start) / WAKE_BOB_S            # 0..1
            anim_offset = math.sin(phase * 2*math.pi) * WAKE_BOB_MM
        else:  # sleep
            phase = (now - anim_start) / SLEEP_BOB_S           # 0..1
            decay = (1.0 - phase)                              # shrink to 0
            anim_offset = math.sin(phase * 3*math.pi) * SLEEP_BOB_MM * decay

    eff_h = cmd_height + anim_offset
    tgt_h, tgt_r = eff_h, clamp_reach(eff_h, cmd_R)
    j3_raw,j5_raw = solve_reach_ik(tgt_h, tgt_r)

    # (No straighten-at-top blend any more: R_FLOOR keeps the arm slightly bent,
    #  so it never reaches the vertical singularity and never needs forcing.)

    # Low-pass filter the outputs for smooth motion
    if sm_j1 is None:
        sm_j1,sm_j3,sm_j5 = cmd_j1,j3_raw,j5_raw
    sm_j1 += OUT_SMOOTH*(cmd_j1 - sm_j1)
    sm_j3 += OUT_SMOOTH*(j3_raw - sm_j3)
    sm_j5 += OUT_SMOOTH*(j5_raw - sm_j5)
    set_state(sm_j1,sm_j3,0.0,sm_j5,gripper_pos)

    # ── Visualisation overlays ──────────────────────────────────────────────────
    animating = now < anim_until
    if res.multi_hand_landmarks:
        wx,wy=int(lm[0].x*cw),int(lm[0].y*h)
        if fist_now:
            # Fist is the override. Draw a ring that FILLS as you hold (1s).
            cv2.circle(cam,(wx,wy),30,(40,48,70),2,cv2.LINE_AA)
            if fist_start is not None:
                frac=clamp((now-fist_start)/FIST_HOLD_S,0.0,1.0)
                cv2.ellipse(cam,(wx,wy),(30,30),-90,0,360*frac,(80,180,255),3,cv2.LINE_AA)
                label="hold..." if frac<1 else "RELEASE"
            else:
                label="fist"     # locked out / already counted
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
                # In/out (R) cue: a vertical bar that fills with the push velocity.
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

            # Gripper gesture cue: line between thumb tip and pinky tip
            tp=(int(lm[4].x*cw),int(lm[4].y*h)); pk=(int(lm[20].x*cw),int(lm[20].y*h))
            gcol=(80,200,120) if grip_closing else (90,100,120)
            cv2.line(cam,tp,pk,gcol,1,cv2.LINE_AA)
            cv2.circle(cam,pk,6,gcol,2,cv2.LINE_AA)
            cv2.putText(cam,"GRIP" if grip_closing else "grip",
                        (pk[0]+8,pk[1]+4),cv2.FONT_HERSHEY_SIMPLEX,0.4,gcol,1,cv2.LINE_AA)

    # Big centred banners for the special states
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
    L("CLUTCH VELOCITY",px+10,26,(80,180,255),0.46)
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

    L(f"max J1 {MAX_J1_SPEED:.0f} d/s",px+10,h-44,(70,80,100),0.32)
    gtxt="OPEN" if gripper_pos>=GRIP_OPEN-1 else ("CLOSED" if gripper_pos<=GRIP_CLOSED+1 else f"{gripper_pos:.0f}")
    L(f"Gripper: {gtxt}",px+10,h-62,(80,200,120) if gripper_pos>GRIP_CLOSED+1 else (200,120,80),0.40)
    L("hold fist=freeze  thumb+pinky=grip",px+10,h-26,(60,70,90),0.28)
    L("Space=toggle Q=quit  G/O/P grip",px+10,h-10,(60,70,90),0.30)

    cv2.imshow("Clutch Velocity — Thor",frame)
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
    elif k==ord('g'):                 # keyboard mirrors the gesture state
        grip_closing = not grip_closing
        print(f"[gripper] {'CLOSING' if grip_closing else 'OPENING'}")
    elif k==ord('o'): grip_closing=False; print("[gripper] OPEN")
    elif k==ord('p'): grip_closing=True;  print("[gripper] CLOSE")

cap.release(); cv2.destroyAllWindows(); print("Done.")
