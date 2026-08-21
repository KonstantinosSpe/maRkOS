/*
  thor_teleop_firmware.ino
  =========================
  REBUILT to match Markos_basic.ino — the user's own, previously-working
  firmware for this exact robot. That sketch drives the motors with direct
  digitalWrite()/delayMicroseconds() bit-banging, no stepper library, and
  moved the real arm. Every previous version of THIS file used the
  AccelStepper library instead and, across extensive testing, never
  produced confirmed physical motion — same pins, no enable pin either way,
  so the AccelStepper layer itself is the prime suspect. This version drops
  it entirely and uses the same proven mechanism as Markos_basic.ino.

  JOG (small bounded nudges) and CAL (passive recording of net steps moved)
  cover A / B(+C mirrored) / D / E, plus the gripper servo. MOVE (explicit
  signed step counts, for continuous gesture-driven streaming from Python)
  now covers A / B / D / E. D was excluded from MOVE for most of this
  project's life ("never move D, full stop") — that guarantee has been
  deliberately lifted by explicit request, not by accident. This firmware
  has no concept of "armed/disarmed": it will execute any MOVE D it
  receives exactly like MOVE A/B/E. All of the caution lives on the Python
  side (mark1os.py's HW_CALIBRATED_D, which starts False) — if that ever
  changes, this firmware will move D on command with no extra gate here.

  Every JOG move is BLOCKING and bounded by construction: a for-loop over a
  fixed step count, checking the relevant limit switch every single pulse
  and stopping early if it trips. There is no async "ramp toward a target"
  state that could get stuck running — the bug class from the AccelStepper
  version (repeated commands extending a burst indefinitely) cannot occur
  here, because there is nothing left running once a JOG call returns.

  ────────────────────────────────────────────────────────────────────────
  JOG — small, precise nudges. Blocking, bounded, checked against limits
  every pulse.
  ────────────────────────────────────────────────────────────────────────
    JOG A+   JOG A-      nudge base by jogStepSteps raw motor steps
    JOG B+   JOG B-      nudge upper elbow (C mirrors automatically)
    JOG D+   JOG D-      nudge elbow
    JOG E+   JOG E-      nudge lower rotation
    JOG G+   JOG G-      nudge the gripper servo by JOG_GRIP_DEG degrees
    JOGSTEP <n>          set jogStepSteps to n (no reflash needed to tune)
    SPEED <vmin> <vmax>  set the ramp speed range in microseconds (bigger =
                          slower; vmin=start/end, vmax=cruise) — live, no reflash

  Prints "[fw] JOG A done, moved N" on completion — N is the actual signed
  step count completed (may be less than requested if a limit switch
  stopped it early). This confirms the command was received and a pulse
  sequence was issued; it does NOT by itself prove the motor physically
  turned — that still depends on the driver actually being powered and
  wired correctly, which is a separate, physical thing to verify by eye.

  ────────────────────────────────────────────────────────────────────────
  CAL — passive recording of net distance traveled, in exact steps.
  ────────────────────────────────────────────────────────────────────────
    CAL A   CAL B   CAL D   CAL E    select an axis, zero its step counter
    CAL STOP                         report the net step count since CAL <axis>

  CAL doesn't move anything itself — jog the joint with JOG while it's
  recording, then read off the total. Workflow: mark a start point, send
  "CAL A" (or B/D/E), JOG to wherever you're measuring to — a limit switch,
  a protractor-marked angle, or back to the exact start for a full 360° —
  then send "CAL STOP". stepsPerDeg is then reported_steps / known_degrees.

  ────────────────────────────────────────────────────────────────────────
  HOME? / HOME <axis> — the homing beacons (pins 25/27/52, found via
  pin_scanner.ino). Not the same thing as LIM_A/B/D/E: those only stop a
  JOG/MOVE if hit mid-travel; these are dedicated beacons for finding a
  true reference position. A has no beacon yet (not needed — still
  disarmed on the Python side).
    HOME?        status only — prints each beacon's current state, moves
                 nothing. Use this to confirm wiring/polarity by hand
                 before trusting HOME <axis> to move anything.
    HOME D/B/E   actually drives that axis to its beacon (bounded search,
                 see doHome()) and zeros its position counter there.
                 D has NO working hardware limit switch (LIM_D never
                 reliably triggers) — its search and every other D move
                 (JOG, MOVE) are clamped in software instead
                 (D_SOFT_MIN/MAX), derived from a hand-tested safe range.
  ────────────────────────────────────────────────────────────────────────
  FIRST BRING-UP CHECKLIST — do this BEFORE trusting any of it near people:
  ────────────────────────────────────────────────────────────────────────
  1. Direction sign is unverified per axis — JOG one small step at a time,
     watch, confirm it goes the way you expect before trusting it.
  2. Limit switch polarity: A/B/D assumed LOW = triggered (INPUT_PULLUP).
     E is assumed INVERTED (HIGH = triggered) per the original hardware's
     own notes. Verify by hand-triggering each switch and watching for
     "[fw] LIMIT" before doing anything else.
  3. GRIP_PIN is 10, matching Markos_basic.ino (the earlier firmware used
     an unverified guess of 11 — this is the corrected, evidence-based
     value). Still worth a visual confirmation of the wiring before trusting it.
  4. jogStepSteps and VMIN_US/VMAX_US (the ramp speed range, defaulted to
     Markos_basic.ino's own proven values) are both tunable live, no
     reflash needed — JOGSTEP <n> for step count, SPEED <vmin> <vmax> for
     the ramp (bigger microseconds = slower; vmin=start/end, vmax=cruise).
  ────────────────────────────────────────────────────────────────────────
*/

#include <Servo.h>

// ── Pins (from thor_control.py's axis table, confirmed by Markos_basic.ino) ─
const int STEP_A = 28, DIR_A = 36, LIM_A = 42;          // upper rotation
const int STEP_B = 26, DIR_B = 34, LIM_B = 44;          // upper elbow motor 1
const int STEP_C = 24, DIR_C = 32;                      // upper elbow motor 2 (mirrors B)
const int STEP_D = 22, DIR_D = 30, LIM_D = 48;          // elbow
const int STEP_E = 23, DIR_E = 31, LIM_E = 49;          // lower rotation

const bool MIRROR_C     = true;   // confirmed in Markos_basic.ino
const bool LIM_E_INVERTED = false; // flipped from true: the original "inverted" note was about
                                    // Grbl's own status-word correction, not necessarily raw
                                    // digitalRead() behavior here — with true, E read as
                                    // permanently triggered in both directions on real hardware
const bool E_DIR_FLIP = true;      // A and E were observed rotating opposite ways for what
                                    // should be the same "+" sense — flips E's sign (in both
                                    // JOG and MOVE) so + means the same rotational direction
                                    // on both axes. Applied here, at the source, so it's
                                    // consistent no matter how E is driven.
const bool B_DIR_FLIP = false;     // set true if JOG B+/MOVE B<positive> moves the shoulder
                                    // opposite to what the height/reach IK expects — B was
                                    // only armed for live MOVE this session, direction vs.
                                    // the IK's sign convention has not been isolated-tested
const bool D_DIR_FLIP = false;     // same as B_DIR_FLIP but for D (elbow/J5) — D has never
                                    // been live-driven before this session at all

const int GRIP_PIN = 10;   // matches Markos_basic.ino (corrected from an earlier guess of 11)

// ── Homing beacons — separate from LIM_A/B/D/E above, which only stop a JOG/
// MOVE early if hit mid-travel. These are dedicated optical endstops found
// via pin_scanner.ino, driven by HOME? (status only) and HOME <axis>
// (actually seeks and zeros) further down.
const int HOME_E = 25;   // E (base twist)
const int HOME_B = 27;   // B/C (distal hinge)
const int HOME_D = 52;   // D (proximal hinge)
const bool HOME_E_ACTIVE_HIGH = true;    // confirmed by hand: HIGH = at home
const bool HOME_B_ACTIVE_HIGH = false;   // confirmed by hand: LOW  = at home
const bool HOME_D_ACTIVE_HIGH = true;    // confirmed by hand: HIGH = at home
// A (mid-arm twist) has no beacon yet — not needed yet either, since A is
// still disarmed on the Python side (HW_CALIBRATED_A = False). Add one here
// the same way if that ever changes.

// ── D's software position clamp ──────────────────────────────────────────
// LIM_D (pin 48, in the original limit-switch block above) does not
// reliably trigger — confirmed by hand-testing, it never fired even well
// past where home is. So unlike B and E, D has NO working hardware backstop
// against overtravel. These bounds are D's only real protection, set to
// the FULL hand-tested safe range (drove to -1700 and +1500 steps from
// home, both with no incident) — not shaved down with extra margin. An
// earlier version subtracted ~100 steps of margin here, but posD is just
// an open-loop step count with no absolute reference except a successful
// HOME D — any drift since the last one (a lost step under load, a prior
// failed search returning to a not-quite-true start) eats directly into
// whatever margin is subtracted, and in practice caused HOME D to give up
// early rather than actually add safety. Using the full tested range
// instead gives real drift tolerance while still never exceeding what was
// hand-verified safe.
const long D_SOFT_MIN = -1700;
const long D_SOFT_MAX = 1500;

// ── Bounded search distances for HOME <axis> ─────────────────────────────
// D's search request is deliberately much bigger than D_SOFT_MIN/MAX's full
// span (3200) — NOT because D is allowed to travel that far, but so the
// soft limit (an absolute position check) is always what stops the search,
// never the requested step count itself. If this were set equal to
// D_SOFT_MAX like an earlier version had it, the search would only cover
// its full intended distance when posD happened to already be exactly 0 —
// any other starting position (completely normal after manual JOG testing)
// would hit the absolute soft limit boundary after far fewer steps than
// intended, cutting the search short well before it should give up. The
// soft limit itself is unchanged and still the real safety bound; this
// only affects how far the search is willing to LOOK.
const long HOME_SEARCH_D_PLUS  = 3500;
const long HOME_SEARCH_D_MINUS = 3500;
// B's is a generous guess since its beacon sits at an edge and LIM_B works
// as a real backstop either way. E's covers a bit over one full rotation
// (2200 steps = 360°), since direction doesn't matter for a beacon on a
// continuous twist.
const long HOME_SEARCH_B       = 2000;
const long HOME_SEARCH_E       = 2300;

// ── Ramp speed range (microseconds between pulse edges). BIGGER = SLOWER.
// Starts/ends slow, cruises fast. Was Markos_basic.ino's own 2200/900
// default; raised to the mark1os.py GUI's "slow" preset after confirming
// on real hardware it tracks noticeably better — likely because every
// limit/beacon/soft-limit check happens once per step, and the faster
// default left less margin to catch a trigger cleanly before overshooting.
// Still adjustable live with no reflash via SPEED <vmin> <vmax>.
int VMIN_US = 3000;
int VMAX_US = 1800;
const float RAMP_FRAC = 0.3;   // fraction of the move spent accelerating (and decelerating)

// Homing searches run even slower than normal JOG/MOVE speed. D in
// particular can lose steps under load, and which direction is "uphill"
// depends on D's current angle, not a fixed sign — so rather than guess
// which direction needs more margin, doHome() slows every homing search
// down uniformly for more torque margin regardless of direction.
const int HOME_VMIN_US = 4200;
const int HOME_VMAX_US = 2600;

long jogStepSteps = 200;         // raw steps per JOG press, runtime-adjustable via JOGSTEP
const long MAX_JOG_STEPS = 20000;

const float JOG_GRIP_DEG = 3.0;  // gripper degrees per JOG G+/G- press

Servo gripper;
int gripAngle = 90;

// ── Net step counters per axis, for CAL — plain accumulation, no library ────
long posA = 0, posB = 0, posD = 0, posE = 0;

bool calMode = false;
int  calAxis = -1;   // 0=A 1=B 2=D 3=E

String rxLine = "";

void setup() {
  Serial.begin(115200);

  int outs[] = {STEP_A, DIR_A, STEP_B, DIR_B, STEP_C, DIR_C, STEP_D, DIR_D, STEP_E, DIR_E};
  for (int p : outs) pinMode(p, OUTPUT);

  pinMode(LIM_A, INPUT_PULLUP);
  pinMode(LIM_B, INPUT_PULLUP);
  pinMode(LIM_D, INPUT_PULLUP);
  pinMode(LIM_E, INPUT_PULLUP);

  pinMode(HOME_E, INPUT_PULLUP);
  pinMode(HOME_B, INPUT_PULLUP);
  pinMode(HOME_D, INPUT_PULLUP);

  gripper.attach(GRIP_PIN);
  gripper.write(gripAngle);

  Serial.println("[fw] thor_teleop_firmware ready (direct-drive rebuild)");
  Serial.println("[fw] JOG A+/A-/B+/B-/D+/D-/E+/E-/G+/G-   CAL A/B/D/E, CAL STOP   JOGSTEP <n>   SPEED <vmin> <vmax>   HOME?   HOME D/B/E");
}

// ── Ramp shape — identical logic to Markos_basic.ino's rampDelay() ──────────
int rampDelay(long i, long N) {
  if (N <= 1) return VMAX_US;
  long rampSteps = (long)(N * RAMP_FRAC);
  if (rampSteps < 1) rampSteps = 1;
  float t;
  if (i < rampSteps) t = (float)i / rampSteps;
  else if (i >= N - rampSteps) t = (float)(N - 1 - i) / rampSteps;
  else t = 1.0;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return (int)(VMIN_US + (VMAX_US - VMIN_US) * t);
}

bool limitTriggered(int pin, bool inverted) {
  int v = digitalRead(pin);
  return inverted ? (v == HIGH) : (v == LOW);
}

bool homeTriggered(int pin, bool activeHigh) {
  int v = digitalRead(pin);
  return activeHigh ? (v == HIGH) : (v == LOW);
}

const long NO_SOFT_LIMIT_MIN = -2000000000L;
const long NO_SOFT_LIMIT_MAX =  2000000000L;

// Blocking move of a single axis. Returns actual signed steps completed —
// may be less than |steps| if a limit switch stopped it early. The
// 5-argument overload behaves exactly as before (A and E use this one).
// The 8-argument overload adds a software position clamp on top of the
// hardware limit switch — pass startPos = the axis's current position
// counter and softMin/softMax to enable it (used for D, which has no
// working hardware limit). Two overloads rather than default arguments —
// the Arduino IDE auto-generates a prototype for every function in the
// sketch, and a default argument that appears in both the auto-generated
// prototype and the real definition is a compile error in C++.
long moveOne(int stepPin, int dirPin, long steps, int limPin, bool limInverted) {
  return moveOne(stepPin, dirPin, steps, limPin, limInverted, 0, NO_SOFT_LIMIT_MIN, NO_SOFT_LIMIT_MAX);
}
long moveOne(int stepPin, int dirPin, long steps, int limPin, bool limInverted,
             long startPos, long softMin, long softMax) {
  bool fwd = (steps >= 0);
  digitalWrite(dirPin, fwd ? HIGH : LOW);
  long N = labs(steps);
  long done = 0;
  long pos = startPos;
  for (long i = 0; i < N; i++) {
    if (limitTriggered(limPin, limInverted)) {
      Serial.println("[fw] LIMIT HIT mid-move - stopping early");
      break;
    }
    long nextPos = pos + (fwd ? 1 : -1);
    if (nextPos < softMin || nextPos > softMax) {
      Serial.println("[fw] SOFT LIMIT HIT mid-move - stopping early");
      break;
    }
    int d = rampDelay(i, N);
    digitalWrite(stepPin, HIGH); delayMicroseconds(d);
    digitalWrite(stepPin, LOW);  delayMicroseconds(d);
    done += fwd ? 1 : -1;
    pos = nextPos;
  }
  return done;
}

// Upper elbow: both motors together, mirrored, sharing LIM_B as the limit.
// The 1-argument overload is unchanged JOG/MOVE behavior. The 3-argument
// overload adds a second stop condition on top of LIM_B — used by homing to
// stop the instant HOME_B triggers. Two overloads, not a default argument —
// same Arduino auto-prototype reason as moveOne() above.
long moveUpperElbow(long steps) {
  return moveUpperElbow(steps, -1, false);
}
long moveUpperElbow(long steps, int extraPin, bool extraActiveHigh) {
  bool fwd = (steps >= 0);
  digitalWrite(DIR_B, fwd ? HIGH : LOW);
  bool cFwd = MIRROR_C ? !fwd : fwd;
  digitalWrite(DIR_C, cFwd ? HIGH : LOW);
  long N = labs(steps);
  long done = 0;
  for (long i = 0; i < N; i++) {
    if (limitTriggered(LIM_B, false)) {
      Serial.println("[fw] LIMIT HIT mid-move - stopping early");
      break;
    }
    if (extraPin >= 0 && homeTriggered(extraPin, extraActiveHigh)) {
      Serial.println("[fw] HOME BEACON HIT mid-move - stopping early");
      break;
    }
    int d = rampDelay(i, N);
    digitalWrite(STEP_B, HIGH); digitalWrite(STEP_C, HIGH);
    delayMicroseconds(d);
    digitalWrite(STEP_B, LOW); digitalWrite(STEP_C, LOW);
    delayMicroseconds(d);
    done += fwd ? 1 : -1;
  }
  return done;
}

long axisPos(int axis) {
  switch (axis) {
    case 0: return posA;
    case 1: return posB;
    case 2: return posD;
    default: return posE;
  }
}
const char* axisName(int axis) {
  switch (axis) {
    case 0: return "A";
    case 1: return "B";
    case 2: return "D";
    default: return "E";
  }
}

void doJog(char axis, float dir) {
  long steps = (long)(dir * jogStepSteps);
  long moved = 0;
  switch (axis) {
    case 'A': moved = moveOne(STEP_A, DIR_A, steps, LIM_A, false); posA += moved; break;
    case 'B': moved = moveUpperElbow(B_DIR_FLIP ? -steps : steps); posB += moved; break;
    case 'D': moved = moveOne(STEP_D, DIR_D, D_DIR_FLIP ? -steps : steps, LIM_D, false,
                               posD, D_SOFT_MIN, D_SOFT_MAX);
              posD += moved; break;
    case 'E': moved = moveOne(STEP_E, DIR_E, E_DIR_FLIP ? -steps : steps, LIM_E, LIM_E_INVERTED);
              posE += moved; break;
  }
  Serial.print("[fw] JOG ");
  Serial.print(axis);
  Serial.print(" done, moved ");
  Serial.println(moved);
}

// Drives one axis to its home beacon and zeros its position counter there.
// Bounded search, never an unbounded seek:
//   D: already clamped to D_SOFT_MIN/MAX by moveOne (see the constants above)
//      no matter which way it searches, so trying '+' first and, if not
//      found, retracing to the start and trying '-' is safe by construction.
//   B: single bounded search toward the edge beacon; LIM_B is still checked
//      as a real hardware backstop the whole time.
//   E: single bounded search covering a bit over one full rotation, since
//      direction doesn't matter on a continuous twist.
// Any axis not already at home when this is called; if the beacon isn't
// found within the bound, it's reported and the axis is returned to where
// it started — never left stranded out at a search boundary.
//
// This is doHomeSearch(), the actual search logic — doHome() below wraps it
// with a slower, homing-specific ramp speed (see HOME_VMIN_US/MAX_US) for
// more torque margin during the search, then restores whatever speed was
// set before. Multiple early returns inside made a save/restore-on-every-
// exit-point version error-prone to keep correct by hand, so it's one
// wrapper around a single call instead.
void doHomeSearch(char axis) {
  if (axis == 'D') {
    if (homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME D already at home");
      posD = 0;
      return;
    }
    long moved = moveOne(STEP_D, DIR_D, HOME_SEARCH_D_PLUS, HOME_D, HOME_D_ACTIVE_HIGH,
                          posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += moved;
    if (homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME D found (+), zeroed");
      posD = 0;
      return;
    }
    long back = moveOne(STEP_D, DIR_D, -moved, HOME_D, HOME_D_ACTIVE_HIGH,
                         posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += back;
    moved = moveOne(STEP_D, DIR_D, -HOME_SEARCH_D_MINUS, HOME_D, HOME_D_ACTIVE_HIGH,
                     posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += moved;
    if (homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME D found (-), zeroed");
      posD = 0;
      return;
    }
    Serial.println("[fw] HOME D FAILED - beacon not found either direction within the safe range. Returning to start.");
    back = moveOne(STEP_D, DIR_D, -moved, HOME_D, HOME_D_ACTIVE_HIGH,
                    posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += back;

  } else if (axis == 'B') {
    if (homeTriggered(HOME_B, HOME_B_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME B already at home");
      posB = 0;
      return;
    }
    long moved = moveUpperElbow(HOME_SEARCH_B, HOME_B, HOME_B_ACTIVE_HIGH);
    posB += moved;
    if (homeTriggered(HOME_B, HOME_B_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME B found, zeroed");
      posB = 0;
    } else {
      Serial.println("[fw] HOME B FAILED - beacon not found within search range. Returning to start.");
      long back = moveUpperElbow(-moved, HOME_B, HOME_B_ACTIVE_HIGH);
      posB += back;
    }

  } else if (axis == 'E') {
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME E already at home");
      posE = 0;
      return;
    }
    long moved = moveOne(STEP_E, DIR_E, HOME_SEARCH_E, HOME_E, HOME_E_ACTIVE_HIGH);
    posE += moved;
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME E found, zeroed");
      posE = 0;
    } else {
      Serial.println("[fw] HOME E FAILED - beacon not found within one full rotation. Returning to start.");
      long back = moveOne(STEP_E, DIR_E, -moved, HOME_E, HOME_E_ACTIVE_HIGH);
      posE += back;
    }

  } else {
    Serial.println("[fw] HOME: unknown axis (use D, B, or E - A has no beacon yet)");
  }
}

void doHome(char axis) {
  int savedVMin = VMIN_US, savedVMax = VMAX_US;
  VMIN_US = HOME_VMIN_US;
  VMAX_US = HOME_VMAX_US;
  doHomeSearch(axis);
  VMIN_US = savedVMin;
  VMAX_US = savedVMax;
}

void applyLine(const String& rawLine) {
  String line = rawLine;
  line.trim();
  if (line.length() == 0) return;

  if (line.startsWith("JOGSTEP ")) {
    long n = line.substring(8).toInt();
    if (n > 0) {
      jogStepSteps = constrain(n, 1, MAX_JOG_STEPS);
      Serial.print("[fw] jog step -> ");
      Serial.println(jogStepSteps);
    }
    return;
  }

  if (line == "HOME?") {
    // Status-only — reports whether each beacon currently reads "at home".
    // Doesn't move anything. Use it while hand-jogging toward each beacon
    // to confirm it flips exactly where expected before anything automatic
    // is built on top of it.
    Serial.print("[fw] home E=");
    Serial.print(homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH) ? "YES" : "no");
    Serial.print(" B=");
    Serial.print(homeTriggered(HOME_B, HOME_B_ACTIVE_HIGH) ? "YES" : "no");
    Serial.print(" D=");
    Serial.println(homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH) ? "YES" : "no");
    return;
  }

  if (line.startsWith("HOME ")) {
    // "HOME D" / "HOME B" / "HOME E" — actually drives that axis to its
    // beacon (bounded search, see doHome() above) and zeros its position
    // counter there. Unlike HOME?, this one moves the arm.
    String arg = line.substring(5);
    arg.trim();
    if (arg.length() == 1) {
      doHome(arg[0]);
    } else {
      Serial.println("[fw] HOME: expected a single axis letter (D, B, or E)");
    }
    return;
  }

  if (line.startsWith("SPEED ")) {
    // "SPEED <vmin_us> <vmax_us>" — same VMIN/VMAX terminology as
    // Markos_basic.ino. BIGGER us = SLOWER. vmin = start/end (slow),
    // vmax = cruise (fast). Adjustable live, no reflash needed.
    String rest = line.substring(6);
    int sp = rest.indexOf(' ');
    if (sp > 0) {
      int newMin = rest.substring(0, sp).toInt();
      int newMax = rest.substring(sp + 1).toInt();
      newMin = constrain(newMin, 200, 6000);
      newMax = constrain(newMax, 200, 6000);
      if (newMax > newMin) newMax = newMin;   // keep fast <= slow
      VMIN_US = newMin;
      VMAX_US = newMax;
      Serial.print("[fw] speed -> VMIN=");
      Serial.print(VMIN_US);
      Serial.print(" VMAX=");
      Serial.println(VMAX_US);
    }
    return;
  }

  if (line.length() >= 2 && line[0] == 'G' && (isDigit(line[1]))) {
    // Absolute gripper set, e.g. "G0" or "G90" — distinct from "JOG G+/-".
    gripAngle = constrain(line.substring(1).toInt(), 0, 180);
    gripper.write(gripAngle);
    Serial.print("[fw] grip -> ");
    Serial.println(gripAngle);
    return;
  }

  if (line.startsWith("MOVE ")) {
    // "MOVE A<n> B<n> D<n> E<n>" — explicit signed step count (not the fixed
    // JOG size), for continuous velocity-control use: Python computes a
    // fresh delta from live gesture velocity and sends it here every tick.
    // Same blocking, ramped, limit-checked move functions as JOG.
    String rest = line.substring(5);
    int i = 0, len = rest.length();
    while (i < len) {
      char c = rest[i];
      if (c == 'A' || c == 'B' || c == 'D' || c == 'E') {
        int j = i + 1;
        while (j < len && rest[j] != ' ') j++;
        long n = rest.substring(i + 1, j).toInt();
        long moved;
        if (c == 'A') {
          moved = moveOne(STEP_A, DIR_A, n, LIM_A, false);
          posA += moved;
        } else if (c == 'B') {
          moved = moveUpperElbow(B_DIR_FLIP ? -n : n);
          posB += moved;
        } else if (c == 'D') {
          moved = moveOne(STEP_D, DIR_D, D_DIR_FLIP ? -n : n, LIM_D, false,
                           posD, D_SOFT_MIN, D_SOFT_MAX);
          posD += moved;
        } else {
          moved = moveOne(STEP_E, DIR_E, E_DIR_FLIP ? -n : n, LIM_E, LIM_E_INVERTED);
          posE += moved;
        }
        Serial.print("[fw] MOVE ");
        Serial.print(c);
        Serial.print(" done, moved ");
        Serial.println(moved);
        i = j;
      } else {
        i++;
      }
    }
    return;
  }

  if (line.startsWith("JOG ")) {
    String arg = line.substring(4);
    if (arg.length() == 2) {
      char axis = arg[0];
      float dir = (arg[1] == '+') ? 1.0 : -1.0;
      if (axis == 'G') {
        gripAngle = constrain(gripAngle + (int)(dir * JOG_GRIP_DEG), 0, 180);
        gripper.write(gripAngle);
        Serial.print("[fw] grip -> ");
        Serial.println(gripAngle);
      } else if (axis == 'A' || axis == 'B' || axis == 'D' || axis == 'E') {
        doJog(axis, dir);
      }
    }
    return;
  }

  if (line.startsWith("CAL ")) {
    String arg = line.substring(4);
    if (arg == "STOP") {
      if (calMode) {
        Serial.print("[fw] CAL DONE axis=");
        Serial.print(axisName(calAxis));
        Serial.print(" steps=");
        Serial.println(axisPos(calAxis));
        calMode = false;
        calAxis = -1;
      }
      return;
    }
    int axis = (arg == "A") ? 0 : (arg == "B") ? 1 : (arg == "D") ? 2
             : (arg == "E") ? 3 : -1;
    if (axis >= 0) {
      calMode = true;
      calAxis = axis;
      switch (axis) {
        case 0: posA = 0; break;
        case 1: posB = 0; break;
        case 2: posD = 0; break;
        default: posE = 0; break;
      }
      Serial.print("[fw] CAL START axis=");
      Serial.println(axisName(axis));
    }
    return;
  }

  Serial.print("[fw] unrecognized: ");
  Serial.println(line);
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      applyLine(rxLine);
      rxLine = "";
    } else if (c != '\r') {
      rxLine += c;
    }
  }
}
