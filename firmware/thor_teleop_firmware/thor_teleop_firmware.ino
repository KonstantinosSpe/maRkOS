/*
  thor_teleop_firmware.ino
  =========================
  This sketch drives the motors the same direct way Markos_basic.ino
  does — plain digitalWrite()/delayMicroseconds() bit-banging, no stepper
  library, no enable pin — because that's the firmware that's actually
  been confirmed to move this exact robot. An AccelStepper-based version
  of this file was tried first and never produced confirmed physical
  motion across extensive testing on the same pins, so the AccelStepper
  layer itself is the prime suspect there; this version sidesteps that
  question entirely by using the same proven mechanism instead.

  JOG (small bounded nudges) and CAL (passive recording of net steps
  moved) cover A / B(+C mirrored) / D / E, plus the gripper servo. MOVE
  (explicit signed step counts, for continuous gesture-driven streaming
  from Python) covers the same four axes, including D — D spent most of
  this project's life excluded from MOVE ("never move D, full stop"), and
  that guarantee has since been lifted by deliberate choice, not by
  accident. This firmware itself has no concept of "armed/disarmed": it
  will execute any MOVE D it receives exactly like MOVE A/B/E. All of that
  caution lives on the Python side instead, in mark1os.py's hw_config.py
  (HW_CALIBRATED_D, which starts False) — if that ever changes, this
  firmware moves D on command with no extra gate here.

  Every JOG move is blocking and bounded by construction: a for-loop over
  a fixed step count, checking the relevant limit switch every single
  pulse and stopping early if it trips. There's no async "ramp toward a
  target" state that could get stuck running, so the failure mode an
  AccelStepper-based version could hit — a repeated command silently
  extending a burst indefinitely — simply can't happen here, since nothing
  is still running once a JOG call returns.

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
  stopped it early). That confirms the command was received and a pulse
  sequence was issued; it doesn't by itself prove the motor physically
  turned, since that still depends on the driver being powered and wired
  correctly — worth a visual check by eye, separately from this.

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
                 see doHome()) and zeros its position counter there. The
                 success line reports steps=<n>, the actual number of
                 steps the search took — compare that against a known
                 commanded distance to measure real step accuracy at a
                 given point in the range (see step-accuracy-calibration
                 branch / calibrate_steps.py).
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
  3. GRIP_PIN is 10, matching Markos_basic.ino — worth a visual check of
     the wiring before trusting it, same as anything else on this list.
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
const bool LIM_E_INVERTED = false; // E's home switch is read directly via
                                    // digitalRead(), the same as the other
                                    // axes (LOW = triggered) — the
                                    // "inverted" note this contradicts
                                    // turned out to describe Grbl's own
                                    // status-word correction, a different
                                    // layer entirely, not the raw pin
                                    // reading used here.
const bool E_DIR_FLIP = true;      // A and E were observed rotating opposite
                                    // ways for what should be the same "+"
                                    // sense, so this flips E's sign (in both
                                    // JOG and MOVE) to bring the two back
                                    // into agreement. Applied here, at the
                                    // source, so it stays consistent no
                                    // matter how E ends up being driven.
const bool B_DIR_FLIP = false;     // set true if JOG B+/MOVE B<positive> moves
                                    // the shoulder opposite to what the
                                    // height/reach IK expects — B's direction
                                    // relative to the IK's sign convention
                                    // hasn't been isolated-tested yet, only
                                    // its magnitude has.
const bool D_DIR_FLIP = false;     // same idea as B_DIR_FLIP, for D — D has
                                    // even less live-driving history than B.

const int GRIP_PIN = 10;   // matches Markos_basic.ino — an earlier version
                            // of this file guessed 11, so it's worth a
                            // visual check of the wiring before trusting
                            // this one either.

// Hand-tested on the real MG996R gripper by jogging it in both directions:
// past 60 degrees toward closed it starts straining against the mechanism,
// and 105 is a confirmed comfortable full open. Both JOG G+/G- and the
// absolute G<n> command clamp to this range below, the same way B_SOFT_MIN/
// MAX and D_SOFT_MIN/MAX protect the other axes — nothing sent to this
// servo, by hand or by gesture control, should be able to drive it into
// the strain zone.
const int GRIP_MIN = 60;
const int GRIP_MAX = 105;

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
// still disarmed on the Python side (HW_CALIBRATED_A = False in
// hw_config.py). Add one here the same way if that ever changes.

// ── B's software position clamp ──────────────────────────────────────────
// B's home sits at one edge (the near/'+' side), and that side already has
// a genuine hardware backstop in the real, working LIM_B switch. The far
// edge ('-' direction, into the usable range) has no switch at all, so its
// true physical extent has to come from somewhere else — hand-measured
// directly, it's 180 steps edge-to-edge. That's noticeably tighter than the
// code's separate J3 soft-angle-limit would suggest (around 245 steps),
// which is too generous for the real mechanism: driving out to that number
// means grinding against the far hard stop for the excess, which is exactly
// what a 51-step discrepancy at the 245-step calibration point turned out
// to be. 10 steps of margin are subtracted off each end below.
const long B_SOFT_MIN = -170;
const long B_SOFT_MAX = 10;

// ── D's software position clamp ──────────────────────────────────────────
// LIM_D (pin 48, in the limit-switch block above) doesn't reliably trigger
// — confirmed by hand-testing, it never fired even well past where home
// is. So unlike B and E, D has no working hardware backstop against
// overtravel, and these bounds are its only real protection. They're set
// to the full hand-tested safe range (driven to -1700 and +1500 steps from
// home, both without incident) rather than shaved down with extra margin,
// because posD is just an open-loop step count with no absolute reference
// except a successful HOME D — any drift since the last one (a lost step
// under load, a prior search that returned to a not-quite-true start) eats
// directly into whatever margin gets subtracted here. Using the full
// tested range instead gives real tolerance for that drift while still
// never exceeding what's actually been verified safe by hand.
const long D_SOFT_MIN = -1700;
const long D_SOFT_MAX = 1500;

// ── Bounded search distances for HOME <axis> ─────────────────────────────
// D's search request (3500) is deliberately larger than D_SOFT_MIN/MAX's
// full span (3200) — not because D is allowed to travel that far, but so
// the soft limit, an absolute position check, is always what stops the
// search rather than the requested step count itself. Sizing the request
// to match D_SOFT_MAX exactly would only cover the intended search
// distance when posD happens to start at exactly 0; starting anywhere
// else — completely normal after manual JOG testing — would hit the
// absolute soft-limit boundary after far fewer steps than intended,
// cutting the search short before it should actually give up. The soft
// limit itself is unchanged and still the real safety bound; this only
// affects how far the search is willing to look.
const long HOME_SEARCH_D_PLUS  = 3500;
const long HOME_SEARCH_D_MINUS = 3500;
// B's is a generous guess since its beacon sits at an edge and LIM_B works
// as a real backstop either way. E's search picks a starting direction
// based on estimated shortest path (see doHomeSearch's E branch) rather
// than always searching the same way, but still needs a bound generous
// enough to find the beacon even if that estimate is wrong and it has to
// fall back the other way.
const long HOME_SEARCH_B       = 2000;
// E_FULL_ROTATION comes from a direct measurement (SWEEP E+/E-, a real
// full lap home to home): 1003 steps forward, 1004 back. That matters
// because a full lap isn't something you can derive from the motor's
// nominal steps/rev the way you might hope — the number here uses the
// average of the two directions, rounded; the 1-step/0.1% difference
// between them isn't worth modeling separately. mark1os.py's
// stepsPerDegE is derived from this same measurement (1004/360), so the
// two stay consistent with each other.
const long E_FULL_ROTATION     = 1004;
const long HOME_SEARCH_E       = 1300;   // full lap + generous margin

// ── Ramp speed range (microseconds between pulse edges). BIGGER = SLOWER.
// Starts/ends slow, cruises fast. Matches Markos_basic.ino's own
// terminology and started from its 2200/900 default; this file's default
// runs a bit slower than that (3000/1800), which tracks noticeably more
// reliably on real hardware — every limit/beacon/soft-limit check happens
// once per step, so the faster default leaves less margin to catch a
// trigger cleanly before overshooting it. Still adjustable live with no
// reflash via SPEED <vmin> <vmax>.
int VMIN_US = 3000;
int VMAX_US = 1800;
const float RAMP_FRAC = 0.3;   // fraction of the move spent accelerating (and decelerating)

// Homing searches run even slower than normal JOG/MOVE speed. D in
// particular can lose steps under load, and which direction counts as
// "uphill" depends on D's current angle rather than a fixed sign — so
// instead of guessing which direction needs more margin, doHome() just
// slows every homing search down uniformly, for more torque margin
// regardless of direction.
const int HOME_VMIN_US = 4200;
const int HOME_VMAX_US = 2600;

long jogStepSteps = 200;         // raw steps per JOG press, runtime-adjustable via JOGSTEP
const long MAX_JOG_STEPS = 20000;

const float JOG_GRIP_DEG = 3.0;  // gripper degrees per JOG G+/G- press

Servo gripper;
int gripAngle = GRIP_MAX;   // boots open, ready to grab something

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
  Serial.println("[fw] JOG A+/A-/B+/B-/D+/D-/E+/E-/G+/G-   CAL A/B/D/E, CAL STOP   JOGSTEP <n>   SPEED <vmin> <vmax>   HOME?   HOME D/B/E   SWEEP E+/E-");
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

// Moves exactly the requested step count with no limit/beacon check at
// all — used only where the point is to deliberately move OUT of a
// sensor's own trigger zone before starting a real, checked search. Using
// moveOne() with that same sensor as limPin for this doesn't work: the
// per-step check runs before any movement, so if already inside the zone
// it sees "triggered" on the very first check and stops at 0 steps,
// unable to ever leave. Small, fixed-distance use only — no safety net,
// so this must never be used for anything but a short, known-safe nudge.
long blindMove(int stepPin, int dirPin, long steps) {
  bool fwd = (steps >= 0);
  digitalWrite(dirPin, fwd ? HIGH : LOW);
  long N = labs(steps);
  for (long i = 0; i < N; i++) {
    int d = rampDelay(i, N);
    digitalWrite(stepPin, HIGH); delayMicroseconds(d);
    digitalWrite(stepPin, LOW);  delayMicroseconds(d);
  }
  return steps;
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
// Three overloads, not default arguments — same Arduino auto-prototype
// reason as moveOne() above:
//   1-arg:  unchanged JOG/MOVE behavior, no extra beacon check, no clamp.
//   3-arg:  adds a second stop condition on top of LIM_B — used by homing
//           to stop the instant HOME_B triggers.
//   6-arg:  additionally adds the B_SOFT_MIN/MAX position clamp (pass
//           startPos = posB to enable it) — used everywhere now, JOG/MOVE
//           included, so B can't be driven past its real 180-step range
//           no matter which command drives it.
long moveUpperElbow(long steps) {
  return moveUpperElbow(steps, -1, false, 0, NO_SOFT_LIMIT_MIN, NO_SOFT_LIMIT_MAX);
}
long moveUpperElbow(long steps, int extraPin, bool extraActiveHigh) {
  return moveUpperElbow(steps, extraPin, extraActiveHigh, 0, NO_SOFT_LIMIT_MIN, NO_SOFT_LIMIT_MAX);
}
long moveUpperElbow(long steps, int extraPin, bool extraActiveHigh,
                     long startPos, long softMin, long softMax) {
  bool fwd = (steps >= 0);
  digitalWrite(DIR_B, fwd ? HIGH : LOW);
  bool cFwd = MIRROR_C ? !fwd : fwd;
  digitalWrite(DIR_C, cFwd ? HIGH : LOW);
  long N = labs(steps);
  long done = 0;
  long pos = startPos;
  for (long i = 0; i < N; i++) {
    if (limitTriggered(LIM_B, false)) {
      Serial.println("[fw] LIMIT HIT mid-move - stopping early");
      break;
    }
    if (extraPin >= 0 && homeTriggered(extraPin, extraActiveHigh)) {
      Serial.println("[fw] HOME BEACON HIT mid-move - stopping early");
      break;
    }
    long nextPos = pos + (fwd ? 1 : -1);
    if (nextPos < softMin || nextPos > softMax) {
      Serial.println("[fw] SOFT LIMIT HIT mid-move - stopping early");
      break;
    }
    int d = rampDelay(i, N);
    digitalWrite(STEP_B, HIGH); digitalWrite(STEP_C, HIGH);
    delayMicroseconds(d);
    digitalWrite(STEP_B, LOW); digitalWrite(STEP_C, LOW);
    delayMicroseconds(d);
    done += fwd ? 1 : -1;
    pos = nextPos;
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

// Moves one axis by a signed step count, applying that axis's own
// direction flip and soft-limit clamp, and updates its position counter.
// This is the one place that logic lives — both doJog() and the MOVE
// command handler in applyLine() need it, and keeping it in one function
// means a change to how an axis is driven (a new clamp, a corrected
// DIR_FLIP) only has to happen once instead of being kept in sync by hand
// in two places.
long applyAxisDelta(char axis, long n) {
  long moved = 0;
  switch (axis) {
    case 'A':
      moved = moveOne(STEP_A, DIR_A, n, LIM_A, false);
      posA += moved;
      break;
    case 'B':
      moved = moveUpperElbow(B_DIR_FLIP ? -n : n, -1, false,
                              posB, B_SOFT_MIN, B_SOFT_MAX);
      posB += moved;
      break;
    case 'D':
      moved = moveOne(STEP_D, DIR_D, D_DIR_FLIP ? -n : n, LIM_D, false,
                       posD, D_SOFT_MIN, D_SOFT_MAX);
      posD += moved;
      break;
    case 'E':
      moved = moveOne(STEP_E, DIR_E, E_DIR_FLIP ? -n : n, LIM_E, LIM_E_INVERTED);
      posE += moved;
      break;
  }
  return moved;
}

void doJog(char axis, float dir) {
  long steps = (long)(dir * jogStepSteps);
  long moved = applyAxisDelta(axis, steps);
  Serial.print("[fw] JOG ");
  Serial.print(axis);
  Serial.print(" done, moved ");
  Serial.println(moved);
}

// Drives one axis to its home beacon and zeros its position counter there.
// Bounded search, never an unbounded seek:
//   D: picks whichever direction posD's sign says is actually shorter
//      (home is centered, not at an edge, so this isn't a fixed guess)
//      and falls back to the other direction if that estimate is wrong —
//      still safe regardless, since D_SOFT_MIN/MAX clamps every move.
//   B: single bounded search toward the edge beacon; LIM_B is still checked
//      as a real hardware backstop the whole time.
//   E: picks whichever direction is shorter from posE mod one full lap
//      (E_FULL_ROTATION), same idea as D but circular — see its own branch
//      below for why a fixed direction doesn't work for a continuous twist.
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
      Serial.println("[fw] HOME D already at home, steps=0");
      posD = 0;
      return;
    }
    // D's home is centered (0), not at an edge, so which direction is
    // actually shorter depends entirely on which side of 0 posD is
    // currently on. Always trying '+' first regardless of that would mean
    // every '+'-side test point's return search needlessly travels all
    // the way out to D_SOFT_MAX before reversing and finding it going '-'.
    // Simpler than E's mod-based estimate since D isn't circular:
    // negative posD means home is toward '+', positive means home is
    // toward '-'.
    bool searchPositive = (posD < 0);
    long primaryReq = searchPositive ? HOME_SEARCH_D_PLUS : -HOME_SEARCH_D_MINUS;
    long moved = moveOne(STEP_D, DIR_D, primaryReq, HOME_D, HOME_D_ACTIVE_HIGH,
                          posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += moved;
    if (homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH)) {
      Serial.print("[fw] HOME D found (");
      Serial.print(searchPositive ? "+" : "-");
      Serial.print("), zeroed, steps=");
      Serial.println(labs(moved));
      posD = 0;
      return;
    }
    long back = moveOne(STEP_D, DIR_D, -moved, HOME_D, HOME_D_ACTIVE_HIGH,
                         posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += back;
    long fallbackReq = searchPositive ? -HOME_SEARCH_D_MINUS : HOME_SEARCH_D_PLUS;
    moved = moveOne(STEP_D, DIR_D, fallbackReq, HOME_D, HOME_D_ACTIVE_HIGH,
                     posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += moved;
    if (homeTriggered(HOME_D, HOME_D_ACTIVE_HIGH)) {
      Serial.print("[fw] HOME D found (");
      Serial.print(searchPositive ? "-" : "+");
      Serial.print(", fallback), zeroed, steps=");
      Serial.println(labs(moved));
      posD = 0;
      return;
    }
    Serial.println("[fw] HOME D FAILED - beacon not found either direction within the safe range. Returning to start.");
    back = moveOne(STEP_D, DIR_D, -moved, HOME_D, HOME_D_ACTIVE_HIGH,
                    posD, D_SOFT_MIN, D_SOFT_MAX);
    posD += back;

  } else if (axis == 'B') {
    if (homeTriggered(HOME_B, HOME_B_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME B already at home, steps=0");
      posB = 0;
      return;
    }
    long moved = moveUpperElbow(HOME_SEARCH_B, HOME_B, HOME_B_ACTIVE_HIGH,
                                 posB, B_SOFT_MIN, B_SOFT_MAX);
    posB += moved;
    if (homeTriggered(HOME_B, HOME_B_ACTIVE_HIGH)) {
      Serial.print("[fw] HOME B found, zeroed, steps=");
      Serial.println(labs(moved));
      posB = 0;
    } else {
      Serial.println("[fw] HOME B FAILED - beacon not found within search range. Returning to start.");
      long back = moveUpperElbow(-moved, HOME_B, HOME_B_ACTIVE_HIGH,
                                  posB, B_SOFT_MIN, B_SOFT_MAX);
      posB += back;
    }

  } else if (axis == 'E') {
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.println("[fw] HOME E already at home, steps=0");
      posE = 0;
      return;
    }
    // E is a continuous rotation with no proximal/distal concept, so
    // unlike D/B there's no reason to always search the same fixed
    // direction — a fixed direction is exactly what makes results
    // inconsistent for a continuous twist, sometimes searching the long
    // way around, sometimes missing the beacon within the search bound
    // entirely. Instead, estimate which direction is actually shorter
    // from posE's accumulated position (mod one full rotation) and try
    // that first, falling back to the other direction if the estimate
    // turns out wrong — same try-then-fallback pattern as D's search
    // above.
    long offset = posE % E_FULL_ROTATION;
    if (offset < 0) offset += E_FULL_ROTATION;
    bool searchPositive = (offset > E_FULL_ROTATION / 2);
    long primary = searchPositive ? HOME_SEARCH_E : -HOME_SEARCH_E;
    long moved = moveOne(STEP_E, DIR_E, primary, HOME_E, HOME_E_ACTIVE_HIGH);
    posE += moved;
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.print("[fw] HOME E found, zeroed, steps=");
      Serial.println(labs(moved));
      posE = 0;
      return;
    }
    long back = moveOne(STEP_E, DIR_E, -moved, HOME_E, HOME_E_ACTIVE_HIGH);
    posE += back;
    moved = moveOne(STEP_E, DIR_E, -primary, HOME_E, HOME_E_ACTIVE_HIGH);
    posE += moved;
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.print("[fw] HOME E found (fallback direction), zeroed, steps=");
      Serial.println(labs(moved));
      posE = 0;
    } else {
      Serial.println("[fw] HOME E FAILED - beacon not found either direction. Returning to start.");
      back = moveOne(STEP_E, DIR_E, -moved, HOME_E, HOME_E_ACTIVE_HIGH);
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

// SWEEP E+ / SWEEP E- — measures a genuine full lap: forces the search in
// the exact requested direction regardless of what looks shorter (unlike
// HOME E, which picks whichever direction its position estimate says is
// closer — the right choice for normal homing, but wrong here, where the
// whole point is to go all the way around on purpose and see how many
// steps a real full rotation actually takes). Meant to be called starting
// from home; if it isn't, the reported count won't mean "one lap."
void doSweepE(int dir) {
  int savedVMin = VMIN_US, savedVMax = VMAX_US;
  VMIN_US = HOME_VMIN_US;
  VMAX_US = HOME_VMAX_US;

  // If we're starting right at home (the normal case — this is meant to
  // be called right after HOME E), the beacon's own trigger zone is still
  // active. blindMove ignores the beacon entirely for this fixed 100-step
  // nudge — using moveOne() with HOME_E as its own limPin here doesn't
  // work, since it would check (and immediately trip on) the very zone
  // it's trying to leave, stopping at 0 steps before moving at all.
  long cleared = 0;
  if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
    cleared = blindMove(STEP_E, DIR_E, dir * 100);
    posE += cleared;
    if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
      Serial.println("[fw] SWEEP E FAILED - still inside the beacon zone after clearing 100 steps; zone wider than expected.");
      VMIN_US = savedVMin;
      VMAX_US = savedVMax;
      return;
    }
  }

  long request = dir * (E_FULL_ROTATION + 300);  // generous margin past one full lap
  long moved = moveOne(STEP_E, DIR_E, request, HOME_E, HOME_E_ACTIVE_HIGH);
  posE += moved;
  long total = labs(cleared) + labs(moved);
  if (homeTriggered(HOME_E, HOME_E_ACTIVE_HIGH)) {
    Serial.print("[fw] SWEEP E found, zeroed, steps=");
    Serial.println(total);
    posE = 0;
  } else {
    Serial.println("[fw] SWEEP E FAILED - beacon not found within one full lap plus margin.");
  }

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

  if (line.startsWith("SWEEP ")) {
    // "SWEEP E+" / "SWEEP E-" — see doSweepE()'s comment. Only meaningful
    // for E (a continuous rotation); A/B/D don't have a "full lap".
    String arg = line.substring(6);
    if (arg.length() == 2 && arg[0] == 'E' && (arg[1] == '+' || arg[1] == '-')) {
      doSweepE(arg[1] == '+' ? 1 : -1);
    } else {
      Serial.println("[fw] SWEEP: expected E+ or E-");
    }
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
    // Absolute gripper set, e.g. "G60" or "G105" — distinct from "JOG G+/-".
    // Clamped to GRIP_MIN/MAX, same as every other axis's hand-tested range.
    gripAngle = constrain(line.substring(1).toInt(), GRIP_MIN, GRIP_MAX);
    gripper.write(gripAngle);
    Serial.print("[fw] grip -> ");
    Serial.println(gripAngle);
    return;
  }

  if (line.startsWith("MOVE ")) {
    // "MOVE A<n> B<n> D<n> E<n>" — explicit signed step count (not the fixed
    // JOG size), for continuous velocity-control use: Python computes a
    // fresh delta from live gesture velocity and sends it here every tick.
    // Same blocking, ramped, limit-checked move as JOG, via applyAxisDelta().
    String rest = line.substring(5);
    int i = 0, len = rest.length();
    while (i < len) {
      char c = rest[i];
      if (c == 'A' || c == 'B' || c == 'D' || c == 'E') {
        int j = i + 1;
        while (j < len && rest[j] != ' ') j++;
        long n = rest.substring(i + 1, j).toInt();
        long moved = applyAxisDelta(c, n);
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
        gripAngle = constrain(gripAngle + (int)(dir * JOG_GRIP_DEG), GRIP_MIN, GRIP_MAX);
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
