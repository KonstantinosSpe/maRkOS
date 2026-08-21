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

// ── Ramp speed range (microseconds between pulse edges) — Markos_basic.ino's
// own proven defaults. BIGGER = SLOWER. Starts/ends slow, cruises fast.
int VMIN_US = 2200;
int VMAX_US = 900;
const float RAMP_FRAC = 0.3;   // fraction of the move spent accelerating (and decelerating)

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

  gripper.attach(GRIP_PIN);
  gripper.write(gripAngle);

  Serial.println("[fw] thor_teleop_firmware ready (direct-drive rebuild)");
  Serial.println("[fw] JOG A+/A-/B+/B-/D+/D-/E+/E-/G+/G-   CAL A/B/D/E, CAL STOP   JOGSTEP <n>   SPEED <vmin> <vmax>");
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

// Blocking move of a single axis. Returns actual signed steps completed —
// may be less than |steps| if a limit switch stopped it early.
long moveOne(int stepPin, int dirPin, long steps, int limPin, bool limInverted) {
  bool fwd = (steps >= 0);
  digitalWrite(dirPin, fwd ? HIGH : LOW);
  long N = labs(steps);
  long done = 0;
  for (long i = 0; i < N; i++) {
    if (limitTriggered(limPin, limInverted)) {
      Serial.println("[fw] LIMIT HIT mid-move - stopping early");
      break;
    }
    int d = rampDelay(i, N);
    digitalWrite(stepPin, HIGH); delayMicroseconds(d);
    digitalWrite(stepPin, LOW);  delayMicroseconds(d);
    done += fwd ? 1 : -1;
  }
  return done;
}

// Upper elbow: both motors together, mirrored, sharing LIM_B as the limit.
long moveUpperElbow(long steps) {
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
    case 'D': moved = moveOne(STEP_D, DIR_D, D_DIR_FLIP ? -steps : steps, LIM_D, false);
              posD += moved; break;
    case 'E': moved = moveOne(STEP_E, DIR_E, E_DIR_FLIP ? -steps : steps, LIM_E, LIM_E_INVERTED);
              posE += moved; break;
  }
  Serial.print("[fw] JOG ");
  Serial.print(axis);
  Serial.print(" done, moved ");
  Serial.println(moved);
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
          moved = moveOne(STEP_D, DIR_D, D_DIR_FLIP ? -n : n, LIM_D, false);
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
