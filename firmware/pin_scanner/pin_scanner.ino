/*
  pin_scanner.ino
  ================
  Diagnostic sketch to find out which physical digital pin each newly-wired
  homing sensor is actually on: the mechanical switch + the 4 optical
  light-sensor beacons.

  It watches every Mega digital pin that ISN'T already used by
  thor_teleop_firmware.ino (steppers, existing limit switches, gripper) and
  prints ONLY when a pin's state changes. Trigger one sensor at a time —
  press the switch, block/unblock a beam — and watch exactly which pin
  number reacts in the Serial Monitor. If you trigger a sensor and nothing
  ever prints, that's the "useless" one: either not wired to a scanned pin,
  or not producing a signal at all.

  How to use:
    1. Flash this sketch (TEMPORARILY — this is not the real firmware, don't
       leave the arm connected to anything that could move while testing).
    2. Open Serial Monitor at 115200 baud.
    3. Trigger each of the 5 sensors one at a time, with a pause between
       each, and write down which pin number printed for which sensor.
    4. Tell me the 5 pin numbers and which physical sensor each one is —
       I'll wire them into thor_teleop_firmware.ino as the real homing pins.

  All candidate pins are read with INPUT_PULLUP (idle HIGH, LOW when
  triggered) — matches how LIM_A/B/D/E are already wired in the main
  firmware. If an optical sensor actively drives its own output (a 3-pin
  module with its own pull-up/down, rather than a bare switch-to-ground),
  the internal pullup is harmless — it's very weak and won't fight a driven
  signal — but idle/triggered polarity might come out the opposite of what
  you'd guess. Doesn't matter for this script: it only needs to show you
  WHICH pin changes, not which direction is "triggered."
*/

// Pins already spoken for by thor_teleop_firmware.ino — excluded from the
// scan so this doesn't fight with wiring whose meaning you already know.
const int USED_PINS[] = {
  10,                                   // gripper servo
  22, 23, 24, 26, 28, 30, 31, 32, 34, 36,  // STEP/DIR for A, B, C, D, E
  42, 44, 48, 49                        // existing LIM_A / LIM_B / LIM_D / LIM_E
};
const int USED_COUNT = sizeof(USED_PINS) / sizeof(USED_PINS[0]);

bool isUsed(int pin) {
  for (int i = 0; i < USED_COUNT; i++) {
    if (USED_PINS[i] == pin) return true;
  }
  return false;
}

// Candidate pins: everything else in the Mega's digital range, skipping
// 0/1 (USB serial) and 14-21 (Serial1-3 / I2C) by default — edit the skip
// range below if you know those are free and want them scanned too.
int candidates[54];
int candidateCount = 0;
bool lastState[54];

void setup() {
  Serial.begin(115200);

  for (int pin = 2; pin <= 53; pin++) {
    if (pin >= 14 && pin <= 21) continue;   // Serial1-3 / I2C — skipped by default
    if (isUsed(pin)) continue;
    pinMode(pin, INPUT_PULLUP);
    candidates[candidateCount] = pin;
    lastState[candidateCount] = digitalRead(pin);
    candidateCount++;
  }

  Serial.println("[scan] pin_scanner ready");
  Serial.print("[scan] watching ");
  Serial.print(candidateCount);
  Serial.println(" free digital pins for changes");
  Serial.println("[scan] trigger sensors ONE AT A TIME and watch which pin number reacts");
}

void loop() {
  for (int i = 0; i < candidateCount; i++) {
    bool now = digitalRead(candidates[i]);
    if (now != lastState[i]) {
      Serial.print("[scan] pin ");
      Serial.print(candidates[i]);
      Serial.println(now ? " -> HIGH (released)" : " -> LOW  (triggered)");
      lastState[i] = now;
    }
  }
  delay(15);
}
