# =============================================================================
#  THERMIS-X  |  Adaptive Thermocouple-Driven Industrial Thermal Intelligence
#  Author     :  Vaibhav Krishna V
#  Revision   :  2.0
#  Architecture:
#    Type-K Thermocouple Acquisition → Multi-Zone Physics Engine →
#    Neural Policy Network (PyTorch) → Adaptive Pump & Motor Speed Control →
#    Predictive Pre-Cooling → SCADA Dashboard (Flask + Plotly)
# =============================================================================

import time, random, threading, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webview
from flask import Flask, jsonify, render_template_string

print("╔══════════════════════════════════════════════════════════════╗")
print("║        THERMIS-X  |  Industrial Thermal Intelligence         ║")
print("║        Author: Vaibhav Krishna V  |  Rev 2.0                 ║")
print("╚══════════════════════════════════════════════════════════════╝")

# =============================================================================
# ZONE CONFIGURATION
# =============================================================================
ZONES = ["CPU", "PowerModule", "MotorA", "MotorB", "PumpA", "PumpB", "Ambient"]
ZONE_LABELS = {
    "CPU":         "CPU Core",
    "PowerModule": "Power Module",
    "MotorA":      "Motor Unit A",
    "MotorB":      "Motor Unit B",
    "PumpA":       "Pump Unit A",
    "PumpB":       "Pump Unit B",
    "Ambient":     "Ambient Air",
}

ACTUATORS = ["MotorA", "MotorB", "PumpA", "PumpB"]

# =============================================================================
# PHYSICAL & ELECTRICAL CONSTANTS
# =============================================================================
# Type-K Thermocouple (Chromel–Alumel) characteristics
TC_SENSITIVITY_UV_PER_C = 41.276   # µV/°C (Seebeck coefficient)
TC_ADC_NOISE_UV         = 0.45     # Gaussian ADC noise standard deviation
TC_COLD_JUNCTION_C      = 25.0     # Cold junction reference temperature °C
TC_SENSOR_LAG_ALPHA   = 0.18
TC_OFFSET_UV          = 1.2
TC_ADC_RESOLUTION_UV  = 0.5
TC_EMI_PROBABILITY    = 0.01
TC_FAULT_PROBABILITY = 0.001
THERMAL_CAPACITANCE = {
    "CPU": 0.42,
    "PowerModule": 0.88,
    "MotorA": 0.92,
    "MotorB": 0.92,
    "PumpA": 0.94,
    "PumpB": 0.94,
    "Ambient": 0.98
}
# Thermal thresholds
CRITICAL_TEMP  = 75.0   # °C → Emergency Cooling engaged
WARNING_TEMP   = 65.0   # °C → Throttled mode
TARGET_TEMP    = 45.0   # °C → Setpoint for AI controller
SPIKE_RATE     = 1.5    # °C/step spike detection threshold

# Motor / Pump rated parameters
MAX_RPM  = {"MotorA": 3000, "MotorB": 3000, "PumpA": 1800, "PumpB": 1800}
RATED_W  = {"MotorA": 200,  "MotorB": 200,  "PumpA": 150,  "PumpB": 150}
BASE_AMP = {"MotorA": 2.0,  "MotorB": 2.0,  "PumpA": 1.5,  "PumpB": 1.5}

# Initial temperatures per zone
BASE_TEMPS = {
    "CPU": 38.0, "PowerModule": 35.0,
    "MotorA": 32.0, "MotorB": 31.0,
    "PumpA": 29.0, "PumpB": 28.5,
    "Ambient": 26.0,
}

# Thermal drift base rates (°C/step) per zone under no-load
DRIFT_BASE = {
    "CPU": 0.48, "PowerModule": 0.42,
    "MotorA": 0.26, "MotorB": 0.23,
    "PumpA": 0.19, "PumpB": 0.17,
    "Ambient": 0.08,
}

# Cooling effectiveness per unit cooling command [0–1]
COOL_EFF = {
    "CPU": 4.2, "PowerModule": 1.6,
    "MotorA": 1.4, "MotorB": 1.3,
    "PumpA": 1.1, "PumpB": 1.0,
    "Ambient": 0.5,
}

# =============================================================================
# SHARED STATE  (thread-safe with GIL for CPython floats/dicts)
# =============================================================================
temps       = BASE_TEMPS.copy()
last_temps  = temps.copy()
history     = {z: collections.deque(maxlen=160) for z in ZONES}
cooling     = {z: 0.30 for z in ZONES}    # per-zone cooling command [0–1]
mode        = "Initializing"

# Thermocouple acquisition results
tc_mv       = {z: 0.0 for z in ZONES}    # mV output
tc_degc     = {z: 0.0 for z in ZONES}    # recovered °C (with CJC + noise)
tc_filtered = BASE_TEMPS.copy()
# Actuator states
actuator_rpm     = {a: 0.0    for a in ACTUATORS}
actuator_state   = {a: "IDLE" for a in ACTUATORS}
actuator_current = {a: 0.0    for a in ACTUATORS}   # Amperes
actuator_on      = {a: True   for a in ACTUATORS}   # automation flag

# Power & energy bookkeeping
power_draw_w     = 0.0
energy_saved_kwh = 0.0

# Analytics
stability        = {z: "INIT"  for z in ZONES}
prev_stability   = stability.copy()
stability_score  = {z: 50.0    for z in ZONES}
ttc              = {z: -1      for z in ZONES}   # steps to CRITICAL_TEMP
ai_confidence    = 0.0
sim_time         = 0

# Rolling alerts
alerts = collections.deque(maxlen=25)
last_alert_time = {}
ALERT_COOLDOWN = 15
def push_alert(key, msg):
    now = sim_time
    if now - last_alert_time.get(key, -999) >= ALERT_COOLDOWN:
        alerts.appendleft(msg)
        last_alert_time[key] = now
# =============================================================================
# TYPE-K THERMOCOUPLE ENGINE
# =============================================================================
def read_thermocouple(zone: str):
    """
    Realistic Type-K acquisition:
    - thermal lag
    - ADC quantisation
    - amplifier offset
    - Gaussian noise
    - occasional EMI spike
    """
    actual = temps[zone]
    if random.random() < TC_FAULT_PROBABILITY:
      return 0.0, -999.0

    tc_filtered[zone] += (actual - tc_filtered[zone]) * TC_SENSOR_LAG_ALPHA
    sensed = tc_filtered[zone]

    noise = random.gauss(0.0, TC_ADC_NOISE_UV)
    emi = random.uniform(-8.0, 8.0) if random.random() < TC_EMI_PROBABILITY else 0.0

    v_uv = (
        TC_SENSITIVITY_UV_PER_C * (sensed - TC_COLD_JUNCTION_C)
        + noise
        + TC_OFFSET_UV
        + emi
    )

    v_uv = round(v_uv / TC_ADC_RESOLUTION_UV) * TC_ADC_RESOLUTION_UV

    v_mv = v_uv / 1000.0
    t_rec = (v_uv / TC_SENSITIVITY_UV_PER_C) + TC_COLD_JUNCTION_C

    return round(v_mv, 4), round(t_rec, 2)
def update_thermocouples():
    for z in ZONES:
        mv, degc       = read_thermocouple(z)
        tc_mv[z]       = mv
        tc_degc[z]     = degc

# =============================================================================
# PUMP & MOTOR AUTOMATED CONTROLLER
# =============================================================================
def update_actuators():
    """
    Continuous automated actuator loop:
    ─ Target RPM derived from thermal urgency + neural cooling command.
    ─ Slew-rate limited ramp (150 RPM/step) for motor protection.
    ─ State machine: IDLE → RUNNING → THROTTLED → FAULT
    ─ Current draw modelled by affinity-law scaling:  I ∝ RPM^1.5
    """
    for a in ACTUATORS:
        if not actuator_on[a]:
            actuator_rpm[a]     = 0.0
            actuator_state[a]   = "DISABLED"
            actuator_current[a] = 0.0
            continue

        zone_temp   = temps[a]
        cool_cmd    = cooling[a]

        # Thermal urgency [0–1] drives RPM independently of neural command
        urgency = float(np.clip((zone_temp - 30.0) / (CRITICAL_TEMP - 30.0), 0.0, 1.0))

        mode_factor = {
            "🌿 EFFICIENCY MODE": 0.22,
            "⚖ BALANCED CONTROL": 0.70,
            "🔥 PERFORMANCE COOLING": 1.0,
            "⚡ SPIKE SUPPRESSION": 1.0,
            "🔮 PREDICTIVE PRE-COOL": 0.85,
            "🚨 EMERGENCY COOLING": 1.0
        }.get(mode, 0.70)

        rpm_target = float(
            np.clip((cool_cmd * 0.55 + urgency * 0.85) * mode_factor, 0.08, 1.0)
        ) * MAX_RPM[a]

        # Slew-rate limited ramp
        delta = rpm_target - actuator_rpm[a]
        actuator_rpm[a] += float(np.clip(delta, -150.0, 150.0))
        actuator_rpm[a]  = max(0.0, actuator_rpm[a])

        # State machine logic
        if zone_temp >= CRITICAL_TEMP:
            actuator_state[a]   = "EMERGENCY"
            actuator_rpm[a]     = float(MAX_RPM[a])   # full-speed emergency
        elif zone_temp >= WARNING_TEMP:
            actuator_state[a]   = "BOOST"
            actuator_rpm[a]     = float(MAX_RPM[a])
        elif actuator_rpm[a] < 120.0:
            actuator_state[a]   = "IDLE"
        else:
            actuator_state[a]   = "RUNNING"

        # Current draw: pump/fan affinity law  P ∝ n³  →  I ∝ n^1.5
        rpm_ratio               = actuator_rpm[a] / MAX_RPM[a]
        actuator_current[a]     = round(BASE_AMP[a] * (rpm_ratio ** 1.5), 3)

# =============================================================================
# THERMAL PHYSICS ENGINE
# =============================================================================
def thermal_shock():
    """Stochastic industrial load spike."""
    if sim_time > 10 and random.random() < 0.006:
        zone = random.choice(["CPU", "MotorA", "MotorB"])
        delta = random.uniform(1.5, 4.0)

        temps[zone] += delta

        push_alert(
            f"shock_{zone}",
            f"⚡ Thermal shock → {ZONE_LABELS[zone]} Δ+{delta:.1f}°C"
        )

def thermal_coupling():
    """
    Inter-zone conductive heat transfer.
    Models PCB trace coupling and shared heatsink paths.
    """
    temps["PowerModule"] += (temps["CPU"]         - 42.0) * 0.010
    temps["MotorA"]      += (temps["PowerModule"] - 38.0) * 0.008
    temps["MotorB"]      += (temps["PowerModule"] - 38.0) * 0.007
    temps["PumpA"]       += (temps["MotorA"]      - 36.0) * 0.005
    temps["PumpB"]       += (temps["MotorB"]      - 36.0) * 0.004
    temps["Ambient"]     += (temps["PumpA"] + temps["PumpB"] - 60.0) * 0.002

def thermal_drift(zone: str) -> float:
    """
    Per-zone temperature drift per simulation step.
    Combines base heat generation, Gaussian noise, and
    cooling contribution (including pump/motor heat rejection penalty).
    """
    base   = DRIFT_BASE[zone]
    noise  = random.gauss(0.0, 0.14)
    temp_excess = max(0.0, temps[zone] - temps["Ambient"])
    cool = cooling[zone] * COOL_EFF[zone] * (temp_excess / 40.0)

    # Actuator self-heating: at full speed, motor/pump adds ~0.04 °C/step
    if zone in ACTUATORS:
        rpm_ratio = actuator_rpm[zone] / MAX_RPM[zone]
        cool     -= rpm_ratio * 0.04     # net still beneficial

    return base + noise - cool

# =============================================================================
# THERMAL STABILITY ANALYSER
# =============================================================================
def analyse_stability():
    """
    Rolling statistical analysis on last 20 temperature samples per zone.
    Stability is classified by standard deviation and linear trend magnitude.
    """
    for z in ZONES:
        old_state = stability[z]
        buf = list(history[z])
        if len(buf) < 10:
            stability[z]       = "INIT"
            stability_score[z] = 50.0
            continue

        window  = buf[-20:]
        std_val = float(np.std(window))
        trend   = float(window[-1] - window[0])

        if std_val < 1.8 and abs(trend) < 1.8:
            stability[z] = "STABLE"
            stability_score[z] = float(np.clip(100.0 - std_val * 7.0, 75.0, 100.0))

        elif std_val < 3.5 or abs(trend) < 4.0:
            stability[z] = "TRANSIENT"
            stability_score[z] = float(np.clip(72.0 - std_val * 10.0, 35.0, 72.0))

        else:
            stability[z] = "UNSTABLE"
            stability_score[z] = float(np.clip(40.0 - std_val * 8.0, 0.0, 40.0))

            if (
                z != "Ambient"
                and old_state != "UNSTABLE"
                and stability[z] == "UNSTABLE"
            ):
                push_alert(
                    f"unstable_{z}",
                    f"⚠ Instability detected in {ZONE_LABELS[z]}"
                )

# =============================================================================
# TIME-TO-CRITICAL PREDICTOR (Linear Extrapolation)
# =============================================================================
def predict_time_to_critical():
    """
    Fits a linear model to the last 30 temperature samples and extrapolates
    the number of steps until CRITICAL_TEMP is reached.
    Returns -1 if temperature is not converging toward critical.
    """
    for z in ZONES:
        buf = list(history[z])
        if len(buf) < 20:
            ttc[z] = -1
            continue

        window = np.array(buf[-30:], dtype=np.float32)
        window = np.convolve(window, np.ones(5)/5, mode='valid')
        xs     = np.arange(len(window), dtype=np.float32)
        slope, intercept = np.polyfit(xs, window, 1)

        if slope <= 0.005:
            ttc[z] = -1
        else:
            steps_left = (CRITICAL_TEMP - window[-1]) / slope
            ttc[z]     = max(0, int(steps_left))

# =============================================================================
# POWER OPTIMIZER  (Pump Affinity Law: P ∝ RPM³)
# =============================================================================
def compute_power_and_savings():
    """
    Computes instantaneous power draw using pump/motor affinity law
    and accumulates energy savings versus an always-full-speed baseline.
    """
    global power_draw_w, energy_saved_kwh

    baseline_w = sum(RATED_W[a] for a in ACTUATORS)    # 700 W at full speed
    actual_w   = 0.0

    for a in ACTUATORS:
        rpm_ratio  = actuator_rpm[a] / MAX_RPM[a]
        # Affinity law: P = P_rated × (n/n_rated)³  +  idle standby load
        standby_factor = 0.08
        actual_w += RATED_W[a] * (rpm_ratio ** 3 + standby_factor)

    power_draw_w    = round(actual_w, 2)
    saved_this_step = max(0.0, (baseline_w - actual_w)) / 3600.0 / 1000.0   # kWh
    energy_saved_kwh += saved_this_step

# =============================================================================
# HYBRID ADAPTIVE NEURAL THERMAL CONTROLLER
# =============================================================================
class ThermalPolicyNet(nn.Module):
    """
    Input  : 14-dimensional state vector
             [T_1..T_7 normalised]  +  [StabilityScore_1..7 normalised]
    Output : 7-dimensional cooling command vector [0, 1] per zone
    Architecture: Encoder with LayerNorm → Sigmoid output head
    """
    def __init__(self, n_zones: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_zones * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, n_zones),
            nn.Sigmoid()          # cooling commands in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

N_ZONES    = len(ZONES)
policy_net = ThermalPolicyNet(N_ZONES)
optimizer  = optim.Adam(policy_net.parameters(), lr=5e-4, weight_decay=1e-5)

def build_state_vector() -> torch.Tensor:
    t_vec = torch.tensor([temps[z] / 100.0         for z in ZONES], dtype=torch.float32)
    s_vec = torch.tensor([stability_score[z] / 100.0 for z in ZONES], dtype=torch.float32)
    return torch.cat([t_vec, s_vec])

def ai_policy_step():
    """
    Inference pass of the neural policy.
    Cooling command = blend of neural output and a rule-based floor
    that guarantees minimum cooling proportional to temperature excess.
    """
    global ai_confidence

    state = build_state_vector()
    with torch.no_grad():
        out = policy_net(state).numpy()

    for i, z in enumerate(ZONES):
        if z == "Ambient":
            continue
        ai_cmd    = float(out[i])
        rule_floor = float(np.clip((temps[z] - 35.0) / (CRITICAL_TEMP - 35.0), 0.05, 1.0))
        # Weighted blend: 60% neural, 40% rule-based floor
        cooling[z] = float(np.clip(ai_cmd * 0.60 + rule_floor * 0.40, 0.05, 1.0))

    # Confidence metric: 0 when average deviation from setpoint is high
    mean_dev = float(np.mean([
        abs(temps[z] - TARGET_TEMP)
        for z in ZONES if z != "Ambient"
    ]))

    avg_cool = sum(
        cooling[z] for z in ZONES if z != "Ambient"
    ) / (N_ZONES - 1)

    unstable_count = sum(
        1 for z in ZONES
        if stability[z] in ("TRANSIENT", "UNSTABLE")
    )

    max_rise = 0.0
    for z in ZONES:
        if z == "Ambient":
            continue
        buf = history[z]
        if len(buf) >= 5:
            rise = abs(buf[-1] - buf[-5])
            max_rise = max(max_rise, rise)

    thermal_score = max(0.0, 100.0 - mean_dev * 2.0)

    stability_avg = np.mean([
        stability_score[z]
        for z in ZONES if z != "Ambient"
    ])

    instability_penalty = unstable_count * 8.0
    rise_penalty = max_rise * 8.0

    ai_confidence = round(
        float(np.clip(
            thermal_score * 0.55 +
            stability_avg * 0.45 -
            instability_penalty -
            rise_penalty,
            0.0,
            100.0
        )),
        2
    )

def train_policy_step():
    """
    Online policy gradient update with composite reward:
      R = − λ_temp × T_max/80  − λ_cool × mean(cooling) × (1/urgency)  + λ_stab
    Higher urgency relaxes the cooling penalty (aggressive cooling tolerated when hot).
    """
    state     = build_state_vector()
    out       = policy_net(state)

    max_temp  = max(temps.values())
    avg_cool  = sum(cooling.values()) / N_ZONES
    avg_stab  = float(np.mean(list(stability_score.values()))) / 100.0

    # Adaptive penalty weights
    lam_temp  = float(np.interp(max_temp, [30.0, 80.0], [0.5,  3.0]))
    lam_cool  = float(np.interp(max_temp, [30.0, 80.0], [1.5,  0.2]))
    lam_stab  = 0.5

    reward = (
        -max_temp * 0.01 * lam_temp
        - avg_cool * lam_cool
        + avg_stab * lam_stab
    )
    loss = -reward * out.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# =============================================================================
# SAFETY, SPIKE DETECTION & PREDICTIVE PRE-COOLING
# =============================================================================
def safety_override() -> bool:
    """Hard safety layer: maximum cooling on all zones above CRITICAL_TEMP."""
    global mode
    critical = [z for z in ZONES if temps[z] > CRITICAL_TEMP]
    if critical:
        for z in ZONES:
            if z != "Ambient":
                cooling[z] = 1.0
        mode = "🚨 EMERGENCY COOLING"
        push_alert(
            "critical",
            "🚨 CRITICAL: " + ", ".join(ZONE_LABELS[z] for z in critical)
        )
        return True
    return False

def spike_detector() -> bool:
    """Detects rapid temperature rise and applies boost cooling to affected zones."""
    global mode
    spikes = [z for z in ZONES if (temps[z] - last_temps.get(z, temps[z])) > SPIKE_RATE]
    if spikes:
        for z in spikes:
            cooling[z] = min(1.0, cooling[z] + 0.35)

        zone_names = ", ".join(ZONE_LABELS[z] for z in spikes)
        push_alert("spike", f"⚡ Rapid thermal rise → {zone_names}")

        mode = "⚡ SPIKE SUPPRESSION"
        return True
    return False

def predictive_precool():
    """
    If any zone is projected to reach CRITICAL_TEMP within 35 steps,
    pre-emptively ramp cooling to avoid emergency escalation.
    """
    global mode
    precool = [z for z in ZONES if 0 < ttc[z] < 35]
    if precool:
        for z in precool:
            cooling[z] = min(1.0, cooling[z] + 0.20)

        zone_names = ", ".join(ZONE_LABELS[z] for z in precool)
        push_alert("precool", f"🔮 Predictive precool → {zone_names}")

        mode = "🔮 PREDICTIVE PRE-COOL"

def anti_windup():
    """
    Anti-windup relaxation:
    Only relax cooling in efficiency mode.
    Never fight active thermal control in balanced/performance modes.
    """
    global mode

    if mode != "🌿 EFFICIENCY MODE":
        return

    avg_temp = sum(temps.values()) / N_ZONES

    for z in ZONES:
        if avg_temp < 38.0:
            cooling[z] *= 0.94
        elif avg_temp < 48.0:
            cooling[z] *= 0.98

        cooling[z] *= 0.993
        cooling[z] = max(0.05, cooling[z])

# =============================================================================
# MASTER CONTROLLER PIPELINE
# =============================================================================
def master_controller():
    global last_temps

    last_temps = temps.copy()

    if safety_override():
        update_actuators()
        return

    if spike_detector():
        update_actuators()
        return

    predictive_precool()

    ai_policy_step()

    hottest = max(temps[z] for z in ZONES if z != "Ambient")

    if mode == "🔥 PERFORMANCE COOLING":
        for z in ZONES:
            if z != "Ambient":
                cooling[z] = max(cooling[z], 0.92)

    elif mode == "⚖ BALANCED CONTROL":
        for z in ZONES:
            if z != "Ambient":
                cooling[z] = max(cooling[z], 0.30)

    elif mode == "🌿 EFFICIENCY MODE":
        for z in ZONES:
            if z != "Ambient":
                cooling[z] = max(cooling[z], 0.20)

    anti_windup()
    update_actuators()

# =============================================================================
# SIMULATION LOOP (runs in background thread)
# =============================================================================
def sim_loop():
    global sim_time, mode

    # Bootstrap warm-up so UI doesn't start empty
    for _ in range(20):
        thermal_coupling()

        for z in ZONES:
            delta = thermal_drift(z)
            temps[z] += delta * (1.0 - THERMAL_CAPACITANCE[z])

            if z != "Ambient":
                min_temp = temps["Ambient"] + 0.8
                if temps[z] < min_temp:
                    temps[z] += (min_temp - temps[z]) * 0.35

            temps[z] = float(np.clip(temps[z], 20.0, 95.0))
            history[z].append(round(temps[z], 2))
        cooling["Ambient"] = 0.05
        update_thermocouples()
        analyse_stability()
        predict_time_to_critical()
        ai_policy_step()
        update_actuators()
        compute_power_and_savings()

        sim_time += 1

    mode = "🌿 EFFICIENCY MODE"

    while True:
        thermal_shock()
        thermal_coupling()

        master_controller()

        for z in ZONES:
            delta = thermal_drift(z)
            temps[z] += delta * (1.0 - THERMAL_CAPACITANCE[z])
            temps[z] = float(np.clip(temps[z], 20.0, 95.0))
            history[z].append(round(temps[z], 2))

        update_thermocouples()
        analyse_stability()
        predict_time_to_critical()
        compute_power_and_savings()
        train_policy_step()

        if mode not in [
            "🚨 EMERGENCY COOLING",
            "⚡ SPIKE SUPPRESSION",
            "🔮 PREDICTIVE PRE-COOL"
        ]:
            hottest = max(temps[z] for z in ZONES if z != "Ambient")

            max_rise = 0.0
            for z in ZONES:
                if z == "Ambient":
                    continue
                buf = history[z]
                if len(buf) >= 5:
                    rise = buf[-1] - buf[-5]
                    max_rise = max(max_rise, rise)

            # hysteresis-based industrial mode control
            if mode == "🔥 PERFORMANCE COOLING":
                if hottest < 40.0 and max_rise < 0.30:
                    mode = "⚖ BALANCED CONTROL"

            elif mode == "⚖ BALANCED CONTROL":
                if hottest >= 44.0 or max_rise >= 0.90:
                    mode = "🔥 PERFORMANCE COOLING"
                elif hottest < 33.0 and max_rise < 0.20:
                    mode = "🌿 EFFICIENCY MODE"

            elif mode == "🌿 EFFICIENCY MODE":
                if hottest >= 36.0 or max_rise >= 0.50:
                    mode = "⚖ BALANCED CONTROL"

            else:
                if hottest >= 48.0:
                    mode = "🔥 PERFORMANCE COOLING"
                elif hottest >= 38.0:
                    mode = "⚖ BALANCED CONTROL"
                else:
                    mode = "🌿 EFFICIENCY MODE"
        sim_time += 1
        time.sleep(1)

# =============================================================================
# SYSTEM-LEVEL METRICS
# =============================================================================
def overall_energy_efficiency() -> int:
    baseline = sum(RATED_W[a] for a in ACTUATORS)
    return int(np.clip(100.0 - (power_draw_w / baseline) * 100.0, 0.0, 100.0))

def overall_thermal_safety() -> int:
    return int(np.interp(max(temps.values()), [30.0, 85.0], [100.0, 5.0]))

# =============================================================================
# SCADA DASHBOARD  (HTML / CSS / JS  –  Served by Flask)
# =============================================================================
DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>THERMIS-X  |  Industrial Thermal Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:ital,wght@0,300;0,400;0,600;0,700;0,900;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{
  --bg:#03070f;--panel:#060d1a;--panel2:#08101f;
  --border:#0e1e34;--border2:#162640;
  --accent:#ff6d00;--accent2:#ff9e40;
  --cyan:#00e5ff;--cyan2:#80f0ff;
  --green:#00e676;--yellow:#ffd600;--red:#ff1744;--purple:#d500f9;
  --text:#9ab8d8;--text2:#c8ddf0;--dim:#5f7ea3;
  --mono:'Consolas','Courier New',monospace;--sans:'Segoe UI',Arial,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{
  background:var(--bg);color:var(--text);font-family:var(--sans);
  background-image:
    linear-gradient(rgba(255,109,0,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,109,0,.018) 1px,transparent 1px);
  background-size:48px 48px;
}

/* ── HEADER ─────────────────────────────────────────────────────── */
.hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:9px 22px;
  background:linear-gradient(90deg,#03070f,#060d1a 40%,#060d1a 60%,#03070f);
  border-bottom:1px solid var(--accent);
  position:relative;z-index:10;
}
.logo{
  font-family:var(--sans);font-weight:900;font-size:19px;
  letter-spacing:5px;color:var(--accent);
  text-shadow:0 0 24px rgba(255,109,0,.5);
}
.logo em{color:var(--cyan);font-style:normal}
.logo sup{
  font-size:8px;color:var(--dim);letter-spacing:1px;
  vertical-align:super;margin-left:4px;
}
.mode-pill{
  font-family:var(--mono);font-size:12px;letter-spacing:2px;
  padding:5px 18px;border:1px solid var(--accent);border-radius:2px;
  color:var(--accent2);background:rgba(255,109,0,.05);
  animation:glowpulse 2.4s ease-in-out infinite;
}
@keyframes glowpulse{
  0%,100%{box-shadow:0 0 0 rgba(255,109,0,0)}
  50%{box-shadow:0 0 12px rgba(255,109,0,.25)}
}
.hdr-right{display:flex;flex-direction:column;align-items:flex-end;gap:2px}
.sim-clock{font-family:var(--mono);font-size:13px;  color:#7aa6d8;letter-spacing:1px}
.author{font-size:10px; color:#4f7398;letter-spacing:2px;font-weight:600}

/* ── ALERT BAR ───────────────────────────────────────────────────── */
.alert-bar{
  height:23px;overflow:hidden;
  background:rgba(255,23,68,.06);border-bottom:1px solid rgba(255,23,68,.2);
  padding:4px 22px;font-family:var(--mono);font-size:12px; font-weight:600;color:#ff8aa0;
  letter-spacing:.5px;white-space:nowrap;display:flex;align-items:center;gap:12px;
}

/* ── MAIN GRID ───────────────────────────────────────────────────── */
.main{
  display:grid;
  grid-template-columns:270px 1fr 280px;
  grid-template-rows:calc(100vh - 164px);
  gap:1px;background:var(--border);
}
.col{background:var(--panel);overflow:hidden;display:flex;flex-direction:column}

/* ── SECTION TITLE ───────────────────────────────────────────────── */
.sec{
  padding:8px 14px 6px;
  border-bottom:1px solid var(--border);
  font-size:11px;font-weight:700;letter-spacing:1.2px;
  color:var(--accent);text-transform:uppercase;flex-shrink:0;
}
.sec-icon{margin-right:5px;opacity:.7}

/* ── THERMOCOUPLE TABLE ──────────────────────────────────────────── */
.tc-wrap{overflow-y:auto;flex:1}
.tc-tbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px}
.tc-tbl th{
  padding:5px 10px;color:var(--dim);font-size:10.5px;letter-spacing:1px;
  border-bottom:1px solid var(--border);text-align:right;
}
.tc-tbl th:first-child{text-align:left}
.tc-tbl td{
  padding:6px 10px;border-bottom:1px solid rgba(14,30,52,.6);
  text-align:right;
}
.tc-tbl td:first-child{
  text-align:left;color:#3a5c7a;font-size:11.5px;letter-spacing:.5px
}
.badge{
  display:inline-block;padding:2px 6px;border-radius:2px;
  font-size:10px; padding:3px 8px;letter-spacing:1px;font-weight:700;
}
.stb{background:rgba(0,230,118,.12);color:var(--green)}
.trn{background:rgba(255,214,0,.12);color:var(--yellow)}
.uns{background:rgba(255,23,68,.12);color:var(--red)}
.ini{background:rgba(42,64,96,.3);color:var(--dim)}

/* ── COOLING COMMAND BARS ────────────────────────────────────────── */
.cool-sec{flex-shrink:0}
.cool-list{padding:8px 14px}
.cr{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.cl{font-family:var(--mono);font-size:9.5px;color:var(--dim);width:60px;flex-shrink:0}
.cb{flex:1;height:5px;background:rgba(255,255,255,.04);border-radius:3px;overflow:hidden}
.cf{height:100%;border-radius:3px;transition:width .6s ease}
.cp{font-family:var(--mono);font-size:9.5px;color:#2a4060;width:30px;text-align:right;flex-shrink:0}

/* ── CENTER CHART ────────────────────────────────────────────────── */
.chart-col{flex:1;min-height:0}
#main-chart{width:100%;height:100%}

/* ── ACTUATOR CARDS ──────────────────────────────────────────────── */
.act-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:7px;
  padding:8px;flex-shrink:0;
}
.ac{
  background:rgba(255,255,255,.018);border:1px solid var(--border2);
  border-radius:3px;padding:9px;position:relative;overflow:hidden;
}
.ac::before{content:'';position:absolute;top:0;left:0;right:0;height:2px}
.ac.run::before{background:var(--green)}
.ac.idle::before{background:var(--border2)}
.ac.thr::before{background:var(--yellow)}
.ac.flt::before{background:var(--red)}
.ac.dis::before{background:#111}
.an{font-size:11px;font-weight:700;letter-spacing:2px;color:var(--dim);margin-bottom:5px}
.ar{font-family:var(--mono);font-size:19px;color:var(--cyan);line-height:1}
.au{font-size:9px;color:var(--border2);margin-left:2px}
.ai{font-family:var(--mono);font-size:12px;color:var(--accent);margin-top:3px}
.as{font-family:var(--mono);font-size:10px;letter-spacing:1px;margin-top:2px}
.run-s{color:var(--green)}
.idle-s{color:var(--dim)}
.thr-s{color:var(--yellow)}
.flt-s{color:var(--red);animation:blink .5s infinite}
.dis-s{color:#1a1a1a}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
.rg{width:100%;height:3px;background:rgba(255,255,255,.04);border-radius:2px;overflow:hidden;margin-top:6px}
.rf{height:100%;background:linear-gradient(90deg,var(--cyan),var(--accent));border-radius:2px;transition:width .5s ease}

/* ── STABILITY TABLE ─────────────────────────────────────────────── */
.stab-wrap{overflow-y:auto;flex:1}

/* ── BOTTOM ROW ──────────────────────────────────────────────────── */
.brow{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:1px;background:var(--border);
  border-top:1px solid var(--border);height:105px;flex-shrink:0;
}
.mp{background:var(--panel2);padding:11px 16px;display:flex;flex-direction:column;justify-content:center}
.ml{font-size:10px;font-weight:700;letter-spacing:2px;color:var(--dim);text-transform:uppercase;margin-bottom:5px}
.mv{font-family:var(--mono);font-size:24px;line-height:1;color:var(--cyan)}
.ms{font-family:var(--mono);font-size:10px;color:var(--dim);margin-top:3px}
.pb{width:100%;height:3px;background:rgba(255,255,255,.03);border-radius:2px;overflow:hidden;margin-top:7px}
.pf{height:100%;border-radius:2px;transition:width .7s ease}

/* ── TTC LIST ────────────────────────────────────────────────────── */
.ttc-inner{font-family:var(--mono);font-size:10px}
.tti{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid rgba(14,30,52,.5)}
.ttz{color:var(--dim);font-size:9px}
.ttv{color:var(--yellow)}
.ttn{color:#0d1e2e}

/* ── SCROLLBAR ───────────────────────────────────────────────────── */
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="logo">THERMIS<em>-X</em><sup>INDUSTRIAL THERMAL AI</sup></div>
  <div id="mode-pill" class="mode-pill">INITIALIZING</div>
  <div class="hdr-right">
    <div class="sim-clock" id="sim-clock">T+0000s</div>
    <div class="author">VAIBHAV KRISHNA V</div>
  </div>
</div>

<!-- ALERT BAR -->
<div id="alert-bar" class="alert-bar">[ SYSTEM NOMINAL  ·  ALL ZONES WITHIN LIMITS ]</div>

<!-- MAIN GRID -->
<div class="main">

  <!-- COL 1: THERMOCOUPLE + COOLING -->
  <div class="col">
    <div class="sec"><span class="sec-icon">⬡</span>Type-K Thermocouple Acquisition</div>
    <div class="tc-wrap">
      <table class="tc-tbl">
        <thead><tr>
          <th>Zone</th><th>°C</th><th>mV</th><th>Stability</th>
        </tr></thead>
        <tbody id="tc-body"></tbody>
      </table>
    </div>
    <div class="cool-sec">
      <div class="sec"><span class="sec-icon">⬡</span>Cooling Commands  (% duty)</div>
      <div class="cool-list" id="cool-list"></div>
    </div>
  </div>

  <!-- COL 2: CHART -->
  <div class="col">
    <div class="sec"><span class="sec-icon">⬡</span>Multi-Zone Thermal Trend  ·  Last 120 s</div>
    <div class="chart-col"><div id="main-chart"></div></div>
  </div>

  <!-- COL 3: ACTUATORS + STABILITY -->
  <div class="col">
    <div class="sec"><span class="sec-icon">⬡</span>Pump & Motor Automation</div>
    <div class="act-grid" id="act-grid"></div>
    <div class="sec"><span class="sec-icon">⬡</span>Thermal Stability Matrix</div>
    <div class="stab-wrap">
      <table class="tc-tbl">
        <thead><tr><th>Zone</th><th>Score</th><th>State</th></tr></thead>
        <tbody id="stab-body"></tbody>
      </table>
    </div>
  </div>

</div><!-- /main -->

<!-- BOTTOM ROW -->
<div class="brow">
  <div class="mp">
    <div class="ml">System Power Draw</div>
    <div class="mv" id="pw-val">— W</div>
    <div class="ms" id="pw-sub">Affinity law  ·  P ∝ n³</div>
    <div class="pb"><div class="pf" id="pw-bar" style="background:var(--accent)"></div></div>
  </div>
  <div class="mp">
    <div class="ml">AI Control Effectiveness</div>
    <div class="mv" id="ai-val">—%</div>
    <div class="ms">Hybrid Neural Adaptive Controller</div>
    <div class="pb"><div class="pf" id="ai-bar" style="background:var(--cyan)"></div></div>
  </div>
  <div class="mp">
    <div class="ml">Cumulative Energy Saved</div>
    <div class="mv" id="es-val">— Wh</div>
    <div class="ms" id="es-sub">vs. always-on baseline</div>
    <div class="pb"><div class="pf" id="es-bar" style="background:var(--green);width:0"></div></div>
  </div>
  <div class="mp">
    <div class="ml">Time-to-Critical  (sec)</div>
    <div class="ttc-inner" id="ttc-inner"></div>
  </div>
</div>

<!-- ═══════════════  JAVASCRIPT  ═══════════════════════════════════ -->
<script>
const ZONES = ["CPU","PowerModule","MotorA","MotorB","PumpA","PumpB","Ambient"];
const ZS = {CPU:"CPU",PowerModule:"PWR MOD",MotorA:"MTR-A",MotorB:"MTR-B",
            PumpA:"PMP-A",PumpB:"PMP-B",Ambient:"AMB"};
const ZCOLORS = {
  CPU:"#ff1744",PowerModule:"#ff6d00",
  MotorA:"#ffd600",MotorB:"#76ff03",
  PumpA:"#00e5ff",PumpB:"#d500f9",Ambient:"#607d8b"
};
const ACTS = ["MotorA","MotorB","PumpA","PumpB"];
const MAX_RPM = {MotorA:3000,MotorB:3000,PumpA:1800,PumpB:1800};
let chartInited = false;

const tempCol = t =>
  t<45?'var(--cyan)':t<60?'var(--yellow)':t<75?'var(--accent)':'var(--red)';

const coolCol = p =>
  p<30?'var(--green)':p<60?'var(--yellow)':p<80?'var(--accent)':'var(--red)';

const scoreCol = s =>
  s>70?'var(--green)':s>40?'var(--yellow)':'var(--red)';

function badgeHtml(stab){
  const cls={STABLE:'stb',TRANSIENT:'trn',UNSTABLE:'uns',INIT:'ini',FAULT:'uns'}[stab]||'ini';
  return `<span class="badge ${cls}">${stab}</span>`;
}
function actClass(st){
  return {
    RUNNING:'run',
    IDLE:'idle',
    THROTTLED:'thr',
    BOOST:'thr',
    EMERGENCY:'flt',
    DISABLED:'dis'
  }[st]||'idle';
}

function stateClass(st){
  return {
    RUNNING:'run-s',
    IDLE:'idle-s',
    THROTTLED:'thr-s',
    BOOST:'thr-s',
    EMERGENCY:'flt-s',
    DISABLED:'dis-s'
  }[st]||'idle-s';
}

async function update(){
  let d;
  try{ d = await fetch('/data').then(r=>r.json()); }
  catch(e){ return; }

  /* Header */
  document.getElementById('mode-pill').textContent = d.mode;
  document.getElementById('sim-clock').textContent = 'T+' + String(d.sim_time).padStart(4,'0') + 's';

  /* Alert bar */
  if(d.alerts && d.alerts.length>0){
    document.getElementById('alert-bar').textContent =
      d.alerts.join('   ·   ');
  }else{
    document.getElementById('alert-bar').textContent =
      '[ SYSTEM NOMINAL  ·  ALL ZONES WITHIN LIMITS ]';
  }

  /* Thermocouple table */
  let tc='';
  for(const z of ZONES){
    const t=d.tc_degc[z], mv=d.tc_mv[z], stab=d.stability[z]||'INIT';
    tc+=`<tr>
      <td>${ZS[z]}</td>
      <td style="color:${t < -100 ? 'var(--red)' : tempCol(t)}">
        ${t < -100 ? 'FAULT' : t.toFixed(1)}
      </td>
      <td style="color:var(--dim)">${mv.toFixed(3)}</td>
      <td>${badgeHtml(stab)}</td>
    </tr>`;
  }
  document.getElementById('tc-body').innerHTML=tc;

  /* Cooling bars */
  let cl='';
  for(const z of ZONES){
    const p=d.cooling[z];
    cl+=`<div class="cr">
      <div class="cl">${ZS[z]}</div>
      <div class="cb"><div class="cf" style="width:${p}%;background:${coolCol(p)}"></div></div>
      <div class="cp">${p.toFixed(0)}%</div>
    </div>`;
  }
  document.getElementById('cool-list').innerHTML=cl;

  /* Actuator cards */
  let ac='';
  for(const a of ACTS){
    const rpm=d.actuator_rpm[a], st=d.actuator_state[a];
    const amps=d.actuator_current[a], rp=(rpm/MAX_RPM[a]*100).toFixed(0);
    ac+=`<div class="ac ${actClass(st)}">
      <div class="an">${a}</div>
      <div class="ar">${Math.round(rpm)}<span class="au">RPM</span></div>
      <div class="ai">${amps.toFixed(3)} A</div>
      <div class="as ${stateClass(st)}">${st}</div>
      <div class="rg"><div class="rf" style="width:${rp}%"></div></div>
    </div>`;
  }
  document.getElementById('act-grid').innerHTML=ac;

  /* Stability table */
  let sb='';
  for(const z of ZONES){
    const sc=d.stability_score[z], stab=d.stability[z];
    sb+=`<tr>
      <td>${ZS[z]}</td>
      <td style="color:${scoreCol(sc)}">${sc.toFixed(0)}</td>
      <td>${badgeHtml(stab)}</td>
    </tr>`;
  }
  document.getElementById('stab-body').innerHTML=sb;

  /* Bottom: Power */
  const pw=d.power_draw_w;
  document.getElementById('pw-val').textContent = pw.toFixed(0)+' W';
  document.getElementById('pw-bar').style.width = Math.min(100,pw/700*100).toFixed(0)+'%';

  /* Bottom: AI */
  const ai=d.ai_confidence;
  const aiEl=document.getElementById('ai-val');
  aiEl.textContent=ai.toFixed(1)+'%';
  aiEl.style.color=ai>70?'var(--cyan)':ai>40?'var(--yellow)':'var(--red)';
  document.getElementById('ai-bar').style.width=ai+'%';

  /* Bottom: Energy saved */
  const es=d.energy_saved_kwh;
  document.getElementById('es-val').textContent=(es*1000).toFixed(2)+' Wh';
  document.getElementById('es-bar').style.width=Math.min(100,es*10000).toFixed(0)+'%';

  /* Bottom: TTC */
  let ttcH='', anyTTC=false;
  for(const z of ZONES){
    const t=d.ttc[z];
    if(t>0 && t<120){
      ttcH+=`<div class="tti"><span class="ttz">${ZS[z]}</span><span class="ttv">${t}s</span></div>`;
      anyTTC=true;
    }
  }
  if(!anyTTC) ttcH='<div class="tti"><span class="ttn" style="color:var(--dim)">ALL ZONES NOMINAL</span></div>';
  document.getElementById('ttc-inner').innerHTML=ttcH;

  /* Main Plotly chart */
  const traces = ZONES.map(z=>({
    y: d.history[z],
    type:'scatter', mode:'lines',
    name: ZS[z],
    line:{color:ZCOLORS[z], width:2}
  }));

  const layout = {
    template:'plotly_dark',
    paper_bgcolor:'#060d1a', plot_bgcolor:'#060d1a',hovermode:'x unified',
    margin:{l:46,r:20,t:6,b:36},
    yaxis:{title:'°C',range:[20,90],gridcolor:'#0e1e34',color:'#6f95bd',titlefont:{size:10}},
    xaxis:{gridcolor:'#0e1e34',color:'#6f95bd',showticklabels:false},
    legend:{orientation:'h',y:1.1,x:.5,xanchor:'center',
            font:{size:9,color:'#3a5c7a'},bgcolor:'rgba(0,0,0,0)'},
    shapes:[
      {type:'rect',xref:'paper',x0:0,x1:1,y0:20,y1:45,
       fillcolor:'rgba(0,230,118,.025)',line:{width:0}},
      {type:'rect',xref:'paper',x0:0,x1:1,y0:45,y1:60,
       fillcolor:'rgba(255,214,0,.025)',line:{width:0}},
      {type:'rect',xref:'paper',x0:0,x1:1,y0:60,y1:90,
       fillcolor:'rgba(255,23,68,.035)',line:{width:0}},
      {type:'line',xref:'paper',x0:0,x1:1,y0:75,y1:75,
       line:{color:'rgba(255,23,68,.45)',dash:'dot',width:1}}
    ]
  };

  if(!chartInited){
    Plotly.newPlot('main-chart',traces,layout,{responsive:true,displayModeBar:false, scrollZoom:false});
    chartInited=true;
  } else {
    Plotly.react('main-chart',traces,layout,{
      responsive:true,
      displayModeBar:false,
      scrollZoom:false
    });
  }
}

setTimeout(update, 1200);
setInterval(update,1000);
</script>
</body>
</html>"""

# =============================================================================
# FLASK ROUTES
# =============================================================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string(DASHBOARD)

@app.route("/data")
def data():
    return jsonify({
        # Temperatures & thermocouple readings
        "temps":      {z: round(temps[z], 2)    for z in ZONES},
        "tc_mv":      {z: tc_mv[z]               for z in ZONES},
        "tc_degc":    {z: tc_degc[z]             for z in ZONES},
        # Per-zone cooling commands (0–100 %)
        "cooling":    {z: round(cooling[z] * 100, 1) for z in ZONES},
        # Actuator telemetry
        "actuator_rpm":     {a: round(actuator_rpm[a], 0)    for a in ACTUATORS},
        "actuator_state":   actuator_state,
        "actuator_current": actuator_current,
        # Stability analytics
        "stability":        stability,
        "stability_score":  {z: round(stability_score[z], 1) for z in ZONES},
        # Predictive analytics
        "ttc":              ttc,
        # Power & energy
        "power_draw_w":     power_draw_w,
        "energy_saved_kwh": round(energy_saved_kwh, 5),
        # AI
        "ai_confidence":    ai_confidence,
        # System
        "mode":             mode,
        "energy_eff":       overall_energy_efficiency(),
        "thermal_safety":   overall_thermal_safety(),
        "alerts": [] if sim_time < 5 else list(dict.fromkeys(alerts))[:2],
        "history":          {z: list(history[z])[-120:] for z in ZONES},
        "sim_time":         sim_time,
        "zone_labels":      ZONE_LABELS,
    })

# =============================================================================
# ENTRY POINT
# =============================================================================
def start_flask():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )

if __name__ == "__main__":
    bg = threading.Thread(target=sim_loop, daemon=True)
    bg.start()

    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    time.sleep(1)

    webview.create_window(
        "THERMIS-X Industrial Thermal Intelligence",
        "http://127.0.0.1:5000",
        width=1920,
        height=1080,
        resizable=False,
        fullscreen=True
        )

    webview.start()
