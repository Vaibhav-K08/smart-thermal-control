# THERMIS-X: Adaptive Thermocouple-Driven Industrial Thermal Intelligence

<div align="center">

![Mode](https://img.shields.io/badge/Revision-2.0-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Neural%20Policy-red?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-SCADA%20Server-black?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**Author:** Vaibhav Krishna V &nbsp;|&nbsp; **Architecture:** Sensor Acquisition → Physics Engine → Neural Controller → SCADA Dashboard

</div>

---

## What This Is

THERMIS-X is a real-time industrial thermal management system built entirely in Python. It simulates at physics level: a seven-zone heat-generating machine: a CPU core, a power module, two motor units, two pump units, and an ambient reference. Across all of these, it runs Type-K thermocouple acquisition with realistic noise, a neural policy network that learns to cool each zone optimally, slew-rate-limited pump and motor speed control, and a predictive pre-cooling mechanism that acts before temperatures get dangerous rather than after.

The entire pipeline: sensor acquisition, physics simulation, AI inference, online training, and a live SCADA dashboard; runs in under 80 lines of critical control code per cycle, with no external PLC hardware required.

This is not a wrapper around a thermal API. Every physical constant, every noise characteristic, every actuator slew limit, and every reward term is grounded in engineering reality.

---

## Why I Built It This Way

Most student thermal projects stop at a PID loop. The honest problem with that is PID doesn't know that Motor A and the power module share a heatsink path, so heating one will raise the other. It doesn't understand that a temperature measurement from a thermocouple includes cold-junction drift and EMI noise. And it absolutely cannot tell you: *"if I do nothing, zone X will hit critical in 27 seconds."*

I wanted a system that reasons about the whole machine, not just one zone at a time. The neural policy receives the thermal state and stability score of all seven zones simultaneously, so it can discover coupling relationships during online training that a hand-tuned PID would miss entirely. The predictive layer then sits on top as a safety net: even if the neural policy is momentarily uncertain, linear extrapolation on the last 30 samples gives an honest time-to-critical estimate that triggers pre-emptive cooling.

The result is a system that, in normal operation, runs its pumps and motors at a fraction of full speed; saving real energy while remaining milliseconds away from full-blast emergency response when thermal conditions actually demand it.

---

## System Architecture

```
Type-K Thermocouple Acquisition
        │
        ▼
  Thermal Physics Engine  ←──── Stochastic Load Shocks
  (7-zone coupling model)
        │
        ▼
 Stability Analyser  ──────────► Time-to-Critical Predictor
        │                                  │
        ▼                                  ▼
 Hybrid Neural-Adaptive Controller ◄── Predictive Pre-Cool
 (Policy Net + Rule-Based Floor)
        │
        ▼
 Pump & Motor Speed Controller
 (Slew-Rate Limited · Affinity Law Power Model)
        │
        ▼
 Flask REST API  →  Plotly SCADA Dashboard  →  pywebview Desktop Window
```

The control pipeline runs once per second on a background thread. Flask serves live telemetry at `/data` as JSON. The dashboard polls at 1 Hz and updates everything: thermocouple readings, actuator states, stability scores, trend charts, and the time-to-critical panel; without a page reload.

---

## Engineering Depth

### Type-K Thermocouple Acquisition

The thermocouple engine does not simply return temperature. It models the full acquisition chain of a real Chromel–Alumel sensor:

- **Seebeck coefficient:** 41.276 µV/°C
- **Thermal lag:** exponential filter with α = 0.18, matching the thermal time constant of a physically mounted sensor
- **ADC quantisation:** 0.5 µV resolution, rounded before conversion
- **Amplifier offset:** fixed 1.2 µV baseline error
- **Gaussian ADC noise:** σ = 0.45 µV per sample
- **EMI spikes:** 1% probability per reading, ±8 µV burst
- **Sensor fault injection:** 0.1% probability of a hard fault (returns −999 °C, displayed as FAULT in red on the dashboard)
- **Cold junction compensation:** referenced to 25.0 °C ambient

Temperature recovery from millivolts follows the inverse Seebeck relation with cold-junction correction applied. Every reading shown on the dashboard to three decimal places in millivolts went through this chain.

### Thermal Physics Engine

Seven zones with individual heat capacitances, base drift rates, and cooling effectiveness coefficients. Zones are thermally coupled: the power module rises when the CPU is hot, both motors receive heat from the power module, both pumps receive heat from the motors, and the ambient slowly rises from pump dissipation. This is modelled as directional conductive coupling with empirically chosen transfer coefficients.

Stochastic load shocks occur at ~0.6% probability per step and inject between +1.5 and +4.0 °C instantaneously into the CPU or motor zones, triggering the spike detector.

Thermal drift per step combines base heat generation, Gaussian process noise (σ = 0.14 °C), cooling effectiveness scaled by temperature excess over ambient, and an actuator self-heating term: at full speed, a motor contributes roughly +0.04 °C/step back into its own zone.

### Neural Policy Network

```
Input  [14]  = [ T₁…T₇ / 100 ]  ⊕  [ S₁…S₇ / 100 ]
               (normalised temps)     (stability scores)

Encoder: Linear(14→64) → LayerNorm → ReLU
         Linear(64→48) → ReLU
         Linear(48→32) → ReLU

Output [7]   = Linear(32→7) → Sigmoid   ∈ [0, 1] per zone
```

The network is trained online one gradient step per simulation step using a composite reward that adapts its penalty weights to thermal urgency:

```
R = −λ_temp × T_max/100  −  λ_cool × mean(cooling) × (1/urgency)  +  λ_stab × S_avg
```

When the system is cool, λ_cool is large (1.5) and λ_temp is small (0.5) — the network is penalised heavily for running pumps unnecessarily. When a zone approaches critical, λ_cool drops to 0.2 and λ_temp rises to 3.0; aggressive cooling becomes acceptable and the policy shifts accordingly, without any hard-coded if-else.

The final cooling command per zone is a 60/40 blend of the neural output and a rule-based floor proportional to temperature excess. This hybrid architecture ensures graceful degradation: even at network initialisation (before any meaningful training), the rule floor provides physically sensible minimum cooling.

### Pump & Motor Speed Control

Each actuator has a maximum rated RPM (motors: 3,000 RPM, pumps: 1,800 RPM) and rated power (motors: 200 W, pumps: 150 W). Speed is not set instantaneously. A slew-rate limiter caps change at ±150 RPM per step, protecting motors from current inrush on rapid acceleration — a standard requirement in industrial drives.

Target RPM is computed from a weighted combination of the neural cooling command and a thermal urgency term, then scaled by a mode factor that varies from 0.22 in Efficiency Mode to 1.0 in Emergency Cooling.

Current draw is modelled via the pump affinity law: I ∝ RPM^1.5. Instantaneous power follows P ∝ RPM³, with an 8% standby load. Energy savings are accumulated continuously against a 700 W always-full-speed baseline.

The state machine for each actuator: IDLE → RUNNING → BOOST → EMERGENCY, with DISABLED available through the automation flag.

### Stability Analyser

A rolling 20-sample window per zone computes standard deviation and net trend. Classification thresholds:

| State | σ condition | Trend condition |
|---|---|---|
| STABLE | σ < 1.8 °C | \|Δ\| < 1.8 °C |
| TRANSIENT | σ < 3.5 °C | \|Δ\| < 4.0 °C |
| UNSTABLE | above | above |

Stability scores (0–100) are derived from σ with soft clipping at state boundaries. An UNSTABLE transition fires a dashboard alert with a 15-step cooldown to avoid noise.

### Time-to-Critical Predictor

The last 30 temperature samples are smoothed with a 5-point moving average, then a linear least-squares model is fitted to the smoothed window. If slope > 0.005 °C/step, the extrapolated time to CRITICAL_TEMP (75 °C) is computed and displayed in the Time-to-Critical panel. Zones within 35 steps of critical trigger predictive pre-cooling — a 20% boost on top of whatever the neural policy commanded.

### Operating Modes

The system transitions between six modes using hysteresis logic on the hottest zone temperature and maximum five-step rise rate. Transitions are guarded so emergency modes cannot be accidentally exited by a normal mode switch while the triggering condition still holds.

| Mode | Trigger | Duty Floor | Motor/Pump Target |
|---|---|---|---|
| 🌿 Efficiency | T_max < 33 °C, rise < 0.20 °C/5s | 20% | 22% of max RPM |
| ⚖ Balanced | 33 – 44 °C nominal range | 30% | 70% of max RPM |
| 🔥 Performance | T_max ≥ 44 °C or rise ≥ 0.90 °C/5s | 92% | 100% of max RPM |
| ⚡ Spike Suppression | Δ > 1.5 °C/step detected | +35% boost | 100% of max RPM |
| 🔮 Predictive Pre-Cool | TTC < 35 steps | +20% boost | 85% of max RPM |
| 🚨 Emergency Cooling | T_max > 75 °C | 100% | Full RPM override |

---

## Dashboard Screenshots

All four screenshots below are live captures from a running simulation, not mock-ups.

### ⚖ Balanced Control Mode

All zones stable, CPU at 39.0 °C, cooling commands at 35% duty. Total system draw: 58 W — 83.9% AI confidence.

![Balanced Mode](Balanced_Mode.png)

### 🌿 Efficiency Mode

Minimal load. CPU cools to 31.3 °C. Cooling commands drop to 11% duty. Motors spin at 240 RPM (8% of max), pumps at 144 RPM. System draw: 56 W — the affinity law keeps power near floor. AI confidence: 80.2%.

![Efficiency Mode](Efficiency_Mode.png)

### 🚨 Emergency Cooling Mode

CPU thermal shock to 76.1 °C triggers full emergency. All cooling commands pin at 100%. Motors ramp to 2,681 RPM drawing 1.689 A. System power jumps to 436 W. CPU stability reads TRANSIENT (score 50) — the system is working to recover. Time-to-critical is visible for PWR MOD (18s), MTR-A (1,045s), MTR-B (1,085s). AI confidence drops to 50.2%, which is the honest answer when a zone is breaching limits.

![Emergency Cooling Mode](Emergency_Cooling_Mode.png)

### 🔥 Performance Cooling Mode

Moderate thermal event resolved. CPU stabilised at 48.9 °C, all zones STABLE. Cooling at 92% duty. Motors at 1,858 RPM, pumps at 911 RPM. System draw: 186 W. Energy savings cumulative: 10.65 Wh. AI confidence: 88.3%.

![Performance Cooling Mode](Performace_Cooling_Mode.png)

---

## Project Structure

```
thermis_x.py          ─ Complete system: 1,250 lines, single-file architecture
├── Zone Configuration
├── Physical & Electrical Constants   (Type-K, motor ratings, thermal properties)
├── Shared State                      (thread-safe; CPython GIL on primitive types)
├── Type-K Thermocouple Engine
├── Pump & Motor Automated Controller
├── Thermal Physics Engine            (drift, coupling, stochastic shock)
├── Thermal Stability Analyser
├── Time-to-Critical Predictor
├── Power Optimizer                   (affinity law, energy savings accumulation)
├── Hybrid Adaptive Neural Controller (ThermalPolicyNet + online training)
├── Safety, Spike Detection & Predictive Pre-Cooling
├── Master Controller Pipeline
├── Simulation Loop                   (background thread, 1 Hz)
├── System-Level Metrics
└── SCADA Dashboard                   (Flask + Plotly, served via pywebview)
```

The single-file design is intentional: the entire system, from sensor physics to pixel rendering, can be read, audited, and understood as one coherent document.

---

## Getting Started

```bash
pip install numpy torch flask pywebview plotly
python thermis_x.py
```

The system boots, runs a 20-step warm-up simulation to pre-populate all history buffers, then opens the dashboard in a native desktop window at 1920×1080. No browser required.

The Flask server binds to 127.0.0.1:5000 and serves only on localhost. The `/data` endpoint returns complete system telemetry as JSON and can be consumed by any external monitoring tool.

**Requirements:** Python 3.10+, PyTorch (CPU-only is fine — the policy network is tiny), Flask, pywebview, NumPy.

---

## What This Demonstrates

From an engineering perspective, THERMIS-X is a vertical cross-section of industrial control systems: it starts at the sensor physics layer, moves through signal conditioning and state estimation, makes decisions through a combination of learned policy and deterministic safety logic, drives actuators with physical constraints respected, and presents the result in a live operator interface.

Every layer was written from first principles rather than assembled from high-level abstractions. The thermocouple model is not a temperature generator, it is a voltage generator that then gets converted back to temperature, the same way a real acquisition card works. The neural controller is not a black box, its architecture, input features, reward function, and the hybrid blending with rule-based floors are all explicit and readable.

The system handles failure gracefully: sensor faults display as FAULT rather than corrupting the control loop, emergency cooling overrides the neural policy completely, and the predictive layer gives the operator advance warning in a format that matters — seconds, not an abstract score.

---

## Technical Highlights at a Glance

- **41.276 µV/°C** Type-K Seebeck coefficient, full ADC noise chain modelled
- **14-dimensional** neural policy input: 7 normalised temperatures + 7 stability scores
- **60/40 hybrid blend** — neural inference + physics-grounded rule floor
- **150 RPM/step** slew-rate limit on all motor and pump speed transitions
- **P ∝ RPM³** pump affinity law for power computation; **I ∝ RPM^1.5** for current
- **Six operating modes** with hysteresis-based transitions, none overrideable during emergency
- **35-step look-ahead** predictive pre-cooling via linear extrapolation
- **0.1% fault injection** rate on thermocouple acquisition
- **58 W nominal / 436 W peak** — ~87% power reduction in normal operation vs. always-on baseline
- **Real-time SCADA** at 1 Hz with Plotly trend chart, 120-second rolling history per zone

---

## Author

**Vaibhav Krishna V**  
Electronics and Communication Engineer 
[GitHub](https://github.com/vaibhavkrishnav) · [LinkedIn](https://linkedin.com/in/vaibhavkrishnav)

> *Built to demonstrate that rigorous sensor engineering, adaptive control theory, and production-quality software are not separate disciplines — they are one.*
