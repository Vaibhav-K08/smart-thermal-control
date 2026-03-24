# 🌡️ Priority Aware Industrial Thermal AI Regulation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Three zone thermal regulation using per-zone PyTorch policy networks, a nonlinear urgency curve, spike detection, and a live Flask web dashboard, all training continuously during simulation.*

</div>

---

## What This Does

This system simulates industrial thermal dynamics across three zones: CPU, Power, and Ambient and regulates temperature using a hybrid controller that combines a nonlinear urgency curve, per-zone AI policy networks, spike detection, safety overrides, and anti-windup relaxation. The policy networks train live during the simulation, adapting their behavior as thermal conditions change.

Everything is visible on a live web dashboard at `http://localhost:5000` built with Flask and Plotly.

---

## How It Works

```
Thermal Simulation (CPU + Power + Ambient zones)
        ↓
  Thermal Coupling  ←  zones influence each other physically
  Random Shocks     ←  sudden load spikes (5% probability per step)
        ↓
  Hybrid Controller Pipeline
  ├── Safety Override   → emergency cooling if any zone > 70°C
  ├── Spike Detector    → boost cooling if temp rises > 1.2°C/step
  ├── Urgency Curve     → nonlinear cooling based on temperature band
  └── AI Control        → per-zone PolicyNet fusion output
        ↓
  Anti-Windup Relaxation  ←  decays cooling when temps are low
        ↓
  Live Training           ←  policy networks update every second
        ↓
  Flask Web Dashboard     ←  real time Plotly chart + metrics panel
```

---

## The AI Layer

Each thermal zone has its own **PolicyNet** — a small PyTorch neural network that takes the full system state and outputs a cooling adjustment.

```
Input:  [CPU_temp/100, Power_temp/100, Ambient_temp/100]
Hidden: Linear(3→32) → ReLU → Linear(32→1) → Tanh
Output: cooling delta (−1 to +1)
```

All three outputs are averaged and applied as a cooling adjustment. The networks train continuously using a reward function that dynamically shifts priority based on current temperature:

```python
temp_weight  = interpolate(max_temp, [30, 70], [0.6, 2.0])  # hotter = penalize temp more
cool_penalty = interpolate(max_temp, [30, 70], [2.0, 0.8])  # hotter = penalize cooling less
reward = -max_temp * 0.01 * temp_weight - cooling * cool_penalty
```

---

## Thermal Urgency Curve

Instead of a fixed PID response, the system uses a nonlinear urgency curve that maps temperature bands to cooling aggressiveness:

| Temperature Band | Cooling Increase | Mode |
|---|---|---|
| Below 40°C | None | Efficiency Zone |
| 40 — 50°C | +0.01 per step | Efficiency → Balanced |
| 50 — 60°C | +0.03 per step | Balanced Cooling |
| Above 60°C | +0.06 per step | Performance Cooling |
| Above 70°C | +0.15 (override) | Emergency Cooling |

---

## Safety and Stability

**Safety Override** — if any zone exceeds 70°C, cooling jumps by 0.15 immediately regardless of the AI output.

**Spike Detector** — if any zone rises faster than 1.2°C in a single step, cooling boosts by 0.1 and the mode switches to Spike Suppression.

**Anti-Windup Relaxation** — prevents the controller from holding unnecessary cooling after temperatures drop. Cooling decays at 0.96× per step below 35°C and 0.98× below 45°C, with a continuous 0.99× passive decay.

**Thermal Coupling** — zones are physically linked. Power temperature drifts toward CPU temperature, and Ambient drifts toward Power. This forces the controller to manage the system as a whole, not zone by zone.

---

## Dashboard

Live web dashboard served at `http://localhost:5000`:

- Temperature curves for all three zones with color coded bands (green / yellow / red)
- Cooling percentage overlaid as a dotted line on a secondary axis
- Sidebar panel showing live values — CPU temp, Power temp, Ambient temp, Cooling %, Energy Efficiency, Thermal Safety, current Mode
- Updates every second

---

## Features

- Per-zone PyTorch policy networks training live during simulation
- Nonlinear urgency curve — not a fixed PID response
- Thermal coupling between zones — physically realistic model
- Spike detector catches sudden load events before they escalate
- Anti-windup relaxation prevents overcooling during low load
- Dynamic reward weighting — AI priorities shift with temperature
- Flask + Plotly web dashboard, no desktop GUI dependency

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | Per-zone PolicyNet training and inference |
| Flask | Web server for live dashboard |
| Plotly | Interactive real time temperature chart |
| NumPy | Thermal dynamics and interpolation |
| threading | Simulation loop runs parallel to web server |

---

## Running It

```bash
git clone https://github.com/YOUR_USERNAME/smart-thermal-control.git
cd smart-thermal-control
pip install -r requirements.txt
python thermal_control.py
```

Then open `http://localhost:5000` in your browser. The simulation starts immediately and the dashboard updates every second.

---

## Project Structure

```
smart-thermal-control/
├── thermal_control.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Industrial Applications

- Thermal regulation in embedded electronics and industrial computers
- Cooling optimization in edge computing devices
- Smart fan control for power electronics and drives
- Industrial thermal monitoring platforms
- Adaptive cooling logic for intelligent HVAC systems

## Societal Applications

- Prevention of overheating in electronic devices
- Energy efficient cooling in consumer electronics
- Reduced energy waste through adaptive cooling
- Extended device lifespan through stable temperature regulation
- Smarter home appliance thermal control

---

## Author

**Vaibhav Krishna V**  
Electronics & Communication Engineer  
📧 vaibhavkv078@gmail.com

---

## License

MIT — see [LICENSE](LICENSE) for details.
