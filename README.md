# 🌡️ AI-Based Smart Thermal Fan Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*Intelligent thermal regulation combining classical feedback control with adaptive AI cooling logic.*

</div>

---

## 📌 Project Overview

A real-time thermal regulation platform that dynamically adjusts cooling behavior based on temperature conditions and system load. The system models realistic thermal dynamics and evaluates intelligent cooling strategies that balance **temperature stability** and **energy efficiency**.

---

## 🏗️ System Architecture

```
Thermal Inputs (CPU Load + Power + Ambient)
        ↓
  Thermal Simulation Engine    ← Dynamic temperature model
        ↓
  Hybrid Controller            ← Feedback + adaptive logic
        ↓
  Stability Constraints        ← Anti-windup + damping
        ↓
  Adaptive Cooling Response    ← Load-aware fan modulation
        ↓
  Real-Time Monitoring         ← Live visualization dashboard
```

---

## 📊 Key Results

| Metric | Value |
|---|---|
| CPU Temperature | 49.6°C |
| Power Temperature | 37.0°C |
| Ambient Temperature | 20.0°C |
| Cooling Intensity | 49% |
| **Energy Efficiency** | **65%** |
| **Thermal Safety Score** | **68%** |
| Control Mode | Balanced Cooling |

---

## ✅ Key Features

- 🔴 **Multi-source thermal inputs** — CPU load, power dissipation, ambient temperature
- 🧠 **Hybrid control** — PID-style feedback + adaptive response
- ⚡ **Anti-windup constraints** — prevents overshoot during thermal spikes
- 📉 **Adaptive cooling** — reduces fan activity under low thermal stress
- 📊 **Real-time Matplotlib dashboard** — live temperature and cooling curves

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `NumPy` | Thermal dynamics numerical modeling |
| `SciPy` | Control mathematics, stability computations |
| `Matplotlib` | Real-time temperature trend visualization |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-thermal-control.git
cd smart-thermal-control
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simulation
```bash
python thermal_control.py
```

---

## 📁 Project Structure

```
smart-thermal-control/
├── thermal_control.py      # Main simulation — run this
├── thermal_model.py        # Dynamic thermal simulation engine
├── hybrid_controller.py    # PID + adaptive cooling controller
├── stability.py            # Anti-windup and damping mechanisms
├── visualizer.py           # Real-time Matplotlib dashboard
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏭 Applications

**Industrial:**
- Thermal regulation in embedded electronics and industrial computers
- Cooling optimization in edge computing devices
- Smart fan control for power electronics and drives
- Industrial thermal monitoring platforms
- Adaptive cooling logic for intelligent HVAC systems

**Societal:**
- Prevention of overheating in electronic devices
- Energy efficient cooling in consumer electronics
- Reduced energy waste through adaptive cooling
- Extended device lifespan through stable temperature regulation

---

## 👤 Author

**Vaibhav Krishna V**  
Electronics & Communication Engineering, NMIT Bengaluru  
USN: 1NT22EC182  
📧 vaibhavkv078@gmail.com

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
