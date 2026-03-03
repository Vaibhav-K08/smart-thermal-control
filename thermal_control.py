"""
AI Based Smart Thermal Fan Control System
Author: Vaibhav Krishna V 
"""

import time, random, threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify
import plotly.graph_objects as go

print("🏭 Priority Aware Industrial Thermal AI Starting...")

# ==========================================================
# CONFIG
# ==========================================================
ZONES = ["CPU", "Power", "Ambient"]
temps = {"CPU": 36.0, "Power": 33.0, "Ambient": 28.0}

history = {z: [] for z in ZONES}
cooling = 0.3
cooling_history = []

mode = "Balanced Control"
last_temps = temps.copy()

CRITICAL_TEMP = 70
SPIKE_RATE = 1.2

# ==========================================================
# AI POLICIES
# ==========================================================
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

policies = {z: PolicyNet() for z in ZONES}
opts = {z: optim.Adam(policies[z].parameters(), lr=0.001) for z in ZONES}

def state_vec():
    return torch.tensor([
        temps["CPU"]/100,
        temps["Power"]/100,
        temps["Ambient"]/100
    ], dtype=torch.float32)

# ==========================================================
# PHYSICS
# ==========================================================
def shock():
    if random.random() < 0.05:
        temps["CPU"] += random.uniform(4, 6)

def coupling():
    temps["Power"] += (temps["CPU"] - 42) * 0.03
    temps["Ambient"] += (temps["Power"] - 38) * 0.02

def drift(zone):
    base = {"CPU":0.35,"Power":0.25,"Ambient":0.1}[zone]
    noise = random.uniform(-0.2,0.25)
    cool = cooling * {"CPU":1.8,"Power":1.2,"Ambient":0.6}[zone]
    return base + noise - cool

# ==========================================================
# SAFETY LAYER
# ==========================================================
def safety_override():
    global cooling, mode
    if max(temps.values()) > CRITICAL_TEMP:
        cooling = min(1.0, cooling + 0.15)
        mode = "Emergency Cooling"
        return True
    return False

# ==========================================================
# SPIKE DETECTOR
# ==========================================================
def spike_detector():
    global cooling, mode
    for z in ZONES:
        if temps[z] - last_temps[z] > SPIKE_RATE:
            cooling = min(1.0, cooling + 0.1)
            mode = "Spike Suppression"
            return True
    return False

# ==========================================================
# 🧠 THERMAL URGENCY CURVE (NEW CORE LOGIC)
# ==========================================================
def urgency_curve():
    global cooling, mode

    max_temp = max(temps.values())

    if max_temp < 40:
        return  # efficiency zone

    elif max_temp < 50:
        cooling += 0.01
        mode = "Efficiency → Balanced"

    elif max_temp < 60:
        cooling += 0.03
        mode = "Balanced Cooling"

    else:
        cooling += 0.06
        mode = "Performance Cooling"

    cooling = min(1.0, cooling)

# ==========================================================
# AI OPTIMIZER (ADAPTIVE PRIORITY)
# ==========================================================
def ai_control():
    global cooling

    s = state_vec()
    outputs = []

    for z in ZONES:
        with torch.no_grad():
            outputs.append(policies[z](s).item())

    fusion = sum(outputs) / len(outputs)
    cooling += fusion * 0.02
    cooling = max(0.05, min(1.0, cooling))

def train_step():
    s = state_vec()
    max_temp = max(temps.values())

    # Dynamic priority weighting
    temp_weight = np.interp(max_temp, [30, 70], [0.6, 2.0])
    cool_penalty = np.interp(max_temp, [30, 70], [2.0, 0.8])

    for z in ZONES:
        opts[z].zero_grad()
        out = policies[z](s)

        reward = -max_temp * 0.01 * temp_weight - cooling * cool_penalty
        loss = -reward * out
        loss.backward()
        opts[z].step()

# ==========================================================
# ANTI-WINDUP RELAXATION
# ==========================================================
def anti_windup():
    global cooling

    avg_temp = sum(temps.values()) / 3

    if avg_temp < 35:
        cooling *= 0.96
    elif avg_temp < 45:
        cooling *= 0.98

    cooling *= 0.99
    cooling = max(0.05, cooling)

# ==========================================================
# CONTROLLER PIPELINE
# ==========================================================
def hybrid_controller():
    global last_temps

    if safety_override():
        pass
    elif spike_detector():
        pass
    else:
        urgency_curve()
        ai_control()

    anti_windup()
    last_temps = temps.copy()

# ==========================================================
# SIM LOOP
# ==========================================================
def sim_loop():
    while True:
        shock()
        coupling()
        hybrid_controller()

        for z in ZONES:
            temps[z] += drift(z)
            temps[z] = max(20, min(95, temps[z]))
            history[z].append(temps[z])

        cooling_history.append(cooling * 100)
        train_step()
        time.sleep(1)

# ==========================================================
# METRICS (DUAL EFFICIENCY)
# ==========================================================
def energy_efficiency():
    return int(100 - cooling * 70)

def thermal_safety():
    max_temp = max(temps.values())
    return int(np.interp(max_temp, [30, 80], [100, 20]))

# ==========================================================
# VISUALIZATION
# ==========================================================
def build_fig():
    fig = go.Figure()

    colors = {"CPU":"#ff453a","Power":"#ff9f0a","Ambient":"#0a84ff"}

    fig.add_hrect(y0=20, y1=40, fillcolor="rgba(0,255,120,0.05)", line_width=0)
    fig.add_hrect(y0=40, y1=60, fillcolor="rgba(255,200,0,0.05)", line_width=0)
    fig.add_hrect(y0=60, y1=90, fillcolor="rgba(255,60,60,0.06)", line_width=0)

    for z in ZONES:
        fig.add_trace(go.Scatter(
            y=history[z][-120:], mode="lines",
            name=f"{z} Temp",
            line=dict(color=colors[z], width=3)
        ))

    fig.add_trace(go.Scatter(
        y=cooling_history[-120:], mode="lines",
        name="Cooling (%)",
        line=dict(color="#64d2ff", dash="dot", width=3),
        yaxis="y2"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        margin=dict(l=70, r=90, t=80, b=60),
        yaxis=dict(title="Temperature (°C)", range=[20,90]),
        yaxis2=dict(title="Cooling (%)", overlaying="y", side="right", range=[0,100]),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center")
    )

    return fig.to_json()

# ==========================================================
# WEB UI
# ==========================================================
app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
    <head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
    body { background:#020617; color:white; font-family:Segoe UI; }
    .wrap { display:flex; gap:24px; padding:20px; }
    .left { flex:3; }
    .right { flex:1; }
    .card { background:#111827; padding:18px; border-radius:14px; }
    </style>
    </head>
    <body>
    <h2 style="text-align:center">🏭 Priority-Aware Industrial Thermal AI</h2>
    <div class="wrap">
      <div class="left"><div id="chart" style="height:560px;"></div></div>
      <div class="right"><div class="card" id="live"></div></div>
    </div>

    <script>
    async function update(){
      let r = await fetch('/data');
      let d = await r.json();
      Plotly.react("chart", JSON.parse(d.fig).data, JSON.parse(d.fig).layout);

      live.innerHTML = `
        CPU: ${d.cpu.toFixed(1)}°C<br>
        Power: ${d.power.toFixed(1)}°C<br>
        Ambient: ${d.ambient.toFixed(1)}°C<br><br>
        Cooling: ${(d.cool*100).toFixed(0)}%<br>
        Energy Efficiency: ${d.energy}%<br>
        Thermal Safety: ${d.safety}%<br><br>
        Mode: ${d.mode}`;
    }
    setInterval(update,1000);
    update();
    </script>
    </body></html>
    """

@app.route("/data")
def data():
    return jsonify({
        "fig": build_fig(),
        "cpu": temps["CPU"],
        "power": temps["Power"],
        "ambient": temps["Ambient"],
        "cool": cooling,
        "energy": energy_efficiency(),
        "safety": thermal_safety(),
        "mode": mode
    })

threading.Thread(target=sim_loop, daemon=True).start()
print("🌐 http://localhost:5000")
app.run(port=5000)
    run_thermal_simulation()
