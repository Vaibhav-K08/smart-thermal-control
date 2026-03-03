"""
AI-Based Smart Thermal Fan Control System
Author: Vaibhav Krishna V | NMIT Bengaluru | 1NT22EC182
"""

import numpy as np
import time

# ── Configuration ─────────────────────────────────────
TARGET_TEMP      = 45.0    # Desired CPU temperature (C)
AMBIENT_TEMP     = 20.0    # Ambient temperature (C)
MAX_COOLING      = 100.0   # Max fan speed (%)
MIN_COOLING      = 10.0    # Min fan speed (%)
TACH_BPM         = 75.0    # Tachycardia threshold (reused for over-temp)
SIMULATION_TIME  = 100     # Number of simulation steps
DT               = 1.0     # Time step (seconds)

# PID Gains
KP = 2.0
KI = 0.05
KD = 0.5

# Thermal model constants
THERMAL_RESISTANCE = 0.3
THERMAL_CAPACITANCE = 10.0

# ── Thermal Simulation Model ───────────────────────────

class ThermalSystem:
    def __init__(self):
        self.cpu_temp    = 35.0
        self.power_temp  = 30.0
        self.ambient     = AMBIENT_TEMP
        self.workload    = 0.5   # 0 to 1

    def update(self, cooling_pct, dt=DT):
        # Random workload variation
        self.workload = np.clip(
            self.workload + np.random.uniform(-0.05, 0.08), 0.1, 1.0
        )

        # Power dissipation based on workload
        power_dissipation = self.workload * 80.0  # Watts

        # Cooling effect
        cooling_effect = (cooling_pct / 100.0) * 60.0

        # Thermal dynamics (simplified RC model)
        d_cpu = (power_dissipation - cooling_effect -
                 (self.cpu_temp - self.ambient) / THERMAL_RESISTANCE) * dt / THERMAL_CAPACITANCE
        self.cpu_temp   = np.clip(self.cpu_temp + d_cpu, AMBIENT_TEMP, 100.0)

        d_power = (power_dissipation * 0.4 -
                   (self.power_temp - self.ambient) * 0.5) * dt / THERMAL_CAPACITANCE
        self.power_temp = np.clip(self.power_temp + d_power, AMBIENT_TEMP, 90.0)

        return self.cpu_temp, self.power_temp

# ── Hybrid Controller ─────────────────────────────────

class HybridThermalController:
    def __init__(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.adaptive_gain = 1.0

    def compute(self, current_temp, dt=DT):
        error = current_temp - TARGET_TEMP

        # PID components
        self.integral   = np.clip(self.integral + error * dt, -50, 50)  # Anti-windup
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error

        pid_output = KP * error + KI * self.integral + KD * derivative

        # Adaptive gain — increase responsiveness under high thermal stress
        if current_temp > TARGET_TEMP + 10:
            self.adaptive_gain = min(self.adaptive_gain + 0.05, 2.0)
        elif current_temp < TARGET_TEMP:
            self.adaptive_gain = max(self.adaptive_gain - 0.02, 0.5)

        cooling = MIN_COOLING + np.clip(pid_output * self.adaptive_gain, 0, MAX_COOLING - MIN_COOLING)
        return np.clip(cooling, MIN_COOLING, MAX_COOLING)

    def get_metrics(self, cpu_temp, cooling_pct):
        energy_efficiency = max(0, 100 - abs(cpu_temp - TARGET_TEMP) * 2 - cooling_pct * 0.3)
        thermal_safety    = max(0, 100 - max(0, cpu_temp - TARGET_TEMP) * 3)
        return round(energy_efficiency, 1), round(thermal_safety, 1)

# ── Main Simulation ───────────────────────────────────

def run_thermal_simulation():
    system     = ThermalSystem()
    controller = HybridThermalController()

    print("=" * 60)
    print("  Priority-Aware Industrial Thermal AI")
    print("  Author: Vaibhav Krishna V | NMIT | 1NT22EC182")
    print("=" * 60)
    print(f"  Target CPU Temp : {TARGET_TEMP}°C")
    print(f"  Ambient Temp    : {AMBIENT_TEMP}°C")
    print(f"  Simulation Steps: {SIMULATION_TIME}")
    print("=" * 60)
    print(f"  {'Step':>4} | {'CPU(°C)':>8} | {'Pwr(°C)':>8} | "
          f"{'Cool(%)':>8} | {'Efficiency':>11} | {'Safety':>8} | Mode")
    print(f"  {'-'*70}")

    for step in range(1, SIMULATION_TIME + 1):
        cpu_temp, pwr_temp = system.update(
            controller.compute(system.cpu_temp)
        )
        cooling = controller.compute(cpu_temp)
        eff, safety = controller.get_metrics(cpu_temp, cooling)

        if cpu_temp > TARGET_TEMP + 15:
            mode = "Emergency Cooling"
        elif cpu_temp > TARGET_TEMP + 5:
            mode = "Aggressive Cooling"
        elif cpu_temp < TARGET_TEMP - 5:
            mode = "Eco Mode"
        else:
            mode = "Balanced Cooling"

        print(f"  {step:>4} | {cpu_temp:>7.1f}C | {pwr_temp:>7.1f}C | "
              f"{cooling:>7.1f}% | {eff:>10.1f}% | {safety:>7.1f}% | {mode}")

        time.sleep(0.03)

    print("\n  Simulation complete.")
    print(f"  Final State — CPU: {cpu_temp:.1f}°C | "
          f"Cooling: {cooling:.0f}% | Efficiency: {eff:.1f}% | Safety: {safety:.1f}%")

if __name__ == "__main__":
    run_thermal_simulation()
