"""
generate_data.py
Generates synthetic pipe network dataset matching the statistical
properties described in the project (39,637 pipes, 8.63% failure rate).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N = 39_637
FAILURE_RATE = 0.0863

Path("data").mkdir(exist_ok=True)

materials = ["CI", "DI", "AC", "ST", "PE", "PVC", "GRP", "PP", "Pb"]
mat_weights = [0.30, 0.22, 0.15, 0.10, 0.10, 0.07, 0.03, 0.02, 0.01]

pipe_types = ["HC", "DP", "HY"]
type_weights = [0.55, 0.38, 0.07]

material = np.random.choice(materials, size=N, p=mat_weights)
pipe_type = np.random.choice(pipe_types, size=N, p=type_weights)

# Realistic install year: large build-out in 1950s-80s, tapering off earlier/later
# Mix of old legacy pipes + mid-century bulk + some modern replacements
install_year = np.concatenate([
    np.random.normal(1955, 12, size=int(N * 0.28)).astype(int),  # post-war build
    np.random.normal(1972, 10, size=int(N * 0.32)).astype(int),  # 70s expansion
    np.random.normal(1988, 8,  size=int(N * 0.22)).astype(int),  # 80s/90s
    np.random.normal(2005, 6,  size=int(N * 0.12)).astype(int),  # modern
    np.random.normal(1930, 15, size=N - int(N*0.28) - int(N*0.32) - int(N*0.22) - int(N*0.12)).astype(int),  # legacy
])
np.random.shuffle(install_year)
install_year = np.clip(install_year, 1900, 2019)
current_year = 2020
age = current_year - install_year

diameter_map = {"HC": (80, 20), "DP": (150, 50), "HY": (200, 60)}
diameter = np.array([
    max(25, int(np.random.normal(*diameter_map[t])))
    for t in pipe_type
])

length = np.random.lognormal(mean=3.5, sigma=0.8, size=N)
length = np.clip(length, 1, 2000)

pressure = np.random.normal(4.5, 1.2, size=N)
pressure = np.clip(pressure, 1.0, 10.0)

HC_Str = np.random.poisson(8, size=N)
HY_Str = np.random.poisson(2, size=N)
Valves_Tot = np.random.poisson(3, size=N)
Valves_St = np.random.poisson(1, size=N)

# failure history correlates with age, material, and pressure
base_fail_prob = (
    0.003 * age / 50
    + 0.002 * (material == "CI").astype(float)
    + 0.001 * (material == "AC").astype(float)
    + 0.001 * (pressure > 6).astype(float)
)
base_fail_prob = np.clip(base_fail_prob, 0.001, 0.3)
Failure_Tot = np.random.poisson(base_fail_prob * 3, size=N)
Failure_New = np.minimum(Failure_Tot, np.random.poisson(base_fail_prob, size=N))

# target: failure label
failure_prob = (
    0.04 * (age / 100)
    + 0.03 * (material == "CI").astype(float)
    + 0.02 * (material == "AC").astype(float)
    + 0.02 * (material == "Pb").astype(float)
    + 0.01 * (pressure > 7).astype(float)
    + 0.02 * (Failure_Tot > 2).astype(float)
    + 0.01 * (diameter < 80).astype(float)
)
failure_prob = np.clip(failure_prob, 0.01, 0.5)
# scale so overall failure rate matches target
scale = FAILURE_RATE / failure_prob.mean()
failure_prob = np.clip(failure_prob * scale, 0, 1)
failure = (np.random.uniform(size=N) < failure_prob).astype(int)

df = pd.DataFrame({
    "age":          age,
    "diameter":     diameter,
    "length":       np.round(length, 2),
    "pressure":     np.round(pressure, 2),
    "HC_Str":       HC_Str,
    "HY_Str":       HY_Str,
    "Valves_Tot":   Valves_Tot,
    "Valves_St":    Valves_St,
    "Failure_Tot":  Failure_Tot,
    "Failure_New":  Failure_New,
    "material":     material,
    "install_year": install_year,
    "type":         pipe_type,
    "failure":      failure,
})

df.to_csv("data/pipe_network.csv", index=False)
print(f"Dataset saved: {len(df):,} pipes, failure rate: {df['failure'].mean():.2%}")
