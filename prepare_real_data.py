"""
prepare_real_data.py
--------------------
Builds pipe_network.csv from the City of Kitchener open water main data.

Sources (open data — City of Kitchener GIS portal):
  Water_Mains.csv        - all 15,903 pipes in the network
  Water_Main_Breaks.csv  - 2,766 recorded break incidents (1985-2023)
  cleaned_break_data.csv - breaks joined to pipe properties + GPS coords

Join: GlobalID (UUID) links mains to their break history.
Label: failure = 1 if the pipe has had at least one recorded break.
"""

import numpy as np
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

mains   = pd.read_csv("data/raw/Water_Mains.csv")
cleaned = pd.read_csv("data/raw/cleaned_break_data.csv")

# ── Break count per pipe ──────────────────────────────────────────────────────
break_counts = cleaned.groupby("GlobalID").size().reset_index(name="Failure_Tot")

df = mains.merge(break_counts, on="GlobalID", how="left")
df["Failure_Tot"] = df["Failure_Tot"].fillna(0).astype(int)
df["failure"]     = (df["Failure_Tot"] > 0).astype(int)

# ── Age ───────────────────────────────────────────────────────────────────────
df["install_year"] = pd.to_datetime(
    df["INSTALLATION_DATE"], errors="coerce"
).dt.year
df["age"] = 2024 - df["install_year"]

# Drop pipes with no install date (~135 pipes)
df = df.dropna(subset=["age"])
df["age"] = df["age"].astype(int)

# ── Material — simplify rare categories ──────────────────────────────────────
mat_map = {
    "PVC": "PVC", "PVCO": "PVC", "PVCB": "PVC", "PVCF": "PVC",
    "DI":  "DI",
    "CI":  "CI",
    "AC":  "AC",
    "HDPE": "HDPE", "HDPE IN CI": "HDPE",
    "CPP": "CPP",
    "COP": "Other", "ST": "Other",
}
df["material"] = df["MATERIAL"].map(mat_map).fillna("Other")

# ── Pressure zone — numeric encoding ─────────────────────────────────────────
zone_map = {z: i for i, z in enumerate(df["PRESSURE_ZONE"].unique())}
df["pressure_zone"] = df["PRESSURE_ZONE"].map(zone_map)

# ── Rename to clean column names ─────────────────────────────────────────────
df = df.rename(columns={
    "PIPE_SIZE":       "diameter",
    "Shape__Length":   "length",
    "CONDITION_SCORE": "condition_score",
    "CRITICALITY":     "criticality",
})

# ── GPS coordinates (available for pipes in breaks dataset) ──────────────────
coords = cleaned.groupby("GlobalID")[["longitude", "latitude"]].first().reset_index()
df = df.merge(coords, on="GlobalID", how="left")

# ── Final columns ─────────────────────────────────────────────────────────────
keep = [
    "age", "diameter", "length", "material", "install_year",
    "condition_score", "criticality", "pressure_zone",
    "Failure_Tot", "failure", "longitude", "latitude",
]
df = df[keep]

df.to_csv("data/pipe_network.csv", index=False)
print(f"Dataset saved:  {len(df):,} pipes")
print(f"Failure rate:   {df['failure'].mean():.2%}  ({df['failure'].sum():,} failures)")
print(f"With GPS coords:{df['longitude'].notna().sum():,} pipes")
