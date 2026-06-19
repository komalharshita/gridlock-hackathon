"""
prepare_dataset.py
-------------------
Converts the raw Astram event export into a clean dataset your
Digital Twin model can train on.

Run:  python3 prepare_dataset.py
Input:  data/Astram_event_data_anonymized.csv
Output: data/processed_events.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/Astram_event_data_anonymized.csv"
OUT_PATH = "data/processed_events.csv"

KEEP_COLS = [
    "id", "event_type", "event_cause", "latitude", "longitude",
    "address", "requires_road_closure", "start_datetime",
    "resolved_datetime", "status", "priority", "zone", "junction",
]

df = pd.read_csv(RAW_PATH, usecols=KEEP_COLS)
print(f"Loaded {len(df)} rows")

df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], utc=True, errors="coerce")

df["resolution_minutes"] = (
    (df["resolved_datetime"] - df["start_datetime"]).dt.total_seconds() / 60
)
df.loc[df["resolution_minutes"] < 0, "resolution_minutes"] = np.nan
df.loc[df["resolution_minutes"] > 24 * 60, "resolution_minutes"] = np.nan

df["hour"] = df["start_datetime"].dt.hour
df["day_of_week"] = df["start_datetime"].dt.day_name()
df["is_weekend"] = df["start_datetime"].dt.dayofweek >= 5

CROWD_MAP = {
    "public_event":      "High",
    "procession":        "High",
    "vip_movement":      "High",
    "protest":           "Medium",
    "congestion":        "Medium",
    "construction":      "Medium",
    "road_conditions":   "Low",
    "water_logging":     "Low",
    "tree_fall":         "Low",
    "pot_holes":         "Low",
    "accident":          "Low",
    "vehicle_breakdown": "Low",
    "Debris":            "Low",
    "others":            "Low",
    "test_demo":         "Low",
}
df["crowd_proxy"] = df["event_cause"].map(CROWD_MAP).fillna("Low")

CROWD_NUMERIC = {"Low": 500, "Medium": 5000, "High": 20000}
df["crowd_estimate"] = df["crowd_proxy"].map(CROWD_NUMERIC)

def derive_risk(row):
    closure = row["requires_road_closure"]
    priority = row["priority"]
    if pd.isna(priority):
        priority = "Low"
    if closure and priority == "High":
        return "Severe"
    if closure and priority == "Low":
        return "Moderate"
    if not closure and priority == "High":
        return "Moderate"
    return "Minor"

df["risk_level"] = df.apply(derive_risk, axis=1)

df["zone"] = df["zone"].fillna("Unknown")
df["junction"] = df["junction"].fillna("Unknown")

final_cols = [
    "id", "event_type", "event_cause", "crowd_proxy", "crowd_estimate",
    "latitude", "longitude", "address", "zone", "junction",
    "requires_road_closure", "priority", "risk_level",
    "start_datetime", "hour", "day_of_week", "is_weekend",
    "resolution_minutes", "status",
]
df_final = df[final_cols]
df_final.to_csv(OUT_PATH, index=False)

print(f"\nSaved cleaned dataset -> {OUT_PATH}")
print(f"Rows: {len(df_final)}")
print("\nRisk level distribution:")
print(df_final["risk_level"].value_counts())
print("\nCrowd proxy distribution:")
print(df_final["crowd_proxy"].value_counts())
print(f"\nResolution time known for {df_final['resolution_minutes'].notna().sum()} rows")
