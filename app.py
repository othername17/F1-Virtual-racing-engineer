"""
Virtual Race Engineer — F1 25 / F1 26 telemetry analyzer (Streamlit)
 
Upload a tab-separated (or comma) telemetry export matching the F1 game
logger schema (266 columns). The app diagnoses balance / braking / traction
issues with frequency, then ranks setup changes with reasons.
 
Usage:
  pip install -r requirements.txt
  streamlit run app.py
"""
 
from __future__ import annotations
 
import io
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
SENTINEL = -1.0
# Wheel index convention (F1 game style): 0=FL, 1=FR, 2=RL, 3=RR
FL, FR, RL, RR = 0, 1, 2, 3
 
# Speed bands (km/h)
V_LOW = 120.0
V_HIGH = 200.0
 
# Slip-angle thresholds (radians) — export values are ~radians
US_ALPHA_THRESH = 0.03  # front more positive slip than rear → understeer
OS_ALPHA_THRESH = 0.03
# Longitudinal slip
LOCK_SLIP = 0.08  # |kappa| under brake
SPIN_SLIP = 0.12  # kappa under throttle (positive spin)
# Steering / G proxies
STEER_BUSY = 0.35  # |d(steering)/dt| proxy via sample diff
# Tire surface temp window (°C) — game scale, tune if needed
TIRE_COLD = 70.0
TIRE_HOT = 95.0
 
PHASE_ENTRY = "entry"
PHASE_MID = "mid"
PHASE_EXIT = "exit"
PHASE_STRAIGHT = "straight"
 
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
 
 
@dataclass
class IssueEvent:
    issue_id: str
    name: str
    lap: float
    distance_m: float
    speed_kph: float
    phase: str
    severity: float  # 0–1+
    detail: str
 
 
@dataclass
class IssueSummary:
    issue_id: str
    name: str
    count: int
    events_per_lap: float
    laps_present: int
    total_laps: int
    lap_presence_pct: float
    hot_corners_m: list  # distance clusters
    mean_severity: float
    max_severity: float
    sample_details: list
    confidence: float  # 0–1
 
 
@dataclass
class SetupChange:
    parameter: str
    direction: str  # "increase" | "decrease" | "adjust"
    amount_hint: str
    reason: str
    linked_issues: list
    priority: float
    validation_metric: str
 
 
# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------
 
 
def _detect_sep(sample: bytes) -> str:
    head = sample[:4000]
    if b"\t" in head:
        return "\t"
    return ","
 
 
def load_telemetry(file) -> pd.DataFrame:
    """Load telemetry from upload or path; TSV or CSV."""
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="replace")
        sep = _detect_sep(raw)
        df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
    else:
        # path
        with open(file, "rb") as f:
            sample = f.read(4000)
        sep = _detect_sep(sample)
        df = pd.read_csv(file, sep=sep, low_memory=False)
 
    df.columns = [str(c).strip() for c in df.columns]
    return clean_telemetry(df)
 
 
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")
 
 
def clean_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numerics, map sentinel -1 → NaN on channels where -1 means invalid."""
    df = df.copy()
 
    # Always numeric-coerce known columns if present
    prefer_numeric = [c for c in df.columns if c not in ("carId", "trackId")]
    for c in prefer_numeric:
        df[c] = _to_num(df[c])
 
    # Sentinel handling: -1 is invalid for most physics channels.
    # Do NOT blank legitimate small negatives (camber, g-force, slip).
    sentinel_cols = [
        "throttle",
        "brake",
        "clutch",
        "steering",
        "gear",
        "rpm",
        "rpm_perc",
        "fuel",
        "lap_number",
        "lap_distance",
        "lap_time",
        "pit_status",
        "wheel_speed_0",
        "wheel_speed_1",
        "wheel_speed_2",
        "wheel_speed_3",
        "tyre_temp_0",
        "tyre_temp_1",
        "tyre_temp_2",
        "tyre_temp_3",
        "tyre_wear_0",
        "tyre_wear_1",
        "tyre_wear_2",
        "tyre_wear_3",
        "tyre_press_0",
        "tyre_press_1",
        "tyre_press_2",
        "tyre_press_3",
        "wing_setup_0",
        "wing_setup_1",
        "drs",
        "ers_store",
        "track_temp",
        "air_temp",
        "front_brake_bias",
        "brake_bias_setup",
        "tyres_age",
        "velocity_X",
        "velocity_Y",
        "velocity_Z",
    ]
    for c in sentinel_cols:
        if c in df.columns:
            df.loc[df[c] == SENTINEL, c] = np.nan
 
    # Speed (m/s and km/h)
    vx = df.get("velocity_X", pd.Series(0.0, index=df.index)).fillna(0.0)
    vy = df.get("velocity_Y", pd.Series(0.0, index=df.index)).fillna(0.0)
    vz = df.get("velocity_Z", pd.Series(0.0, index=df.index)).fillna(0.0)
    df["speed_ms"] = np.sqrt(vx**2 + vy**2 + vz**2)
    df["speed_kph"] = df["speed_ms"] * 3.6
 
    # G forces — export uses gforce_X / gforce_Y; pick lateral as higher
    # correlation with |steering| in mid-speed corners when possible.
    gx = df.get("gforce_X", pd.Series(np.nan, index=df.index))
    gy = df.get("gforce_Y", pd.Series(np.nan, index=df.index))
    df["g_long"], df["g_lat"] = _assign_g_axes(df, gx, gy)
    df["g_lat_abs"] = df["g_lat"].abs()
    df["g_long_signed"] = df["g_long"]  # keep sign; braking often negative
 
    # Axle slip angles (mean FL/FR vs RL/RR)
    for i in range(4):
        col = f"wheel_slip_angle_{i}"
        if col not in df.columns:
            df[col] = np.nan
        col_r = f"wheel_slip_ratio_{i}"
        if col_r not in df.columns:
            df[col_r] = np.nan
 
    df["alpha_f"] = df[["wheel_slip_angle_0", "wheel_slip_angle_1"]].mean(axis=1)
    df["alpha_r"] = df[["wheel_slip_angle_2", "wheel_slip_angle_3"]].mean(axis=1)
    # Balance: positive → front sliding more (understeer tendency)
    df["alpha_balance"] = df["alpha_f"].abs() - df["alpha_r"].abs()
 
    df["kappa_f"] = df[["wheel_slip_ratio_0", "wheel_slip_ratio_1"]].mean(axis=1)
    df["kappa_r"] = df[["wheel_slip_ratio_2", "wheel_slip_ratio_3"]].mean(axis=1)
 
    df["tyre_temp_f"] = df[["tyre_temp_0", "tyre_temp_1"]].mean(axis=1)
    df["tyre_temp_r"] = df[["tyre_temp_2", "tyre_temp_3"]].mean(axis=1)
 
    # Cornering / phase
    thr = df.get("throttle", pd.Series(np.nan, index=df.index)).fillna(0.0)
    brk = df.get("brake", pd.Series(np.nan, index=df.index)).fillna(0.0)
    steer = df.get("steering", pd.Series(np.nan, index=df.index)).fillna(0.0)
    df["throttle"] = thr
    df["brake"] = brk
    df["steering"] = steer
    df["steer_abs"] = steer.abs()
 
    df["phase"] = classify_phase(df)
    df["speed_band"] = pd.cut(
        df["speed_kph"],
        bins=[-np.inf, V_LOW, V_HIGH, np.inf],
        labels=["low", "medium", "high"],
    )
 
    # Valid on-track samples
    df["valid_sample"] = (
        df["speed_kph"].notna()
        & (df["speed_kph"] > 5)
        & df.get("lap_number", pd.Series(1, index=df.index)).notna()
    )
    if "lap_number" in df.columns:
        df.loc[df["lap_number"] < 0, "valid_sample"] = False
    if "lap_time_invalid" in df.columns:
        # 1 often means invalid; keep NaN as ok
        df.loc[df["lap_time_invalid"] == 1, "valid_sample"] = False
    if "pit_status" in df.columns:
        df.loc[df["pit_status"].fillna(0) > 0, "valid_sample"] = False
 
    # Steering activity (sample-to-sample)
    df["steer_delta"] = df["steering"].diff().abs()
 
    return df
 
 
def _assign_g_axes(
    df: pd.DataFrame, gx: pd.Series, gy: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Return (g_long, g_lat). Prefer axis more correlated with steering as lateral.
    Fallback: |gy| peak higher → lat = Y (common in this export).
    """
    thr = df.get("throttle", pd.Series(0, index=df.index)).fillna(0)
    brk = df.get("brake", pd.Series(0, index=df.index)).fillna(0)
    steer = df.get("steering", pd.Series(0, index=df.index)).fillna(0)
    mask = (steer.abs() > 0.15) & (brk < 0.2) & (thr < 0.5)
    if mask.sum() > 50:
        sx = gx[mask].corr(steer[mask].abs())
        sy = gy[mask].corr(steer[mask].abs())
        sx = 0.0 if pd.isna(sx) else abs(sx)
        sy = 0.0 if pd.isna(sy) else abs(sy)
        if sy >= sx:
            return gx, gy  # X long, Y lat
        return gy, gx
    # peak magnitude fallback
    if gy.abs().max(skipna=True) >= gx.abs().max(skipna=True):
        return gx, gy
    return gy, gx
 
 
def classify_phase(df: pd.DataFrame) -> pd.Series:
    thr = df["throttle"].fillna(0)
    brk = df["brake"].fillna(0)
    g_lat = df["g_lat_abs"].fillna(0)
    g_long = df["g_long_signed"].fillna(0)
 
    phase = pd.Series(PHASE_STRAIGHT, index=df.index, dtype=object)
 
    # Entry: braking with rising / significant lateral
    entry = (brk > 0.12) & ((g_long < -0.15) | (g_lat > 0.4))
    # Exit: throttle on, accelerating, still some lateral
    exit_ = (thr > 0.25) & (brk < 0.08) & (g_long > -0.05) & (g_lat > 0.35)
    # Mid: little long input, high lateral
    mid = (brk < 0.08) & (thr < 0.30) & (g_lat > 0.55)
 
    phase[entry] = PHASE_ENTRY
    phase[mid] = PHASE_MID
    phase[exit_] = PHASE_EXIT
    # Mid wins over weak entry if both (apex)
    phase[mid & (brk < 0.05)] = PHASE_MID
    return phase
 
 
# ---------------------------------------------------------------------------
# Setup snapshot
# ---------------------------------------------------------------------------
 
 
def extract_setup(df: pd.DataFrame) -> dict[str, Any]:
    """Read setup from a mid-session valid row."""
    setup_cols = [c for c in df.columns if c.endswith("_setup") or c.endswith("_setup_0")
                  or c.startswith("wing_setup") or c.startswith("diff_")
                  or c.startswith("arb_setup") or c.startswith("camber_setup")
                  or c.startswith("toe_setup") or c.startswith("susp_")
                  or c.startswith("tyre_press_setup") or c in (
                      "brake_bias_setup", "brake_press_setup", "brake_engine_setup",
                      "front_brake_bias", "traction_ctrl_setup", "abs_setup",
                      "fuel_setup", "ballast_setup",
                  )]
    # simpler: any column containing 'setup'
    setup_cols = [c for c in df.columns if "setup" in c or c == "front_brake_bias"]
    valid = df[df["valid_sample"]] if "valid_sample" in df.columns else df
    if valid.empty:
        valid = df
    row = valid.iloc[len(valid) // 2]
    out = {}
    for c in setup_cols:
        v = row[c]
        if pd.isna(v) or v == SENTINEL:
            continue
        out[c] = float(v) if isinstance(v, (int, float, np.floating)) else v
    # Friendly aliases
    aliases = {
        "front_wing": out.get("wing_setup_0"),
        "rear_wing": out.get("wing_setup_1"),
        "diff_on_throttle": out.get("diff_onThrottle_setup"),
        "diff_off_throttle": out.get("diff_offThrottle_setup"),
        "arb_front": out.get("arb_setup_0"),
        "arb_rear": out.get("arb_setup_1"),
        "brake_bias": out.get("front_brake_bias", out.get("brake_bias_setup")),
    }
    out["_friendly"] = {k: v for k, v in aliases.items() if v is not None}
    return out
 
 
# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
 
 
def _corner_bin(distance_m: float, bin_m: float = 50.0) -> float:
    if pd.isna(distance_m):
        return np.nan
    return float(bin_m * round(distance_m / bin_m))
 
 
def run_diagnostics(df: pd.DataFrame) -> tuple[list[IssueEvent], list[IssueSummary]]:
    d = df[df["valid_sample"]].copy()
    if d.empty:
        return [], []
 
    events: list[IssueEvent] = []
 
    def add(issue_id, name, idx, severity, detail):
        row = d.loc[idx]
        events.append(
            IssueEvent(
                issue_id=issue_id,
                name=name,
                lap=float(row.get("lap_number", np.nan)),
                distance_m=float(row.get("lap_distance", np.nan)),
                speed_kph=float(row.get("speed_kph", np.nan)),
                phase=str(row.get("phase", "")),
                severity=float(severity),
                detail=detail,
            )
        )
 
    # --- Balance via slip angles (best channel in this export) ---
    # Mid-corner samples with meaningful lateral load
    mid = d[
        (d["phase"] == PHASE_MID)
        & (d["g_lat_abs"] > 0.6)
        & d["alpha_f"].notna()
        & d["alpha_r"].notna()
    ]
 
    for idx, row in mid.iterrows():
        bal = row["alpha_balance"]
        band = str(row["speed_band"])
        if bal > US_ALPHA_THRESH:
            sev = min(2.0, bal / US_ALPHA_THRESH)
            if band == "low":
                add(
                    "us_low",
                    "Low-speed understeer",
                    idx,
                    sev,
                    f"α_f−α_r={bal:.3f} rad, v={row['speed_kph']:.0f} kph, |ay|={row['g_lat_abs']:.2f}",
                )
            elif band == "high":
                add(
                    "us_high",
                    "High-speed understeer",
                    idx,
                    sev,
                    f"α_f−α_r={bal:.3f} rad, v={row['speed_kph']:.0f} kph, |ay|={row['g_lat_abs']:.2f}",
                )
            else:
                add(
                    "us_mid_speed",
                    "Medium-speed understeer",
                    idx,
                    sev,
                    f"α_f−α_r={bal:.3f} rad, v={row['speed_kph']:.0f} kph",
                )
        elif bal < -OS_ALPHA_THRESH:
            sev = min(2.0, abs(bal) / OS_ALPHA_THRESH)
            if band == "low":
                add(
                    "os_low",
                    "Low-speed oversteer",
                    idx,
                    sev,
                    f"α_r dominates, α_f−α_r={bal:.3f}, v={row['speed_kph']:.0f}",
                )
            elif band == "high":
                add(
                    "os_high",
                    "High-speed oversteer",
                    idx,
                    sev,
                    f"α_r dominates, α_f−α_r={bal:.3f}, v={row['speed_kph']:.0f}",
                )
            else:
                add(
                    "os_mid_speed",
                    "Medium-speed oversteer",
                    idx,
                    sev,
                    f"α_f−α_r={bal:.3f}, v={row['speed_kph']:.0f}",
                )
 
    # Entry under/oversteer
    entry = d[
        (d["phase"] == PHASE_ENTRY)
        & (d["g_lat_abs"] > 0.5)
        & d["alpha_balance"].notna()
    ]
    for idx, row in entry.iterrows():
        bal = row["alpha_balance"]
        if bal > US_ALPHA_THRESH * 1.1:
            add(
                "us_entry",
                "Entry understeer",
                idx,
                min(2.0, bal / US_ALPHA_THRESH),
                f"Turn-in push α_bal={bal:.3f}, brake={row['brake']:.2f}",
            )
        elif bal < -OS_ALPHA_THRESH * 1.1:
            add(
                "os_entry",
                "Entry oversteer",
                idx,
                min(2.0, abs(bal) / OS_ALPHA_THRESH),
                f"Rear rotates on entry α_bal={bal:.3f}, brake={row['brake']:.2f}",
            )
 
    # Exit traction oversteer / spin
    exit_df = d[(d["phase"] == PHASE_EXIT) & (d["throttle"] > 0.4)]
    for idx, row in exit_df.iterrows():
        kr = row.get("kappa_r", np.nan)
        bal = row.get("alpha_balance", np.nan)
        if pd.notna(kr) and kr > SPIN_SLIP:
            add(
                "traction_spin",
                "Exit traction limitation (wheelspin)",
                idx,
                min(2.0, kr / SPIN_SLIP),
                f"κ_r={kr:.3f}, throttle={row['throttle']:.2f}, gear={row.get('gear', float('nan'))}",
            )
        if pd.notna(bal) and bal < -OS_ALPHA_THRESH and row["throttle"] > 0.5:
            add(
                "os_exit",
                "Exit oversteer",
                idx,
                min(2.0, abs(bal) / OS_ALPHA_THRESH),
                f"Power oversteer α_bal={bal:.3f}, T={row['throttle']:.2f}",
            )
        if pd.notna(bal) and bal > US_ALPHA_THRESH and row["throttle"] > 0.55:
            add(
                "us_exit",
                "Exit understeer",
                idx,
                min(2.0, bal / US_ALPHA_THRESH),
                f"Push on power α_bal={bal:.3f}, T={row['throttle']:.2f}",
            )
 
    # Lockups
    braking = d[(d["brake"] > 0.35) & (d["speed_kph"] > 40)]
    for idx, row in braking.iterrows():
        kf = row.get("kappa_f", np.nan)
        kr = row.get("kappa_r", np.nan)
        # Negative slip ratio under brake = lock tendency in many game exports
        if pd.notna(kf) and abs(kf) > LOCK_SLIP and kf < 0:
            add(
                "lock_front",
                "Front lockup",
                idx,
                min(2.0, abs(kf) / LOCK_SLIP),
                f"κ_f={kf:.3f}, brake={row['brake']:.2f}, v={row['speed_kph']:.0f}",
            )
        if pd.notna(kr) and abs(kr) > LOCK_SLIP and kr < 0:
            add(
                "lock_rear",
                "Rear lockup",
                idx,
                min(2.0, abs(kr) / LOCK_SLIP),
                f"κ_r={kr:.3f}, brake={row['brake']:.2f}, v={row['speed_kph']:.0f}",
            )
 
    # Steering corrections mid-corner (instability / OS proxy)
    busy = d[
        (d["phase"].isin([PHASE_MID, PHASE_EXIT]))
        & (d["g_lat_abs"] > 0.7)
        & (d["steer_delta"] > 0.04)
    ]
    for idx, row in busy.iterrows():
        add(
            "steer_corrections",
            "High steering correction (instability)",
            idx,
            min(2.0, row["steer_delta"] / 0.04),
            f"Δsteer={row['steer_delta']:.3f}, phase={row['phase']}, v={row['speed_kph']:.0f}",
        )
 
    # Tire temps
    for idx, row in d[d["tyre_temp_f"].notna()].iloc[::5].iterrows():
        tf, tr = row["tyre_temp_f"], row["tyre_temp_r"]
        if tf < TIRE_COLD or tr < TIRE_COLD:
            add(
                "tires_cold",
                "Tires below temperature window",
                idx,
                max(0.5, (TIRE_COLD - min(tf, tr)) / 10),
                f"T_f={tf:.0f}°C T_r={tr:.0f}°C",
            )
        if tf > TIRE_HOT or tr > TIRE_HOT:
            add(
                "tires_hot",
                "Tires above temperature window",
                idx,
                max(0.5, (max(tf, tr) - TIRE_HOT) / 10),
                f"T_f={tf:.0f}°C T_r={tr:.0f}°C",
            )
        if abs(tf - tr) > 12:
            add(
                "tires_axle_imbalance",
                "Front/rear tire temp imbalance",
                idx,
                min(2.0, abs(tf - tr) / 12),
                f"T_f={tf:.0f} T_r={tr:.0f} (Δ={tf - tr:.0f})",
            )
 
    # Aero / speed-dependent balance: bin U = steer/|ay| vs speed
    cornering = d[(d["g_lat_abs"] > 0.7) & (d["steer_abs"] > 0.05)]
    if len(cornering) > 100:
        low_u = (
            cornering.loc[cornering["speed_band"] == "low", "steer_abs"]
            / cornering.loc[cornering["speed_band"] == "low", "g_lat_abs"].clip(lower=0.2)
        )
        high_u = (
            cornering.loc[cornering["speed_band"] == "high", "steer_abs"]
            / cornering.loc[cornering["speed_band"] == "high", "g_lat_abs"].clip(lower=0.2)
        )
        if len(low_u) > 30 and len(high_u) > 30:
            # Compare medians via synthetic events on high-speed subset
            med_l, med_h = low_u.median(), high_u.median()
            if med_h > med_l * 1.15:
                # more steering per G at speed → high-speed US aero signature
                for idx in high_u.nlargest(min(40, len(high_u))).index:
                    add(
                        "aero_us_hs",
                        "Aero imbalance (high-speed understeer trend)",
                        idx,
                        min(2.0, med_h / max(med_l, 1e-3)),
                        f"U_high={med_h:.3f} vs U_low={med_l:.3f} (steer/|ay|)",
                    )
            elif med_l > med_h * 1.15:
                for idx in low_u.nlargest(min(40, len(low_u))).index:
                    add(
                        "aero_os_or_mech",
                        "Low-speed mechanical understeer vs aero",
                        idx,
                        min(2.0, med_l / max(med_h, 1e-3)),
                        f"U_low={med_l:.3f} vs U_high={med_h:.3f} — prefer mechanical fix",
                    )
 
    summaries = summarize_events(events, d)
    return events, summaries
 
 
def summarize_events(
    events: list[IssueEvent], d: pd.DataFrame
) -> list[IssueSummary]:
    if not events:
        return []
 
    laps = d["lap_number"].dropna().unique()
    total_laps = max(1, len([x for x in laps if x >= 0]))
 
    by_id: dict[str, list[IssueEvent]] = {}
    for e in events:
        by_id.setdefault(e.issue_id, []).append(e)
 
    out: list[IssueSummary] = []
    for issue_id, evs in by_id.items():
        # Downsample consecutive samples into "events": cluster by lap + 50m bin
        clusters: dict[tuple, list[IssueEvent]] = {}
        for e in evs:
            key = (round(e.lap, 0), _corner_bin(e.distance_m, 50))
            clusters.setdefault(key, []).append(e)
        cluster_list = list(clusters.values())
        count = len(cluster_list)
        laps_hit = {round(e.lap, 0) for e in evs if not np.isnan(e.lap)}
        severities = [max(x.severity for x in cl) for cl in cluster_list]
        # Hot spots: most common distance bins
        bin_counts: dict[float, int] = {}
        for (lap, dist), cl in clusters.items():
            if dist is None or (isinstance(dist, float) and np.isnan(dist)):
                continue
            bin_counts[dist] = bin_counts.get(dist, 0) + 1
        hot = sorted(bin_counts.keys(), key=lambda k: -bin_counts[k])[:5]
 
        # Confidence: more clusters + more laps → higher
        conf = min(
            1.0,
            0.35
            + 0.1 * min(count, 5)
            + 0.4 * (len(laps_hit) / total_laps)
            + 0.1 * (np.mean(severities) if severities else 0),
        )
 
        out.append(
            IssueSummary(
                issue_id=issue_id,
                name=evs[0].name,
                count=count,
                events_per_lap=count / total_laps,
                laps_present=len(laps_hit),
                total_laps=total_laps,
                lap_presence_pct=100.0 * len(laps_hit) / total_laps,
                hot_corners_m=hot,
                mean_severity=float(np.mean(severities)) if severities else 0.0,
                max_severity=float(np.max(severities)) if severities else 0.0,
                sample_details=[cl[0].detail for cl in cluster_list[:3]],
                confidence=conf,
            )
        )
 
    # Sort by frequency * severity * confidence
    out.sort(
        key=lambda s: s.count * s.mean_severity * s.confidence,
        reverse=True,
    )
    return out
 
 
# ---------------------------------------------------------------------------
# Setup recommendations
# ---------------------------------------------------------------------------
 
# issue_id → list of change templates
ISSUE_TO_CHANGES: dict[str, list[dict]] = {
    "us_low": [
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1 to +2 clicks",
            "reason": "Low-speed mid-corner understeer (front slip angle > rear).",
            "validation_metric": "α_balance mid-corner low-speed closer to 0; less steer for same |ay|",
            "weight": 1.0,
        },
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Softer front anti-roll adds front mechanical grip in slow corners.",
            "validation_metric": "Fewer us_low events per lap at same corners",
            "weight": 0.9,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Stiffer rear can rotate the car mid-corner at low speed (use if OS not already present).",
            "validation_metric": "No rise in os_low frequency",
            "weight": 0.5,
        },
    ],
    "us_high": [
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1 to +3",
            "reason": "High-speed understeer — aero front load shortfall.",
            "validation_metric": "us_high count down; high-speed mid |ay| up",
            "weight": 1.2,
        },
        {
            "parameter": "Rear wing",
            "direction": "decrease",
            "amount_hint": "−1 (if top speed allows)",
            "reason": "Reduces rear aero dominance that pushes the car wide in fast corners.",
            "validation_metric": "Trap speed acceptable; os_high not increased",
            "weight": 0.7,
        },
    ],
    "us_mid_speed": [
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Medium-speed mid-corner push.",
            "validation_metric": "us_mid_speed frequency",
            "weight": 0.8,
        },
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1",
            "reason": "Mechanical front grip for mid-speed corners.",
            "validation_metric": "α_balance mid phase",
            "weight": 0.8,
        },
    ],
    "os_low": [
        {
            "parameter": "On-throttle differential",
            "direction": "decrease",
            "amount_hint": "−10% to −20% (e.g. 90% → 70–80%)",
            "reason": "Low-speed oversteer — less aggressive locking on power/mid helps rear stability.",
            "validation_metric": "os_low and os_exit counts",
            "weight": 0.7,
        },
        {
            "parameter": "Rear ARB",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Softer rear anti-roll reduces low-speed rear slip.",
            "validation_metric": "α_balance not rear-dominant in low speed",
            "weight": 1.0,
        },
        {
            "parameter": "Rear wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Extra rear load stabilizes the rear (small effect at very low speed).",
            "validation_metric": "os_low frequency",
            "weight": 0.4,
        },
    ],
    "os_high": [
        {
            "parameter": "Rear wing",
            "direction": "increase",
            "amount_hint": "+1 to +2",
            "reason": "High-speed rear instability / oversteer.",
            "validation_metric": "os_high count; steer corrections at v>200",
            "weight": 1.2,
        },
        {
            "parameter": "Rear ARB",
            "direction": "decrease",
            "amount_hint": "−1",
            "reason": "More rear mechanical compliance in fast direction changes.",
            "validation_metric": "steer_corrections high-speed",
            "weight": 0.8,
        },
    ],
    "os_entry": [
        {
            "parameter": "Brake bias",
            "direction": "increase",
            "amount_hint": "+1% to +2% forward",
            "reason": "Entry oversteer — more forward bias stabilizes rear on turn-in.",
            "validation_metric": "os_entry frequency; rear lockups not rising",
            "weight": 1.1,
        },
        {
            "parameter": "Off-throttle differential",
            "direction": "increase",
            "amount_hint": "+5% to +10%",
            "reason": "More coast locking can stabilize entry rotation (game-dependent feel).",
            "validation_metric": "os_entry events",
            "weight": 0.6,
        },
    ],
    "us_entry": [
        {
            "parameter": "Brake bias",
            "direction": "decrease",
            "amount_hint": "−1% forward (more rear)",
            "reason": "Entry understeer — slight rear bias helps rotation on trail-brake.",
            "validation_metric": "us_entry count; watch rear lockups",
            "weight": 1.0,
        },
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "More front aero/mechanical turn-in bite.",
            "validation_metric": "us_entry frequency",
            "weight": 0.7,
        },
    ],
    "os_exit": [
        {
            "parameter": "On-throttle differential",
            "direction": "decrease",
            "amount_hint": "−10% to −20%",
            "reason": "Exit oversteer — open on-throttle diff reduces inside-wheel drive spike.",
            "validation_metric": "os_exit and traction_spin counts",
            "weight": 1.2,
        },
        {
            "parameter": "Rear ARB",
            "direction": "decrease",
            "amount_hint": "−1",
            "reason": "Helps put power down with a calmer rear.",
            "validation_metric": "κ_r on exit",
            "weight": 0.8,
        },
    ],
    "us_exit": [
        {
            "parameter": "On-throttle differential",
            "direction": "increase",
            "amount_hint": "+5% to +10%",
            "reason": "Exit understeer / push on power — more locking can help rotate (if not spinning).",
            "validation_metric": "us_exit; ensure traction_spin does not rise",
            "weight": 0.7,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Can free the rear slightly on exit to reduce push.",
            "validation_metric": "us_exit frequency",
            "weight": 0.6,
        },
    ],
    "traction_spin": [
        {
            "parameter": "On-throttle differential",
            "direction": "decrease",
            "amount_hint": "−10% to −20%",
            "reason": "Rear wheelspin on exit (high κ_r).",
            "validation_metric": "Mean peak κ_r on exit per lap",
            "weight": 1.3,
        },
        {
            "parameter": "Rear wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Slightly more rear load for traction (minor at very low speed).",
            "validation_metric": "traction_spin frequency",
            "weight": 0.4,
        },
    ],
    "lock_front": [
        {
            "parameter": "Brake bias",
            "direction": "decrease",
            "amount_hint": "−1% to −2% forward",
            "reason": "Front lockups under braking.",
            "validation_metric": "lock_front event count",
            "weight": 1.2,
        },
        {
            "parameter": "Brake pressure",
            "direction": "decrease",
            "amount_hint": "−1% to −2% if available",
            "reason": "Softer initial bite if still locking after bias change.",
            "validation_metric": "lock_front severity",
            "weight": 0.5,
        },
    ],
    "lock_rear": [
        {
            "parameter": "Brake bias",
            "direction": "increase",
            "amount_hint": "+1% to +2% forward",
            "reason": "Rear lockups — move bias forward.",
            "validation_metric": "lock_rear count",
            "weight": 1.2,
        },
    ],
    "steer_corrections": [
        {
            "parameter": "Rear wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "High mid/exit steering corrections suggest rear nervousness.",
            "validation_metric": "steer_corrections frequency",
            "weight": 0.6,
        },
        {
            "parameter": "Rear ARB",
            "direction": "decrease",
            "amount_hint": "−1",
            "reason": "Calmer rear mechanical response.",
            "validation_metric": "steer_corrections + os_* counts",
            "weight": 0.7,
        },
    ],
    "aero_us_hs": [
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1 to +2",
            "reason": "Steer-per-G rises with speed (aero understeer trend).",
            "validation_metric": "U_high vs U_low ratio closer to 1",
            "weight": 1.1,
        },
    ],
    "tires_cold": [
        {
            "parameter": "Driving / out-lap",
            "direction": "adjust",
            "amount_hint": "Warm-up: weave, later brake, build load",
            "reason": "Tires below window — setup changes are secondary until temps rise.",
            "validation_metric": "tyre_temp_* into window before push laps",
            "weight": 0.3,
        },
    ],
    "tires_hot": [
        {
            "parameter": "Driving / wing or camber",
            "direction": "adjust",
            "amount_hint": "Reduce sliding; small camber/pressure tweaks",
            "reason": "Over-temp tires often from sustained slide (fix balance first).",
            "validation_metric": "Peak tyre_temp and slide issue frequency",
            "weight": 0.4,
        },
    ],
    "tires_axle_imbalance": [
        {
            "parameter": "Wing / ARB balance",
            "direction": "adjust",
            "amount_hint": "Align with hotter axle diagnosis",
            "reason": "Front much hotter → front sliding (US); rear hotter → rear sliding (OS).",
            "validation_metric": "|T_f − T_r| mid-stint",
            "weight": 0.5,
        },
    ],
}
 
 
def recommend_setup(
    summaries: list[IssueSummary], setup: dict[str, Any]
) -> list[SetupChange]:
    """Aggregate issue-linked changes; merge duplicates; rank by priority."""
    acc: dict[str, SetupChange] = {}
 
    for s in summaries:
        templates = ISSUE_TO_CHANGES.get(s.issue_id, [])
        freq_factor = min(2.0, 0.5 + s.events_per_lap + 0.01 * s.lap_presence_pct)
        for t in templates:
            key = f"{t['parameter']}|{t['direction']}"
            pr = t["weight"] * s.mean_severity * s.confidence * freq_factor
            if key not in acc:
                acc[key] = SetupChange(
                    parameter=t["parameter"],
                    direction=t["direction"],
                    amount_hint=t["amount_hint"],
                    reason=t["reason"]
                    + f" [seen {s.count}×, {s.lap_presence_pct:.0f}% of laps]",
                    linked_issues=[s.name],
                    priority=pr,
                    validation_metric=t["validation_metric"],
                )
            else:
                acc[key].priority += pr
                if s.name not in acc[key].linked_issues:
                    acc[key].linked_issues.append(s.name)
                # strengthen reason with frequency once
                if "seen" not in acc[key].reason:
                    acc[key].reason += f" [freq weighted]"
 
    changes = sorted(acc.values(), key=lambda c: c.priority, reverse=True)
 
    # Conflict filter: don't increase and decrease same param in top list
    final: list[SetupChange] = []
    seen_param_dir: dict[str, str] = {}
    for c in changes:
        prev = seen_param_dir.get(c.parameter)
        if prev and prev != c.direction and c.direction in ("increase", "decrease"):
            continue  # skip conflicting lower-priority change
        seen_param_dir[c.parameter] = c.direction
        final.append(c)
        if len(final) >= 8:
            break
 
    # Annotate with current setup values when known
    friendly = setup.get("_friendly", {})
    param_map = {
        "Front wing": "front_wing",
        "Rear wing": "rear_wing",
        "On-throttle differential": "diff_on_throttle",
        "Off-throttle differential": "diff_off_throttle",
        "Front ARB": "arb_front",
        "Rear ARB": "arb_rear",
        "Brake bias": "brake_bias",
    }
    for c in final:
        key = param_map.get(c.parameter)
        if key and key in friendly:
            c.amount_hint = f"{c.amount_hint} (current ≈ {friendly[key]})"
 
    return final
 
 
# ---------------------------------------------------------------------------
# Session stats helpers
# ---------------------------------------------------------------------------
 
 
def session_overview(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df["valid_sample"]] if "valid_sample" in df.columns else df
    laps = sorted([x for x in d["lap_number"].dropna().unique() if x >= 0])
    lap_times = []
    for lap in laps:
        sub = d[d["lap_number"] == lap]
        if "lap_time" in sub.columns and sub["lap_time"].notna().any():
            lt = sub["lap_time"].max()
            if pd.notna(lt) and lt > 0:
                lap_times.append((lap, float(lt)))
    best = min(lap_times, key=lambda x: x[1]) if lap_times else None
    return {
        "track": df["trackId"].dropna().iloc[0] if "trackId" in df.columns and df["trackId"].notna().any() else "?",
        "car": df["carId"].dropna().iloc[0] if "carId" in df.columns and df["carId"].notna().any() else "?",
        "laps": laps,
        "lap_times": lap_times,
        "best_lap": best,
        "samples": len(d),
        "vmax_kph": float(d["speed_kph"].max()) if len(d) else 0.0,
        "track_temp": float(d["track_temp"].median()) if "track_temp" in d and d["track_temp"].notna().any() else None,
        "air_temp": float(d["air_temp"].median()) if "air_temp" in d and d["air_temp"].notna().any() else None,
    }
 
 
# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
 
 
def main():
    st.set_page_config(
        page_title="Virtual Race Engineer",
        page_icon="🏎️",
        layout="wide",
    )
    st.title("🏎️ Virtual Race Engineer")
    st.caption(
        "F1 25 / F1 26 telemetry → diagnostics with **frequency** → ranked setup changes with reasons."
    )
 
    with st.sidebar:
        st.header("Session")
        uploaded = st.file_uploader(
            "Telemetry file (TSV/CSV)",
            type=["csv", "tsv", "txt"],
            help="Tab-separated F1 export (e.g. F12025-….csv)",
        )
        st.markdown("---")
        st.subheader("Thresholds")
        global US_ALPHA_THRESH, OS_ALPHA_THRESH, LOCK_SLIP, SPIN_SLIP
        US_ALPHA_THRESH = st.slider("Understeer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        OS_ALPHA_THRESH = st.slider("Oversteer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        LOCK_SLIP = st.slider("Lockup |κ| threshold", 0.03, 0.25, 0.08, 0.01)
        SPIN_SLIP = st.slider("Wheelspin κ threshold", 0.05, 0.35, 0.12, 0.01)
        min_count = st.number_input("Min event count to show issue", 1, 50, 2)
        min_presence = st.slider("Min lap presence %", 0, 100, 15)
        st.markdown("---")
        st.caption(
            "Wheel map: 0=FL 1=FR 2=RL 3=RR · Sentinel −1 → missing · "
            "Phases: entry / mid / exit from brake, throttle, G."
        )
 
    if uploaded is None:
        st.info(
            "Upload a telemetry export to begin. "
            "Your logger format (tab-separated, 266 columns) is supported."
        )
        with st.expander("What this app assesses"):
            st.markdown(
                """
- **Balance:** low/med/high-speed understeer & oversteer (slip angles)
- **Phases:** entry / mid / exit
- **Brakes:** front & rear lockup
- **Traction:** exit wheelspin & power oversteer
- **Tires:** cold / hot / axle imbalance
- **Aero trend:** steer-per-G vs speed
- Every issue includes **count, per-lap rate, % of laps, hot distance bins, confidence**
                """
            )
        return
 
    with st.spinner("Loading and analyzing…"):
        try:
            df = load_telemetry(uploaded)
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return
 
        overview = session_overview(df)
        setup = extract_setup(df)
        events, summaries = run_diagnostics(df)
 
        # Filter for display
        shown = [
            s
            for s in summaries
            if s.count >= min_count and s.lap_presence_pct >= min_presence
        ]
        changes = recommend_setup(shown if shown else summaries[:5], setup)
 
    # ----- Overview -----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Track", str(overview["track"]))
    c2.metric("Laps", f"{len(overview['laps'])}")
    if overview["best_lap"]:
        c3.metric("Best lap", f"L{overview['best_lap'][0]:.0f}  {overview['best_lap'][1]:.3f}s")
    else:
        c3.metric("Best lap", "—")
    c4.metric("Vmax", f"{overview['vmax_kph']:.0f} kph")
    c5.metric("Samples", f"{overview['samples']:,}")
 
    if setup.get("_friendly"):
        st.subheader("Current setup (from telemetry)")
        st.json(setup["_friendly"])
 
    # ----- Top recommendations -----
    st.subheader("Recommended setup changes")
    if not changes:
        st.success("No strong setup signals — check thresholds or drive more representative laps.")
    else:
        for i, ch in enumerate(changes, 1):
            with st.container(border=True):
                st.markdown(
                    f"**{i}. {ch.parameter}** — `{ch.direction}` · {ch.amount_hint}"
                )
                st.write(ch.reason)
                st.caption(
                    f"Linked: {', '.join(ch.linked_issues)} · "
                    f"Priority {ch.priority:.2f} · Validate: {ch.validation_metric}"
                )
 
    # ----- Issues table -----
    st.subheader("Issues (with frequency)")
    if not shown:
        st.warning(
            "No issues passed the frequency filters. Lower “Min event count” / "
            “Min lap presence” in the sidebar, or inspect raw summaries below."
        )
        shown = summaries[:15]
 
    if shown:
        table = pd.DataFrame(
            [
                {
                    "Issue": s.name,
                    "Count": s.count,
                    "Per lap": round(s.events_per_lap, 2),
                    "Laps": f"{s.laps_present}/{s.total_laps}",
                    "Presence %": round(s.lap_presence_pct, 1),
                    "Avg severity": round(s.mean_severity, 2),
                    "Confidence": round(s.confidence, 2),
                    "Hot spots (m)": ", ".join(f"{h:.0f}" for h in s.hot_corners_m),
                    "Example": s.sample_details[0] if s.sample_details else "",
                }
                for s in shown
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
 
        # Frequency bar
        fig = px.bar(
            table,
            x="Issue",
            y="Count",
            color="Presence %",
            title="Issue frequency (clustered events)",
        )
        fig.update_layout(xaxis_tickangle=-35, height=400)
        st.plotly_chart(fig, use_container_width=True)
 
    # ----- Charts -----
    st.subheader("Telemetry snapshots")
    d = df[df["valid_sample"]].copy()
    if not d.empty:
        lap_opts = sorted([int(x) for x in d["lap_number"].dropna().unique() if x >= 0])
        if lap_opts:
            lap_sel = st.selectbox("Lap for trace", lap_opts, index=len(lap_opts) - 1)
            ld = d[d["lap_number"] == lap_sel].sort_values("lap_distance")
            if len(ld) > 5:
                t1, t2 = st.tabs(["Speed / inputs", "Balance (slip angles)"])
                with t1:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=ld["lap_distance"],
                            y=ld["speed_kph"],
                            name="Speed kph",
                            yaxis="y1",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=ld["lap_distance"],
                            y=ld["throttle"] * 100,
                            name="Throttle %",
                            yaxis="y2",
                            opacity=0.7,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=ld["lap_distance"],
                            y=ld["brake"] * 100,
                            name="Brake %",
                            yaxis="y2",
                            opacity=0.7,
                        )
                    )
                    fig.update_layout(
                        height=380,
                        yaxis=dict(title="kph"),
                        yaxis2=dict(
                            title="Input %",
                            overlaying="y",
                            side="right",
                            range=[0, 100],
                        ),
                        legend=dict(orientation="h"),
                        margin=dict(l=40, r=40, t=30, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with t2:
                    fig2 = go.Figure()
                    fig2.add_trace(
                        go.Scatter(
                            x=ld["lap_distance"],
                            y=ld["alpha_balance"],
                            name="α_f − α_r (abs balance)",
                        )
                    )
                    fig2.add_hline(y=US_ALPHA_THRESH, line_dash="dot", annotation_text="US")
                    fig2.add_hline(y=-OS_ALPHA_THRESH, line_dash="dot", annotation_text="OS")
                    fig2.update_layout(
                        height=380,
                        xaxis_title="Distance (m)",
                        yaxis_title="Slip angle balance (rad)",
                        margin=dict(l=40, r=40, t=30, b=40),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
 
                    st.caption(
                        "Positive α balance → front sliding more (**understeer**). "
                        "Negative → rear sliding more (**oversteer**)."
                    )
 
    with st.expander("Engineer notes / method"):
        st.markdown(
            f"""
**Phases:** entry (brake + long G), mid (low throttle/brake + high |ay|), exit (throttle + residual lateral).
 
**Understeer / oversteer:** compare mean front vs rear wheel slip angles  
`α_balance = |α_f| − |α_r|` with thresholds {US_ALPHA_THRESH} / {OS_ALPHA_THRESH} rad.
 
**Frequency:** raw samples are clustered by **lap + 50 m distance bin** so one long push corner
counts as one event, not hundreds of rows. Presence % = share of laps where the issue appeared.
 
**Priority:** `severity × confidence × frequency` then mapped to setup levers; conflicting
directions on the same parameter are dropped (higher priority wins).
 
**Axes:** lateral/longitudinal G auto-assigned from correlation with steering when possible.
            """
        )
 
    with st.expander("All issue IDs (debug)"):
        st.write([asdict(s) for s in summaries])
 
 
if __name__ == "__main__":
    main()
