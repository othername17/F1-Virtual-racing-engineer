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

# F1 25 / F1 26 car setup limits.
# Only ranges marked source="user" are confirmed from the user / in-game UI.
# Other entries remain placeholders until confirmed — do not treat them as fact.
SETUP_LIMITS: dict[str, dict[str, Any]] = {
    "Front wing": {
        "field": "wing_setup_0",
        "min": 0,
        "max": 50,
        "step": 1,  # software: amount per click/increment
        "unit": "clicks",
        "source": "user",  # confirmed: Wings 0-50
    },
    "Rear wing": {
        "field": "wing_setup_1",
        "min": 0,
        "max": 50,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: Wings 0-50
    },
    "Front ARB": {
        "field": "arb_setup_0",
        "min": 1,
        "max": 21,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: front and rear ARB 1-21
    },
    "Rear ARB": {
        "field": "arb_setup_1",
        "min": 1,
        "max": 21,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: front and rear ARB 1-21
    },
    "On-throttle differential": {
        "field": "diff_onThrottle_setup",
        "min": 0.10,  # 10%
        "max": 1.00,  # 100%
        "step": 0.05,
        "unit": "fraction (UI % = value×100)",
        "display_as_pct": True,
        "source": "user",  # confirmed: on and off throttle 10%-100%
    },
    "Off-throttle differential": {
        "field": "diff_offThrottle_setup",
        "min": 0.10,  # 10%
        "max": 1.00,  # 100%
        "step": 0.05,
        "unit": "fraction (UI % = value×100)",
        "display_as_pct": True,
        "source": "user",  # confirmed: on and off throttle 10%-100%
    },
    # Percent is FRONT share. 70% = more forward; 50% = more rearward.
    "Brake bias (% front)": {
        "field": "front_brake_bias",
        "alt_fields": ["brake_bias_setup"],
        "min": 0.50,  # 50% front = more rearward end of range
        "max": 0.70,  # 70% front = more forward end of range
        "step": 0.01,
        "unit": "% front (70% = more forward, 50% = more rearward)",
        "display_as_pct": True,
        "min_label": "more rearward 50%",
        "max_label": "more forward 70%",
        "increase_means": "more forward (toward 70% front)",
        "decrease_means": "more rearward (toward 50% front)",
        "source": "user",
    },
    "Brake pressure": {
        "field": "brake_press_setup",
        "min": 0.80,  # 80%
        "max": 1.00,  # 100%
        "step": 0.01,
        "unit": "percent",
        "display_as_pct": True,
        "min_label": "80%",
        "max_label": "100%",
        "source": "user",  # confirmed: brake pressure 80% to 100%
    },
    "Front tire pressure": {
        "field": "tyre_press_setup_0",
        "alt_fields": ["tyre_press_setup_1"],
        "min": 22.5,
        "max": 29.5,
        "step": 0.1,
        "unit": "psi (in-game)",
        # Telemetry stores ~Pascals (e.g. 201327 ≈ 29.2 psi)
        "telemetry_in_pascals": True,
        "min_label": "22.5 psi",
        "max_label": "29.5 psi",
        "source": "user",  # confirmed: front tire pressure 22.5-29.5
    },
    "Rear tire pressure": {
        "field": "tyre_press_setup_2",
        "alt_fields": ["tyre_press_setup_3"],
        "min": 20.5,
        "max": 26.5,
        "step": 0.1,
        "unit": "psi (in-game)",
        "telemetry_in_pascals": True,
        "min_label": "20.5 psi",
        "max_label": "26.5 psi",
        "source": "user",  # confirmed: rear tire pressure 20.5-26.5
    },
    "Front spring": {
        "field": "susp_spring_setup_0",
        "min": 1,
        "max": 41,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: front and rear suspension 1-41
    },
    "Rear spring": {
        "field": "susp_spring_setup_2",
        "min": 1,
        "max": 41,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: front and rear suspension 1-41
    },
    "Front camber": {
        "field": "camber_setup_0",
        "alt_fields": ["camber_setup_1"],
        "min": -3.5,
        "max": -2.5,
        "step": 0.1,
        "unit": "degrees (in-game)",
        # Telemetry stores radians; convert before compare/display
        "telemetry_in_radians": True,
        "source": "user",  # confirmed: front camber -3.5 to -2.5 deg
    },
    "Rear camber": {
        "field": "camber_setup_2",
        "alt_fields": ["camber_setup_3"],
        "min": -2.0,
        "max": -1.0,
        "step": 0.1,
        "unit": "degrees (in-game)",
        "telemetry_in_radians": True,
        "source": "user",  # confirmed: rear camber -2 to -1 deg
    },
    "Front toe out": {
        "field": "toe_setup_0",
        "alt_fields": ["toe_setup_1"],
        "min": 0.0,
        "max": 0.2,
        "step": 0.01,
        "unit": "degrees (in-game)",
        "telemetry_in_radians": True,
        "source": "user",  # confirmed: front toe out 0 to 0.2
    },
    "Rear toe in": {
        "field": "toe_setup_2",
        "alt_fields": ["toe_setup_3"],
        "min": 0.1,
        "max": 0.25,
        "step": 0.01,
        "unit": "degrees (in-game)",
        "telemetry_in_radians": True,
        "source": "user",  # confirmed: rear toe in 0.1 to 0.25
    },
    "Front ride height": {
        "field": "susp_height_setup_0",
        "min": 15,
        "max": 35,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: front height 15-35
    },
    "Rear ride height": {
        "field": "susp_height_setup_2",
        "min": 40,
        "max": 60,
        "step": 1,
        "unit": "clicks",
        "source": "user",  # confirmed: rear height 40-60
    },
}

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
    criticality: float = 0.0  # combined impact proxy (not true tenths)
    tier: str = "C"  # S / A / B / C


@dataclass
class SetupChange:
    parameter: str
    direction: str  # "increase" | "decrease" | "adjust"
    amount_hint: str
    reason: str
    linked_issues: list
    priority: float
    validation_metric: str
    current: Optional[float] = None
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    feasible: bool = True
    blocked_reason: str = ""
    issue_id: str = ""
    option_label: str = ""  # e.g. "Option A"


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
    """Coerce numerics, map sentinel -1 → NaN on channels where -1 means invalid.

    Builds derived columns in one batch (avoids pandas PerformanceWarning from
    calling frame.insert / assigning many columns one-by-one on a 260+ col frame).
    """
    df = df.copy()
    idx = df.index

    # Bulk numeric coerce (identity / label cols stay as-is)
    keep_as_is = {"carId", "trackId"}
    num_cols = [c for c in df.columns if c not in keep_as_is]
    if num_cols:
        coerced = {c: _to_num(df[c]) for c in num_cols}
        id_part = df[[c for c in df.columns if c in keep_as_is]]
        df = pd.concat([id_part, pd.DataFrame(coerced, index=idx)], axis=1)
        # restore original column order where possible
        ordered = [c for c in list(id_part.columns) + num_cols if c in df.columns]
        df = df.reindex(columns=ordered)

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

    # --- Derived channels (compute series first, join once) ---
    derived: dict[str, pd.Series] = {}

    vx = df["velocity_X"].fillna(0.0) if "velocity_X" in df.columns else pd.Series(0.0, index=idx)
    vy = df["velocity_Y"].fillna(0.0) if "velocity_Y" in df.columns else pd.Series(0.0, index=idx)
    vz = df["velocity_Z"].fillna(0.0) if "velocity_Z" in df.columns else pd.Series(0.0, index=idx)
    derived["speed_ms"] = np.sqrt(vx**2 + vy**2 + vz**2)
    derived["speed_kph"] = derived["speed_ms"] * 3.6

    gx = df["gforce_X"] if "gforce_X" in df.columns else pd.Series(np.nan, index=idx)
    gy = df["gforce_Y"] if "gforce_Y" in df.columns else pd.Series(np.nan, index=idx)
    # Temporary columns needed for G-axis correlation helper
    tmp = df.copy()
    for i in range(4):
        sa = f"wheel_slip_angle_{i}"
        sr = f"wheel_slip_ratio_{i}"
        if sa not in tmp.columns:
            tmp[sa] = np.nan
        if sr not in tmp.columns:
            tmp[sr] = np.nan
    g_long, g_lat = _assign_g_axes(tmp, gx, gy)
    derived["g_long"] = g_long
    derived["g_lat"] = g_lat
    derived["g_lat_abs"] = g_lat.abs()
    derived["g_long_signed"] = g_long

    for i in range(4):
        sa = f"wheel_slip_angle_{i}"
        sr = f"wheel_slip_ratio_{i}"
        if sa not in df.columns:
            derived[sa] = pd.Series(np.nan, index=idx)
        if sr not in df.columns:
            derived[sr] = pd.Series(np.nan, index=idx)

    # Merge missing wheel cols into working frame for axle means
    work = pd.concat([df, pd.DataFrame({k: v for k, v in derived.items() if k.startswith("wheel_")}, index=idx)], axis=1)
    for i in range(4):
        sa, sr = f"wheel_slip_angle_{i}", f"wheel_slip_ratio_{i}"
        if sa not in work.columns:
            work[sa] = np.nan
        if sr not in work.columns:
            work[sr] = np.nan

    derived["alpha_f"] = work[["wheel_slip_angle_0", "wheel_slip_angle_1"]].mean(axis=1)
    derived["alpha_r"] = work[["wheel_slip_angle_2", "wheel_slip_angle_3"]].mean(axis=1)
    derived["alpha_balance"] = derived["alpha_f"].abs() - derived["alpha_r"].abs()
    derived["kappa_f"] = work[["wheel_slip_ratio_0", "wheel_slip_ratio_1"]].mean(axis=1)
    derived["kappa_r"] = work[["wheel_slip_ratio_2", "wheel_slip_ratio_3"]].mean(axis=1)

    t0 = work["tyre_temp_0"] if "tyre_temp_0" in work.columns else pd.Series(np.nan, index=idx)
    t1 = work["tyre_temp_1"] if "tyre_temp_1" in work.columns else pd.Series(np.nan, index=idx)
    t2 = work["tyre_temp_2"] if "tyre_temp_2" in work.columns else pd.Series(np.nan, index=idx)
    t3 = work["tyre_temp_3"] if "tyre_temp_3" in work.columns else pd.Series(np.nan, index=idx)
    derived["tyre_temp_f"] = pd.concat([t0, t1], axis=1).mean(axis=1)
    derived["tyre_temp_r"] = pd.concat([t2, t3], axis=1).mean(axis=1)

    thr = work["throttle"].fillna(0.0) if "throttle" in work.columns else pd.Series(0.0, index=idx)
    brk = work["brake"].fillna(0.0) if "brake" in work.columns else pd.Series(0.0, index=idx)
    steer = work["steering"].fillna(0.0) if "steering" in work.columns else pd.Series(0.0, index=idx)
    derived["throttle"] = thr
    derived["brake"] = brk
    derived["steering"] = steer
    derived["steer_abs"] = steer.abs()
    derived["steer_delta"] = steer.diff().abs()

    # Need g_lat_abs / inputs on a frame for phase
    phase_df = pd.DataFrame(
        {
            "throttle": thr,
            "brake": brk,
            "g_lat_abs": derived["g_lat_abs"],
            "g_long_signed": derived["g_long_signed"],
        },
        index=idx,
    )
    derived["phase"] = classify_phase(phase_df)
    derived["speed_band"] = pd.cut(
        derived["speed_kph"],
        bins=[-np.inf, V_LOW, V_HIGH, np.inf],
        labels=["low", "medium", "high"],
    )

    lap_num = work["lap_number"] if "lap_number" in work.columns else pd.Series(1.0, index=idx)
    valid = (
        derived["speed_kph"].notna()
        & (derived["speed_kph"] > 5)
        & lap_num.notna()
    )
    if "lap_number" in work.columns:
        valid = valid & (lap_num >= 0)
    if "lap_time_invalid" in work.columns:
        valid = valid & (work["lap_time_invalid"].fillna(0) != 1)
    if "pit_status" in work.columns:
        valid = valid & (work["pit_status"].fillna(0) <= 0)
    derived["valid_sample"] = valid

    # Drop keys that already exist as identical source cols we'll overwrite cleanly
    # Join derived in one shot, then single defragmenting copy
    out = pd.concat([df, pd.DataFrame(derived, index=idx)], axis=1)
    # If throttle/brake/steering existed, derived overwrote via concat duplicate names —
    # prefer last (derived):
    out = out.loc[:, ~out.columns.duplicated(keep="last")]
    return out.copy()


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


def _row_setup_value(row: pd.Series, field: str, alt_fields: Optional[list] = None) -> Optional[float]:
    """Read one setup channel; treat -1 / NaN as missing."""
    candidates = [field] + list(alt_fields or [])
    for f in candidates:
        if f not in row.index:
            continue
        v = row[f]
        if pd.isna(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv == SENTINEL:
            continue
        return fv
    return None


def extract_setup(df: pd.DataFrame) -> dict[str, Any]:
    """
    Read setup from a mid-session valid row.

    ARB / wing mapping (F1 game logger):
      wing_setup_0 = front wing, wing_setup_1 = rear wing
      arb_setup_0  = front ARB,  arb_setup_1  = rear ARB
    """
    setup_cols = [c for c in df.columns if "setup" in c or c == "front_brake_bias"]
    valid = df[df["valid_sample"]] if "valid_sample" in df.columns else df
    if valid.empty:
        valid = df
    row = valid.iloc[len(valid) // 2]

    raw: dict[str, Any] = {}
    for c in setup_cols:
        v = _row_setup_value(row, c)
        if v is not None:
            raw[c] = v

    # Explicit named values (never rely on fuzzy key search for ARB/wing)
    named = {
        "Front wing": _row_setup_value(row, "wing_setup_0"),
        "Rear wing": _row_setup_value(row, "wing_setup_1"),
        "Front ARB": _row_setup_value(row, "arb_setup_0"),
        "Rear ARB": _row_setup_value(row, "arb_setup_1"),
        "On-throttle differential": _row_setup_value(row, "diff_onThrottle_setup"),
        "Off-throttle differential": _row_setup_value(row, "diff_offThrottle_setup"),
        "Brake bias (% front)": _row_setup_value(
            row, "front_brake_bias", ["brake_bias_setup"]
        ),
        "Brake pressure": _row_setup_value(row, "brake_press_setup"),
        "Front tire pressure": _row_setup_value(
            row, "tyre_press_setup_0", ["tyre_press_setup_1"]
        ),
        "Rear tire pressure": _row_setup_value(
            row, "tyre_press_setup_2", ["tyre_press_setup_3"]
        ),
        "Front spring": _row_setup_value(row, "susp_spring_setup_0"),
        "Rear spring": _row_setup_value(row, "susp_spring_setup_2"),
        "Front camber": _row_setup_value(row, "camber_setup_0", ["camber_setup_1"]),
        "Rear camber": _row_setup_value(row, "camber_setup_2", ["camber_setup_3"]),
        "Front toe out": _row_setup_value(row, "toe_setup_0", ["toe_setup_1"]),
        "Rear toe in": _row_setup_value(row, "toe_setup_2", ["toe_setup_3"]),
        "Front ride height": _row_setup_value(row, "susp_height_setup_0"),
        "Rear ride height": _row_setup_value(row, "susp_height_setup_2"),
    }

    # Table for UI: parameter, telemetry field, current, min, max
    # Values shown with format_setup_value (label + percent for brake bias, etc.)
    table_rows = []
    for param, meta in SETUP_LIMITS.items():
        cur = named.get(param)
        cur_units = _to_limit_units(param, cur)
        lo, hi = float(meta["min"]), float(meta["max"])
        table_rows.append(
            {
                "Parameter": param,
                "Telemetry field": meta["field"],
                "Current": format_setup_value(param, cur_units),
                "Min": meta.get("min_label") or format_setup_value(param, lo),
                "Max": meta.get("max_label") or format_setup_value(param, hi),
                "At min": cur_units is not None and cur_units <= lo,
                "At max": cur_units is not None and cur_units >= hi,
                "Source": meta.get("source", "placeholder"),
            }
        )

    return {
        "raw": raw,
        "named": named,
        "table": table_rows,
        # back-compat for any older callers
        "_friendly": {
            "front_wing": named.get("Front wing"),
            "rear_wing": named.get("Rear wing"),
            "arb_front": named.get("Front ARB"),
            "arb_rear": named.get("Rear ARB"),
            "diff_on_throttle": named.get("On-throttle differential"),
            "diff_off_throttle": named.get("Off-throttle differential"),
            "brake_bias": named.get("Brake bias (% front)"),
        },
    }


def _to_limit_units(parameter: str, value: Optional[float]) -> Optional[float]:
    """Convert telemetry raw value into the units used by SETUP_LIMITS min/max."""
    if value is None:
        return None
    meta = SETUP_LIMITS.get(parameter) or {}
    v = float(value)
    if meta.get("telemetry_in_radians"):
        return v * (180.0 / np.pi)
    if meta.get("telemetry_in_pascals"):
        # 1 psi = 6894.757 Pa (matches F1 logger: ~201327 Pa ≈ 29.2 psi)
        return v / 6894.757
    return v


def setup_current(setup: dict[str, Any], parameter: str) -> Optional[float]:
    """Current setup value in the same units as SETUP_LIMITS min/max."""
    named = setup.get("named") or {}
    raw_val: Optional[float] = None
    if parameter in named and named[parameter] is not None:
        raw_val = float(named[parameter])
    else:
        meta = SETUP_LIMITS.get(parameter)
        if not meta:
            return None
        raw = setup.get("raw") or {}
        if meta["field"] in raw:
            raw_val = float(raw[meta["field"]])
        else:
            for alt in meta.get("alt_fields") or []:
                if alt in raw:
                    raw_val = float(raw[alt])
                    break
    return _to_limit_units(parameter, raw_val)


def feasibility(
    parameter: str, direction: str, setup: dict[str, Any]
) -> tuple[bool, str, Optional[float], Optional[float], Optional[float]]:
    """
    Return (feasible, blocked_reason, current, min, max).
    Block increase at max / decrease at min for known limited parameters.
    current/min/max are in limit units (e.g. degrees for camber, % for diffs).
    """
    meta = SETUP_LIMITS.get(parameter)
    cur = setup_current(setup, parameter)
    if meta is None:
        return True, "", cur, None, None
    lo, hi = float(meta["min"]), float(meta["max"])
    if cur is None:
        # Unknown current — allow suggestion but note uncertainty
        return True, "Current value not in telemetry; verify in-game before changing.", None, lo, hi
    if direction == "increase" and cur >= hi:
        extra = ""
        if meta.get("increase_means"):
            extra = f" (increase would mean: {meta['increase_means']})"
        return (
            False,
            f"Already at maximum ({format_setup_value(parameter, cur)}; max {meta.get('max_label') or format_setup_value(parameter, hi)}). Cannot increase.{extra}",
            cur,
            lo,
            hi,
        )
    if direction == "decrease" and cur <= lo:
        extra = ""
        if meta.get("decrease_means"):
            extra = f" (decrease would mean: {meta['decrease_means']})"
        return (
            False,
            f"Already at minimum ({format_setup_value(parameter, cur)}; min {meta.get('min_label') or format_setup_value(parameter, lo)}). Cannot decrease.{extra}",
            cur,
            lo,
            hi,
        )
    return True, "", cur, lo, hi


def format_setup_value(parameter: str, value: Optional[float]) -> str:
    """Format a value already in limit units (not raw telemetry radians)."""
    if value is None:
        return "unknown"
    meta = SETUP_LIMITS.get(parameter) or {}
    if meta.get("display_as_pct"):
        pct = value * 100.0
        # Brake bias: always "label + percent" (e.g. more forward 70%)
        if "Brake bias" in parameter:
            if pct >= 65:
                label = "more forward"
            elif pct <= 55:
                label = "more rearward"
            else:
                label = "mid-range"
            return f"{label} {pct:.0f}%"
        return f"{pct:.0f}%"
    if meta.get("telemetry_in_radians") or meta.get("unit", "").startswith("degrees"):
        return f"{value:.2f}°"
    if meta.get("telemetry_in_pascals") or "psi" in meta.get("unit", "").lower():
        return f"{value:.1f} psi"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _corner_bin(distance_m: float, bin_m: float = 50.0) -> float:
    if pd.isna(distance_m):
        return np.nan
    return float(bin_m * round(distance_m / bin_m))


def run_diagnostics(
    df: pd.DataFrame,
    us_alpha: float = US_ALPHA_THRESH,
    os_alpha: float = OS_ALPHA_THRESH,
    lock_slip: float = LOCK_SLIP,
    spin_slip: float = SPIN_SLIP,
) -> tuple[list[IssueEvent], list[IssueSummary]]:
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
        if bal > us_alpha:
            sev = min(2.0, bal / us_alpha)
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
        elif bal < -os_alpha:
            sev = min(2.0, abs(bal) / os_alpha)
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
        if bal > us_alpha * 1.1:
            add(
                "us_entry",
                "Entry understeer",
                idx,
                min(2.0, bal / us_alpha),
                f"Turn-in push α_bal={bal:.3f}, brake={row['brake']:.2f}",
            )
        elif bal < -os_alpha * 1.1:
            add(
                "os_entry",
                "Entry oversteer",
                idx,
                min(2.0, abs(bal) / os_alpha),
                f"Rear rotates on entry α_bal={bal:.3f}, brake={row['brake']:.2f}",
            )

    # Exit traction oversteer / spin
    exit_df = d[(d["phase"] == PHASE_EXIT) & (d["throttle"] > 0.4)]
    for idx, row in exit_df.iterrows():
        kr = row.get("kappa_r", np.nan)
        bal = row.get("alpha_balance", np.nan)
        if pd.notna(kr) and kr > spin_slip:
            add(
                "traction_spin",
                "Exit traction limitation (wheelspin)",
                idx,
                min(2.0, kr / spin_slip),
                f"κ_r={kr:.3f}, throttle={row['throttle']:.2f}, gear={row.get('gear', float('nan'))}",
            )
        if pd.notna(bal) and bal < -os_alpha and row["throttle"] > 0.5:
            add(
                "os_exit",
                "Exit oversteer",
                idx,
                min(2.0, abs(bal) / os_alpha),
                f"Power oversteer α_bal={bal:.3f}, T={row['throttle']:.2f}",
            )
        if pd.notna(bal) and bal > us_alpha and row["throttle"] > 0.55:
            add(
                "us_exit",
                "Exit understeer",
                idx,
                min(2.0, bal / us_alpha),
                f"Push on power α_bal={bal:.3f}, T={row['throttle']:.2f}",
            )

    # Lockups
    braking = d[(d["brake"] > 0.35) & (d["speed_kph"] > 40)]
    for idx, row in braking.iterrows():
        kf = row.get("kappa_f", np.nan)
        kr = row.get("kappa_r", np.nan)
        # Negative slip ratio under brake = lock tendency in many game exports
        if pd.notna(kf) and abs(kf) > lock_slip and kf < 0:
            add(
                "lock_front",
                "Front lockup",
                idx,
                min(2.0, abs(kf) / lock_slip),
                f"κ_f={kf:.3f}, brake={row['brake']:.2f}, v={row['speed_kph']:.0f}",
            )
        if pd.notna(kr) and abs(kr) > lock_slip and kr < 0:
            add(
                "lock_rear",
                "Rear lockup",
                idx,
                min(2.0, abs(kr) / lock_slip),
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

    # Criticality + tiers, then sort
    out = assign_criticality(out)
    out.sort(
        key=lambda s: (
            {"S": 0, "A": 1, "B": 2, "C": 3}.get(s.tier, 9),
            -s.criticality,
        ),
    )
    return out


# ---------------------------------------------------------------------------
# Criticality tiers, driver analysis, session grade
# ---------------------------------------------------------------------------

# Higher = more likely to cost lap time or safety if frequent
ISSUE_IMPACT_WEIGHT: dict[str, float] = {
    "lock_front": 1.35,
    "lock_rear": 1.35,
    "os_entry": 1.30,
    "os_high": 1.25,
    "os_exit": 1.20,
    "traction_spin": 1.15,
    "us_high": 1.15,
    "aero_us_hs": 1.10,
    "us_entry": 1.10,
    "us_low": 1.05,
    "us_mid_speed": 1.05,
    "us_exit": 1.00,
    "os_low": 1.05,
    "os_mid_speed": 1.05,
    "steer_corrections": 0.95,
    "tires_hot": 0.90,
    "tires_cold": 0.75,
    "tires_axle_imbalance": 0.80,
    "aero_os_or_mech": 0.95,
}

SAFETY_ISSUE_IDS = {
    "lock_front",
    "lock_rear",
    "os_entry",
    "os_high",
    "os_exit",
    "traction_spin",
}


def assign_criticality(summaries: list[IssueSummary]) -> list[IssueSummary]:
    """
    Attach criticality score and S/A/B/C tier.
    Criticality is a proxy (frequency × severity × confidence × issue weight),
    not measured tenths of a second.
    """
    for s in summaries:
        w = ISSUE_IMPACT_WEIGHT.get(s.issue_id, 1.0)
        s.criticality = (
            s.events_per_lap
            * max(s.mean_severity, 0.05)
            * max(s.confidence, 0.05)
            * w
            * (0.35 + s.lap_presence_pct / 100.0)
        )
        # Tier rules (presence + severity + category)
        if (
            s.issue_id in SAFETY_ISSUE_IDS
            and s.lap_presence_pct >= 20
            and s.mean_severity >= 0.85
        ) or (s.criticality >= 2.5 and s.lap_presence_pct >= 30):
            s.tier = "S"
        elif s.lap_presence_pct >= 45 and s.mean_severity >= 0.7:
            s.tier = "A"
        elif s.lap_presence_pct >= 25 or s.count >= 5:
            s.tier = "B"
        else:
            s.tier = "C"
        # Promote high-criticality mid-pack issues
        if s.criticality >= 1.8 and s.tier == "B":
            s.tier = "A"
        if s.criticality >= 3.0 and s.tier in ("A", "B"):
            s.tier = "S"
    return summaries


def extract_completed_lap_times(
    df: pd.DataFrame,
) -> list[tuple[float, float]]:
    """
    Return [(lap_number, lap_time_seconds), ...] for completed flying laps.

    Telemetry `lap_time` is elapsed time *within* the current lap (resets each lap).
    Finished lap time ≈ last sample's lap_time when the lap is complete.

    Important: many F1 exports are **distance-binned** (every lap has bins across
    the full trackLength even when the car only ran 16s). Distance coverage alone
    is NOT enough — short elapsed times must be rejected, and statistical outliers
    (e.g. 0:16 when the rest are 1:13s) are dropped.
    """
    if df is None or df.empty or "lap_number" not in df.columns or "lap_time" not in df.columns:
        return []

    work = df.copy()
    work = work[work["lap_number"].notna() & (work["lap_number"] >= 0)]
    work = work[work["lap_time"].notna() & (work["lap_time"] >= 0)]
    if work.empty:
        return []

    track_len = None
    if "trackLength" in work.columns and work["trackLength"].notna().any():
        tl = float(work["trackLength"].median())
        if tl > 1000:  # meters, sane circuit length
            track_len = tl

    # Hard floor: no modern F1-game flying lap is under this (Canada ~1:13, etc.)
    ABS_MIN_LAP_S = 55.0

    candidates: list[tuple[float, float, dict]] = []
    rejected: list[dict] = []

    for lap, sub in work.groupby("lap_number"):
        meta: dict[str, Any] = {"lap": float(lap)}
        if "lap_distance" in sub.columns:
            sub = sub.sort_values(["lap_distance", "lap_time"])
            max_dist = float(sub["lap_distance"].max())
        else:
            max_dist = None
        meta["max_dist"] = max_dist

        last_lt = float(sub["lap_time"].iloc[-1])
        max_lt = float(sub["lap_time"].max())
        # End-of-lap elapsed time; prefer last when sorted by distance
        lt = last_lt if last_lt >= max_lt * 0.98 else max_lt
        meta["lap_time"] = lt

        # Heavily invalid → skip
        if "lap_time_invalid" in sub.columns:
            tail = sub.tail(max(5, len(sub) // 10))
            if (tail["lap_time_invalid"].fillna(0) == 1).mean() > 0.5:
                meta["reason"] = "invalid flag on lap end"
                rejected.append(meta)
                continue

        if lt < ABS_MIN_LAP_S:
            meta["reason"] = f"too short ({lt:.3f}s < {ABS_MIN_LAP_S}s) — incomplete/out"
            rejected.append(meta)
            continue

        # Distance check only when we trust track length AND it looks like sparse
        # raw sampling (not full-bin dumps). Full-bin dumps often have max_dist≈track
        # even on partial time.
        n = len(sub)
        looks_full_bins = track_len is not None and n >= track_len * 0.8
        if track_len and max_dist is not None and not looks_full_bins:
            if max_dist < track_len * 0.90:
                meta["reason"] = f"distance only {max_dist:.0f}/{track_len:.0f} m"
                rejected.append(meta)
                continue

        candidates.append((float(lap), float(lt), meta))

    if not candidates:
        # stash rejections on function attribute for UI debug
        extract_completed_lap_times.last_rejected = rejected  # type: ignore[attr-defined]
        extract_completed_lap_times.last_raw = []  # type: ignore[attr-defined]
        return []

    times = [t for _, t, _ in candidates]
    med = float(np.median(times))
    extract_completed_lap_times.last_raw = [  # type: ignore[attr-defined]
        (ln, lt) for ln, lt, _ in candidates
    ]

    # Statistical gate: drop absurd outliers vs the session median
    # e.g. 16s when median is 74s, or a 3-minute lunch break lap
    out: list[tuple[float, float]] = []
    for ln, lt, meta in candidates:
        if med >= ABS_MIN_LAP_S and lt < med * 0.88:
            meta["reason"] = f"outlier fast vs median {med:.3f}s ({lt:.3f}s)"
            rejected.append(meta)
            continue
        if med >= ABS_MIN_LAP_S and lt > med * 1.20:
            meta["reason"] = f"outlier slow vs median {med:.3f}s ({lt:.3f}s)"
            rejected.append(meta)
            continue
        out.append((ln, lt))

    # If median filter wiped everything, fall back to candidates ≥ ABS_MIN only
    if not out and candidates:
        out = [(ln, lt) for ln, lt, _ in candidates if lt >= ABS_MIN_LAP_S]

    extract_completed_lap_times.last_rejected = rejected  # type: ignore[attr-defined]
    out.sort(key=lambda x: x[0])
    return out


def _lap_time_for_lap(d: pd.DataFrame, lap: float) -> Optional[float]:
    """Lookup one lap from completed-lap extraction (preferred)."""
    for ln, lt in extract_completed_lap_times(d):
        if ln == lap or abs(ln - lap) < 1e-6:
            return lt
    # Fallback: last/max on that lap only if complete enough
    sub = d[d["lap_number"] == lap]
    if sub.empty or "lap_time" not in sub.columns:
        return None
    if "lap_distance" in sub.columns:
        sub = sub.sort_values("lap_distance")
    lt = float(sub["lap_time"].iloc[-1]) if len(sub) else None
    if lt is None or lt < 40:
        return None
    return lt


def format_lap_time(seconds: float) -> str:
    """Format seconds as M:SS.mmm"""
    if seconds is None or seconds != seconds:
        return "—"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"


def analyze_driver(df: pd.DataFrame, bin_m: float = 25.0) -> dict[str, Any]:
    """
    Compare session to best valid lap: time-loss zones, brake/throttle, scrub.
    Returns structured notes for the UI.
    """
    d = df[df["valid_sample"]].copy() if "valid_sample" in df.columns else df.copy()
    empty = {
        "best_lap": None,
        "best_lap_time": None,
        "time_loss_zones": [],
        "notes": [],
        "brake_notes": [],
        "throttle_notes": [],
        "scrub_notes": [],
        "delta_series": None,
        "ref_speed": None,
        "cmp_speed": None,
        "lap_times": [],
    }
    # Lap times from full frame (not only valid_sample)
    lap_times = extract_completed_lap_times(df)
    empty["lap_times"] = lap_times

    if d.empty or "lap_distance" not in d.columns:
        empty["notes"] = ["Not enough valid samples for driver analysis."]
        return empty

    if not lap_times:
        empty["notes"] = [
            "No **completed** lap times found. "
            "(Need full-track coverage and lap_time ≥ 40s; incomplete last laps are ignored.)"
        ]
        return empty

    best_lap, best_t = min(lap_times, key=lambda x: x[1])
    ref = d[d["lap_number"] == best_lap].sort_values("lap_distance")
    if len(ref) < 20:
        # fall back to unfiltered rows for that lap
        ref = df[df["lap_number"] == best_lap].sort_values("lap_distance")
    if len(ref) < 20:
        empty["notes"] = ["Best lap has too few samples."]
        empty["best_lap"] = best_lap
        empty["best_lap_time"] = best_t
        return empty

    # Other completed laps for comparison (exclude best)
    others = [lap for lap, _ in lap_times if lap != best_lap]
    if not others:
        cmp = ref
        comparing = "best lap only (single completed lap)"
    else:
        cmp = d[d["lap_number"].isin(others)]
        if cmp.empty:
            cmp = df[df["lap_number"].isin(others)]
        comparing = f"other completed laps vs best L{best_lap:.0f} ({format_lap_time(best_t)})"

    def bin_profile(sub: pd.DataFrame) -> pd.DataFrame:
        s = sub.dropna(subset=["lap_distance", "speed_kph"]).copy()
        if s.empty:
            return pd.DataFrame()
        s["bin"] = (s["lap_distance"] / bin_m).round() * bin_m
        if "steer_abs" not in s.columns:
            s["steer_abs"] = s["steering"].abs() if "steering" in s.columns else 0.0
        if "g_lat_abs" not in s.columns:
            s["g_lat_abs"] = 0.0
        if "alpha_balance" not in s.columns:
            s["alpha_balance"] = 0.0
        for col in ("throttle", "brake"):
            if col not in s.columns:
                s[col] = 0.0
        g = s.groupby("bin", as_index=False).agg(
            speed_kph=("speed_kph", "mean"),
            throttle=("throttle", "mean"),
            brake=("brake", "mean"),
            steer_abs=("steer_abs", "mean"),
            g_lat_abs=("g_lat_abs", "mean"),
            alpha_balance=("alpha_balance", "mean"),
            n=("speed_kph", "count"),
        )
        return g.sort_values("bin")

    ref_p = bin_profile(ref)
    cmp_p = bin_profile(cmp if others else ref)
    if ref_p.empty:
        empty["notes"] = ["Could not build distance profile."]
        return empty

    # Merge on bin
    m = ref_p.merge(cmp_p, on="bin", suffixes=("_ref", "_cmp"), how="inner")
    if m.empty:
        empty["notes"] = ["No overlapping distance bins for comparison."]
        return empty

    # Approximate time loss (s) in bin: ds * (1/v_cmp - 1/v_ref), v in m/s
    ds = float(bin_m)
    v_ref = (m["speed_kph_ref"] / 3.6).clip(lower=1.0)
    v_cmp = (m["speed_kph_cmp"] / 3.6).clip(lower=1.0)
    m["time_loss_s"] = ds * (1.0 / v_cmp - 1.0 / v_ref)
    # Only positive = slower than reference
    m["time_loss_pos"] = m["time_loss_s"].clip(lower=0)

    zones = m.nlargest(8, "time_loss_pos")
    time_loss_zones = []
    for _, row in zones.iterrows():
        if row["time_loss_pos"] < 0.01:
            continue
        time_loss_zones.append(
            {
                "distance_m": float(row["bin"]),
                "time_loss_s": float(row["time_loss_pos"]),
                "speed_ref_kph": float(row["speed_kph_ref"]),
                "speed_cmp_kph": float(row["speed_kph_cmp"]),
                "speed_delta_kph": float(row["speed_kph_ref"] - row["speed_kph_cmp"]),
            }
        )

    notes = [
        f"Reference: best lap **L{best_lap:.0f}** ({best_t:.3f}s). Comparing: {comparing}.",
        "Time-loss is a **proxy** from speed difference by distance bin (not sector telemetry).",
    ]

    # Brake points: first brake > 0.25 on ref vs mean of others
    def first_brake_points(sub: pd.DataFrame) -> list[float]:
        pts = []
        for lap, g in sub.groupby("lap_number"):
            g = g.sort_values("lap_distance")
            hit = g[g["brake"] > 0.25]
            if not hit.empty:
                pts.append(float(hit.iloc[0]["lap_distance"]))
        return pts

    ref_bp = first_brake_points(ref)
    brake_notes = []
    if others:
        o_bp = first_brake_points(d[d["lap_number"].isin(others)])
        if ref_bp and o_bp:
            # Compare median first-brake distance — only meaningful as session scatter
            med_o = float(np.median(o_bp))
            med_r = float(np.median(ref_bp))
            # Also: variance of brake points (consistency)
            if len(o_bp) + len(ref_bp) >= 3:
                all_bp = o_bp + ref_bp
                std_bp = float(np.std(all_bp))
                if std_bp > 40:
                    brake_notes.append(
                        f"Brake points are inconsistent (σ ≈ {std_bp:.0f} m). "
                        "Pick a reference marker and repeat the same point."
                    )
            # Per-bin hard brake with low speed vs ref
            hard = m[(m["brake_cmp"] > 0.4) & (m["speed_kph_cmp"] < m["speed_kph_ref"] - 8)]
            if len(hard) >= 2:
                spots = ", ".join(f"{int(x)} m" for x in hard["bin"].head(4))
                brake_notes.append(
                    f"Heavier braking / lower speed than best lap near: {spots}."
                )
    if not brake_notes:
        brake_notes.append("No strong brake-point issues detected vs best lap.")

    # Throttle / exit
    throttle_notes = []
    if "throttle_cmp" in m.columns:
        exit_like = m[(m["throttle_cmp"] > 0.5) & (m["brake_cmp"] < 0.1)]
        slow_exit = exit_like[exit_like["speed_kph_cmp"] < exit_like["speed_kph_ref"] - 6]
        if len(slow_exit) >= 2:
            spots = ", ".join(f"{int(x)} m" for x in slow_exit.nlargest(4, "time_loss_pos")["bin"])
            throttle_notes.append(
                f"Exit / power zones slower than best lap near: {spots}. "
                "Try earlier progressive throttle or check traction OS flags."
            )
    if not throttle_notes:
        throttle_notes.append("No strong exit-speed loss pattern vs best lap.")

    # Scrub: high steer, low lat G relative, or high alpha, while slower than ref
    scrub_notes = []
    if "steer_abs_cmp" in m.columns and "g_lat_abs_cmp" in m.columns:
        scrub = m[
            (m["steer_abs_cmp"] > 0.25)
            & (m["g_lat_abs_cmp"] < 0.9)
            & (m["speed_kph_cmp"] < m["speed_kph_ref"] - 5)
        ]
        if len(scrub) >= 2:
            spots = ", ".join(f"{int(x)} m" for x in scrub.nlargest(4, "time_loss_pos")["bin"])
            scrub_notes.append(
                f"Possible mid-corner scrub (lots of lock, less lateral load, lower speed) near: {spots}."
            )
    if "alpha_balance_cmp" in m.columns:
        push = m[
            (m["alpha_balance_cmp"] > 0.04)
            & (m["time_loss_pos"] > 0.02)
        ]
        if len(push) >= 2:
            spots = ", ".join(f"{int(x)} m" for x in push.nlargest(3, "time_loss_pos")["bin"])
            scrub_notes.append(
                f"Understeer-ish slip with time loss near: {spots} — may be car or early throttle / wrong line."
            )
    if not scrub_notes:
        scrub_notes.append("No clear scrub signature vs best lap.")

    # Driver vs car hint from top zone
    if time_loss_zones:
        top = time_loss_zones[0]
        notes.append(
            f"Largest proxy time loss: **+{top['time_loss_s']:.3f}s** around **{top['distance_m']:.0f} m** "
            f"({top['speed_cmp_kph']:.0f} vs {top['speed_ref_kph']:.0f} kph on best)."
        )

    return {
        "best_lap": best_lap,
        "best_lap_time": best_t,
        "time_loss_zones": time_loss_zones,
        "notes": notes,
        "brake_notes": brake_notes,
        "throttle_notes": throttle_notes,
        "scrub_notes": scrub_notes,
        "delta_series": m,
        "comparing": comparing,
        "lap_times": lap_times,
    }


def session_grade(
    df: pd.DataFrame,
    summaries: list[IssueSummary],
    driver: dict[str, Any],
) -> dict[str, Any]:
    """
    0–100 composite grade with component breakdown.

    Pace + consistency dominate. Diagnostic noise (US/OS/lock spam) can only
    shave a little — a tight 5-lap run like 1:13.5–1:14.2 must score well.
    """
    d = df[df["valid_sample"]].copy() if "valid_sample" in df.columns else df.copy()
    components: dict[str, dict[str, Any]] = {}

    # Completed lap times only (correct extraction — not raw max on partials)
    completed = extract_completed_lap_times(df)
    raw_laps = [lt for _, lt in completed]

    # Use push laps only: within 3% of best (drops in-laps / cool-downs still timed full)
    laps = list(raw_laps)
    if len(raw_laps) >= 2:
        best0 = min(raw_laps)
        push = [t for t in raw_laps if t <= best0 * 1.03]
        if len(push) >= 2:
            laps = push

    # --- Pace (45%): how close the set is to the best lap ---
    if len(laps) >= 2:
        best = min(laps)
        mean_t = float(np.mean(laps))
        # Mean gap in seconds (more intuitive than % for ~70s laps)
        mean_gap = mean_t - best
        # 0.0s gap → 100; 0.5s → ~90; 1.0s → ~80; 2.0s → ~60
        pace_score = float(np.clip(100 - mean_gap * 20, 40, 100))
        components["Pace"] = {
            "score": pace_score,
            "weight": 0.45,
            "detail": (
                f"Best {best:.3f}s · set mean {mean_t:.3f}s "
                f"(+{mean_gap:.3f}s) · {len(laps)} push laps"
            ),
        }
    elif len(laps) == 1:
        components["Pace"] = {
            "score": 80.0,
            "weight": 0.45,
            "detail": f"Single timed lap {laps[0]:.3f}s — no set to compare.",
        }
    else:
        components["Pace"] = {
            "score": 60.0,
            "weight": 0.45,
            "detail": "No usable lap times — pace unscored.",
        }

    # --- Consistency (35%): spread of the push set ---
    if len(laps) >= 3:
        std = float(np.std(laps))
        spread = float(max(laps) - min(laps))
        # σ 0.25s → ~95; 0.5s → ~88; 1.0s → ~75; 2.0s → ~55
        cons = float(np.clip(100 - std * 25, 45, 100))
        # reward tight total window
        if spread <= 0.5:
            cons = max(cons, 92.0)
        elif spread <= 1.0:
            cons = max(cons, 85.0)
        components["Consistency"] = {
            "score": cons,
            "weight": 0.35,
            "detail": f"σ ≈ {std:.3f}s · spread {spread:.3f}s over {len(laps)} laps",
        }
    elif len(laps) == 2:
        gap = abs(laps[0] - laps[1])
        cons = float(np.clip(100 - gap * 30, 50, 100))
        if gap <= 0.4:
            cons = max(cons, 90.0)
        components["Consistency"] = {
            "score": cons,
            "weight": 0.35,
            "detail": f"Two laps, gap {gap:.3f}s",
        }
    else:
        components["Consistency"] = {
            "score": 75.0,
            "weight": 0.35,
            "detail": "Need 2+ push laps for consistency.",
        }

    # --- Cleanliness (10%): soft penalty only ---
    clean_ids = {"lock_front", "lock_rear", "traction_spin", "os_entry"}
    epl = sum(s.events_per_lap for s in summaries if s.issue_id in clean_ids)
    # 0 → 100; 2/lap → ~90; 5/lap → ~80; never below 60 from this alone
    clean_score = float(np.clip(100 - epl * 4, 60, 100))
    components["Cleanliness"] = {
        "score": clean_score,
        "weight": 0.10,
        "detail": f"Lock/spin/entry-OS events per lap ≈ {epl:.2f} (soft penalty)",
    }

    # --- Balance control (5%): soft ---
    bal_ids = [
        i for i in summaries if i.issue_id.startswith("us_") or i.issue_id.startswith("os_")
    ]
    bal_epl = sum(s.events_per_lap for s in bal_ids)
    bal_score = float(np.clip(100 - bal_epl * 2, 65, 100))
    components["Balance control"] = {
        "score": bal_score,
        "weight": 0.05,
        "detail": f"US/OS events per lap ≈ {bal_epl:.2f} (soft penalty)",
    }

    # --- Tire window (5%): soft ---
    tire = [s for s in summaries if s.issue_id in ("tires_cold", "tires_hot")]
    tire_epl = sum(s.events_per_lap for s in tire)
    tire_score = float(np.clip(100 - tire_epl * 3, 70, 100))
    components["Tires"] = {
        "score": tire_score,
        "weight": 0.05,
        "detail": f"Out-of-window tire events/lap ≈ {tire_epl:.2f} (soft penalty)",
    }

    total = sum(c["score"] * c["weight"] for c in components.values())

    # Floors from pure lap-time quality (so a clean 1:13.5–1:14.2 set cannot be an F)
    if len(laps) >= 3:
        best = min(laps)
        spread = max(laps) - min(laps)
        if spread <= 0.5:
            total = max(total, 90.0)
        elif spread <= 0.8:
            total = max(total, 85.0)
        elif spread <= 1.2:
            total = max(total, 78.0)
        # All laps within 1.0s of best
        if max(laps) <= best + 1.0:
            total = max(total, 82.0)
        if max(laps) <= best + 0.6:
            total = max(total, 88.0)

    total = float(np.clip(total, 0, 100))

    letter = (
        "A" if total >= 90
        else "B" if total >= 80
        else "C" if total >= 70
        else "D" if total >= 60
        else "F"
    )
    return {
        "score": round(total, 1),
        "letter": letter,
        "components": components,
        "disclaimer": (
            "Session-relative grade (push laps within 3% of best). "
            "Pace + consistency dominate; car flags only soft-penalize. "
            "Not a world ranking."
        ),
    }


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
            "weight": 1.0,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Stiffer rear can rotate the car mid-corner at low speed (use if OS not already present).",
            "validation_metric": "No rise in os_low frequency",
            "weight": 0.85,
        },
        {
            "parameter": "Off-throttle differential",
            "direction": "decrease",
            "amount_hint": "−5% to −10%",
            "reason": "Less coast locking can free rotation into/through slow corners.",
            "validation_metric": "us_low / us_entry frequency",
            "weight": 0.65,
        },
        {
            "parameter": "Front spring",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Softer front spring can increase front mechanical grip mid-corner.",
            "validation_metric": "us_low events per lap",
            "weight": 0.55,
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
            "weight": 1.0,
        },
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Mechanical front grip if front wing is already maxed or you want less drag.",
            "validation_metric": "us_high frequency",
            "weight": 0.85,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Stiffer rear can free rotation in fast corners (watch for OS).",
            "validation_metric": "No rise in os_high",
            "weight": 0.55,
        },
        {
            "parameter": "Front ride height",
            "direction": "decrease",
            "amount_hint": "−1 if not bottoming",
            "reason": "Lower front can add front aero load (game-dependent).",
            "validation_metric": "us_high + floor not scraping",
            "weight": 0.45,
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
            "weight": 0.9,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Rotate mid-corner without adding front wing (if FW at limit).",
            "validation_metric": "us_mid_speed; watch os_mid",
            "weight": 0.7,
        },
        {
            "parameter": "Front spring",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Softer front spring can put more load on front tires mid-corner.",
            "validation_metric": "us_mid_speed frequency",
            "weight": 0.55,
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
            "parameter": "Brake bias (% front)",
            "direction": "increase",
            "amount_hint": "more forward +1% to +2% (toward more forward 70%)",
            "reason": "Entry oversteer — move toward more forward 70% (higher % front) to stabilize the rear on turn-in.",
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
            "parameter": "Brake bias (% front)",
            "direction": "decrease",
            "amount_hint": "more rearward −1% to −2% (toward more rearward 50%)",
            "reason": "Entry understeer — move toward more rearward 50% (lower % front) to help rotation on trail-brake.",
            "validation_metric": "us_entry count; watch rear lockups",
            "weight": 1.0,
        },
        {
            "parameter": "Front wing",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "More front aero turn-in bite (skip if already at max).",
            "validation_metric": "us_entry frequency",
            "weight": 0.7,
        },
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1",
            "reason": "Softer front ARB helps turn-in if wing cannot go up.",
            "validation_metric": "us_entry frequency",
            "weight": 0.85,
        },
        {
            "parameter": "Off-throttle differential",
            "direction": "decrease",
            "amount_hint": "−5% to −10%",
            "reason": "Less coast lock can free the car on entry.",
            "validation_metric": "us_entry events",
            "weight": 0.65,
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
            "parameter": "Brake bias (% front)",
            "direction": "decrease",
            "amount_hint": "more rearward −1% to −2% (toward more rearward 50%)",
            "reason": "Front lockups — move toward more rearward 50% (lower % front) to unload the fronts under braking.",
            "validation_metric": "lock_front event count",
            "weight": 1.2,
        },
        {
            "parameter": "Brake pressure",
            "direction": "decrease",
            "amount_hint": "−1% to −2% if available",
            "reason": "Softer initial bite if still locking after bias change.",
            "validation_metric": "lock_front severity",
            "weight": 0.8,
        },
        {
            "parameter": "Driving / out-lap",
            "direction": "adjust",
            "amount_hint": "Slightly earlier brake / less initial spike",
            "reason": "Driver input often fixes lock without setup change.",
            "validation_metric": "lock_front frequency",
            "weight": 0.45,
        },
    ],
    "lock_rear": [
        {
            "parameter": "Brake bias (% front)",
            "direction": "increase",
            "amount_hint": "more forward +1% to +2% (toward more forward 70%)",
            "reason": "Rear lockups — move toward more forward 70% (higher % front) to unload the rears under braking.",
            "validation_metric": "lock_rear count",
            "weight": 1.2,
        },
        {
            "parameter": "Brake pressure",
            "direction": "decrease",
            "amount_hint": "−1% to −2%",
            "reason": "Softer overall bite if bias alone does not stop rear lock.",
            "validation_metric": "lock_rear count",
            "weight": 0.7,
        },
        {
            "parameter": "Driving / out-lap",
            "direction": "adjust",
            "amount_hint": "Smoother initial brake pressure",
            "reason": "Technique can eliminate rear lock without setup change.",
            "validation_metric": "lock_rear frequency",
            "weight": 0.4,
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
        {
            "parameter": "On-throttle differential",
            "direction": "decrease",
            "amount_hint": "−5% to −10%",
            "reason": "If corrections happen on power, open the diff.",
            "validation_metric": "steer_corrections on exit",
            "weight": 0.55,
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
        {
            "parameter": "Rear wing",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Alternative when front wing is already at max: reduce rear aero.",
            "validation_metric": "U_high vs U_low; trap speed",
            "weight": 1.05,
        },
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1 to −2",
            "reason": "Mechanical path if wings are at limits.",
            "validation_metric": "aero_us_hs / us_high frequency",
            "weight": 0.85,
        },
        {
            "parameter": "Rear ARB",
            "direction": "increase",
            "amount_hint": "+1",
            "reason": "Rotate with rear mechanical, not more front wing.",
            "validation_metric": "us_high; watch os_high",
            "weight": 0.7,
        },
    ],
    "tires_cold": [
        {
            "parameter": "Driving / out-lap",
            "direction": "adjust",
            "amount_hint": "Warm-up: weave, later brake, build load",
            "reason": "Tires below window — setup changes are secondary until temps rise.",
            "validation_metric": "tyre_temp_* into window before push laps",
            "weight": 0.9,
        },
        {
            "parameter": "Front wing",
            "direction": "adjust",
            "amount_hint": "No big aero cuts while cold",
            "reason": "Do not strip wing to chase cold-tire understeer.",
            "validation_metric": "Temps first, then re-check balance",
            "weight": 0.3,
        },
    ],
    "tires_hot": [
        {
            "parameter": "Driving / out-lap",
            "direction": "adjust",
            "amount_hint": "Reduce sliding / scrubbing",
            "reason": "Over-temp often from sustained slide — fix balance issues first.",
            "validation_metric": "Peak tyre_temp and slide issue frequency",
            "weight": 0.8,
        },
        {
            "parameter": "Front ARB",
            "direction": "adjust",
            "amount_hint": "See hotter axle (front vs rear)",
            "reason": "Hot fronts → reduce front slide (often softer front ARB or less US). Hot rears → calm rear.",
            "validation_metric": "T_f vs T_r",
            "weight": 0.5,
        },
    ],
    "tires_axle_imbalance": [
        {
            "parameter": "Front ARB",
            "direction": "decrease",
            "amount_hint": "−1 if fronts much hotter",
            "reason": "Front much hotter often means front sliding (understeer).",
            "validation_metric": "|T_f − T_r| mid-stint",
            "weight": 0.6,
        },
        {
            "parameter": "Rear ARB",
            "direction": "decrease",
            "amount_hint": "−1 if rears much hotter",
            "reason": "Rear much hotter often means rear sliding (oversteer).",
            "validation_metric": "|T_f − T_r| mid-stint",
            "weight": 0.6,
        },
        {
            "parameter": "Front wing",
            "direction": "adjust",
            "amount_hint": "Only if not at limit; match hotter axle",
            "reason": "Aero balance is an alternate lever when ARB is already extreme.",
            "validation_metric": "Tire axle delta + balance issues",
            "weight": 0.45,
        },
    ],
}


def recommend_setup(
    summaries: list[IssueSummary], setup: dict[str, Any]
) -> list[SetupChange]:
    """
    Build setup advice:
    - Multiple alternative levers per issue (never a single forced path)
    - Respect SETUP_LIMITS (no +wing at max, etc.)
    - Infeasible options kept but marked blocked so you see why they were skipped
    """
    all_changes: list[SetupChange] = []

    for s in summaries:
        templates = ISSUE_TO_CHANGES.get(s.issue_id, [])
        if not templates:
            continue
        freq_factor = min(2.0, 0.5 + s.events_per_lap + 0.01 * s.lap_presence_pct)
        issue_opts: list[SetupChange] = []

        for t in templates:
            pr = t["weight"] * s.mean_severity * s.confidence * freq_factor
            ok, blocked, cur, lo, hi = feasibility(t["parameter"], t["direction"], setup)
            cur_txt = format_setup_value(t["parameter"], cur)
            meta = SETUP_LIMITS.get(t["parameter"]) or {}
            range_txt = ""
            if lo is not None and hi is not None:
                lo_l = meta.get("min_label") or format_setup_value(t["parameter"], lo)
                hi_l = meta.get("max_label") or format_setup_value(t["parameter"], hi)
                range_txt = f" [range {lo_l} … {hi_l}]"
            dir_txt = ""
            if t["direction"] == "increase" and meta.get("increase_means"):
                dir_txt = f" → {meta['increase_means']}"
            elif t["direction"] == "decrease" and meta.get("decrease_means"):
                dir_txt = f" → {meta['decrease_means']}"
            amount = t["amount_hint"]
            if cur is not None:
                amount = f"{amount} (current {cur_txt}{range_txt}){dir_txt}"
            else:
                amount = f"{amount}{range_txt}{dir_txt}"

            issue_opts.append(
                SetupChange(
                    parameter=t["parameter"],
                    direction=t["direction"],
                    amount_hint=amount,
                    reason=t["reason"]
                    + f" [seen {s.count}×, {s.lap_presence_pct:.0f}% of laps]",
                    linked_issues=[s.name],
                    priority=pr if ok else pr * 0.01,
                    validation_metric=t["validation_metric"],
                    current=cur,
                    min_v=lo,
                    max_v=hi,
                    feasible=ok,
                    blocked_reason=blocked,
                    issue_id=s.issue_id,
                )
            )

        # Prefer feasible options first, then by priority
        issue_opts.sort(key=lambda c: (not c.feasible, -c.priority))
        labels = "ABCDEFGH"
        for i, c in enumerate(issue_opts):
            c.option_label = f"Option {labels[i] if i < len(labels) else i + 1}"
            all_changes.append(c)

    return all_changes


def recommendations_by_issue(
    changes: list[SetupChange],
) -> dict[str, list[SetupChange]]:
    by: dict[str, list[SetupChange]] = {}
    for c in changes:
        key = c.linked_issues[0] if c.linked_issues else c.issue_id
        by.setdefault(key, []).append(c)
    return by


def _clamp_setup_value(parameter: str, value: float) -> float:
    meta = SETUP_LIMITS.get(parameter) or {}
    lo = meta.get("min")
    hi = meta.get("max")
    v = float(value)
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    # Snap near-integers for click params
    step = meta.get("step", 1)
    try:
        step = float(step)
    except (TypeError, ValueError):
        step = 1.0
    if step >= 1 and not meta.get("display_as_pct") and not meta.get("telemetry_in_radians"):
        v = round(v)
    elif meta.get("display_as_pct") and step >= 0.01:
        # snap to 0.01
        v = round(v / step) * step
    return v


def _n_steps_for_mode(mode: str, severity: float, weight: float) -> int:
    """How many setup increments to apply for one lever."""
    if mode == "aggressive":
        if severity >= 1.3 or weight >= 1.15:
            return 2
        return 1
    # conservative: always 1 click / one step
    return 1


def build_suggested_setup(
    setup: dict[str, Any],
    summaries: list[IssueSummary],
    mode: str = "conservative",
) -> dict[str, Any]:
    """
    Build one full suggested setup from current values + non-conflicting deltas.

    mode:
      conservative — Tier S/A only, max 3 knobs, 1 step each
      aggressive   — Tier S/A/B, max 6 knobs, 1–2 steps by severity
    """
    mode = (mode or "conservative").lower()
    if mode not in ("conservative", "aggressive"):
        mode = "conservative"

    max_changes = 3 if mode == "conservative" else 6
    allowed_tiers = {"S", "A"} if mode == "conservative" else {"S", "A", "B"}

    # Baseline: all known setup params from telemetry (limit units)
    current: dict[str, Optional[float]] = {}
    for param in SETUP_LIMITS:
        current[param] = setup_current(setup, param)

    suggested = {k: v for k, v in current.items()}
    applied: list[dict[str, Any]] = []
    skipped: list[str] = []

    # Issues in criticality order
    issues = [
        s
        for s in summaries
        if s.tier in allowed_tiers and s.issue_id in ISSUE_TO_CHANGES
    ]
    issues.sort(key=lambda s: (-s.criticality, s.tier))

    used_params: set[str] = set()
    # Track family conflicts lightly (don't both raise FW and cut FW)
    param_dir: dict[str, str] = {}

    for s in issues:
        if len(applied) >= max_changes:
            break
        templates = list(ISSUE_TO_CHANGES.get(s.issue_id, []))
        # highest weight first
        templates.sort(key=lambda t: -t.get("weight", 0))

        picked = None
        for t in templates:
            param = t["parameter"]
            direction = t["direction"]
            if direction == "adjust":
                continue  # not a numeric setup click
            if param not in SETUP_LIMITS:
                continue
            if param in used_params:
                continue
            if param in param_dir and param_dir[param] != direction:
                continue
            ok, blocked, cur, lo, hi = feasibility(param, direction, setup)
            if not ok or cur is None:
                continue
            # Would a step actually move the value?
            n_steps = _n_steps_for_mode(mode, s.mean_severity, t.get("weight", 1.0))
            step = float(SETUP_LIMITS[param].get("step", 1))
            delta = n_steps * step * (1 if direction == "increase" else -1)
            new_val = _clamp_setup_value(param, cur + delta)
            if abs(new_val - cur) < step * 0.25:
                continue  # clamped to same value
            picked = {
                "parameter": param,
                "direction": direction,
                "from": cur,
                "to": new_val,
                "steps": n_steps,
                "issue": s.name,
                "issue_id": s.issue_id,
                "tier": s.tier,
                "reason": t["reason"],
                "weight": t.get("weight", 1.0),
            }
            break

        if not picked:
            skipped.append(f"{s.name}: no feasible numeric lever left")
            continue

        param = picked["parameter"]
        suggested[param] = picked["to"]
        used_params.add(param)
        param_dir[param] = picked["direction"]
        applied.append(picked)

    # Full table rows (all params we know)
    rows = []
    for param in SETUP_LIMITS:
        cur = current.get(param)
        sug = suggested.get(param)
        if cur is None and sug is None:
            continue
        changed = (
            cur is not None
            and sug is not None
            and abs(float(sug) - float(cur)) > 1e-9
        )
        delta_str = "—"
        if changed:
            dlt = float(sug) - float(cur)
            # pretty delta
            meta = SETUP_LIMITS[param]
            if meta.get("display_as_pct"):
                delta_str = f"{dlt * 100:+.1f}%"
            elif meta.get("telemetry_in_radians") or "degree" in meta.get("unit", ""):
                delta_str = f"{dlt:+.2f}°"
            elif meta.get("telemetry_in_pascals") or "psi" in meta.get("unit", ""):
                delta_str = f"{dlt:+.1f} psi"
            else:
                delta_str = f"{dlt:+g}"
        rows.append(
            {
                "Parameter": param,
                "Current": format_setup_value(param, cur),
                "Suggested": format_setup_value(param, sug),
                "Delta": delta_str if changed else "—",
                "Changed": "yes" if changed else "",
            }
        )

    return {
        "mode": mode,
        "current": current,
        "suggested": suggested,
        "applied": applied,
        "skipped": skipped,
        "rows": rows,
        "max_changes": max_changes,
        "tiers_used": sorted(allowed_tiers),
    }


# ---------------------------------------------------------------------------
# Session stats helpers
# ---------------------------------------------------------------------------


def session_overview(df: pd.DataFrame) -> dict[str, Any]:
    d = df[df["valid_sample"]] if "valid_sample" in df.columns else df
    lap_times = extract_completed_lap_times(df)
    laps = [ln for ln, _ in lap_times]
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
        st.caption(
            "Higher α/κ = fewer flags (only worse events). "
            "If the issue list is huge, raise thresholds or presence; "
            "if empty but the car felt bad, lower them slightly."
        )
        us_alpha = st.slider("Understeer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        st.caption("Default 0.03. Try 0.04–0.05 if everything is US; 0.02 if you feel push but see none.")
        os_alpha = st.slider("Oversteer α threshold (rad)", 0.01, 0.10, 0.03, 0.005)
        st.caption("Default 0.03. Raise if OS spam; lower if the car feels loose with no flags.")
        lock_slip = st.slider("Lockup |κ| threshold", 0.03, 0.25, 0.08, 0.01)
        st.caption("Default 0.08. Lower (0.05) if locks never show; raise if noisy.")
        spin_slip = st.slider("Wheelspin κ threshold", 0.05, 0.35, 0.12, 0.01)
        st.caption("Default 0.12. Lower if exits feel greasy with no traction flags.")
        min_count = st.number_input("Min event count to show issue", 1, 50, 2)
        st.caption("Hide rare one-offs. Use 2–3 normally; 4–5 for long noisy sessions.")
        min_presence = st.slider("Min lap presence %", 0, 100, 15)
        st.caption("% of laps that must show the issue. 15–25 default; 30–40 for “every lap” only.")
        st.markdown("---")
        st.subheader("Suggested setup")
        setup_mode = st.radio(
            "Build mode",
            ["Conservative", "Aggressive"],
            index=0,
            help=(
                "Conservative: Tier S/A only, max 3 knobs, 1 step each. "
                "Aggressive: Tier S/A/B, max 6 knobs, up to 2 steps on strong issues."
            ),
        )
        st.caption(
            "Starts from **your current setup**, applies non-conflicting clicks only, "
            "clamped to confirmed min/max. Toggle freely — more options is the point."
        )
        st.markdown("---")
        st.caption(
            "Wheel map: 0=FL 1=FR 2=RL 3=RR · Sentinel −1 → missing · "
            "Phases: entry / mid / exit from brake, throttle, G. "
            "Criticality tiers S→C and the 0–100 grade are session-relative proxies."
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
- **Criticality tiers** S / A / B / C (fix-first order)
- **Driver coach:** time-loss zones vs best lap, brake/exit/scrub notes
- **Session grade** 0–100 with component breakdown
- **Suggested setup** (Conservative / Aggressive toggle)
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
        events, summaries = run_diagnostics(
            df,
            us_alpha=us_alpha,
            os_alpha=os_alpha,
            lock_slip=lock_slip,
            spin_slip=spin_slip,
        )

        # Filter for display
        shown = [
            s
            for s in summaries
            if s.count >= min_count and s.lap_presence_pct >= min_presence
        ]
        if not shown:
            shown = summaries[:15]
        driver = analyze_driver(df)
        grade = session_grade(df, summaries, driver)
        changes = recommend_setup(shown if shown else summaries[:5], setup)
        suggested = build_suggested_setup(
            setup,
            shown if shown else summaries,
            mode=setup_mode.lower(),
        )

    # ----- Overview -----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Track", str(overview["track"]))
    c2.metric("Completed laps", f"{len(overview['laps'])}")
    if overview["best_lap"]:
        c3.metric(
            "Best lap",
            f"L{overview['best_lap'][0]:.0f}  {format_lap_time(overview['best_lap'][1])}",
        )
    else:
        c3.metric("Best lap", "—")
    c4.metric("Vmax", f"{overview['vmax_kph']:.0f} kph")
    c5.metric("Samples", f"{overview['samples']:,}")
    c6.metric("Session grade", f"{grade['score']:.0f}/100 ({grade['letter']})")

    # Always show how lap times were read (debug the #1 trust issue)
    if overview.get("lap_times"):
        st.markdown("**Completed lap times** (used for best lap / grade / coach)")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Lap": int(ln),
                        "Time": format_lap_time(lt),
                        "Seconds": round(lt, 3),
                    }
                    for ln, lt in overview["lap_times"]
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "`lap_time` = elapsed time within the lap (resets each lap). "
            "Kept only if ≥ **55s** and not an outlier vs session median "
            "(drops incomplete gems like a **0:16** partial). "
            "Distance-binned exports can show full track bins even on short runs — time wins."
        )
    else:
        st.warning(
            "No completed lap times detected (need ≥55s flying laps that aren't outliers)."
        )

    rejected = getattr(extract_completed_lap_times, "last_rejected", None) or []
    if rejected:
        with st.expander(f"Ignored laps ({len(rejected)}) — why they didn't count"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Lap": int(r.get("lap", -1)),
                            "Time": format_lap_time(float(r["lap_time"]))
                            if r.get("lap_time") is not None
                            else "—",
                            "Reason": r.get("reason", ""),
                        }
                        for r in rejected
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    # ----- Session grade -----
    st.subheader("Session grade")
    st.caption(grade["disclaimer"])
    gcols = st.columns(len(grade["components"]))
    for col, (name, comp) in zip(gcols, grade["components"].items()):
        col.metric(name, f"{comp['score']:.0f}", help=comp["detail"])
        col.caption(comp["detail"])

    # ----- Fix first (criticality tiers) -----
    st.subheader("Fix first (criticality tiers)")
    st.caption(
        "**S** = safety / high impact · **A** = consistent pace killers · "
        "**B** = noticeable · **C** = noise / one-offs. "
        "Criticality = frequency × severity × confidence × issue weight (proxy, not tenths)."
    )
    tier_shown = [s for s in shown if s.tier in ("S", "A", "B", "C")]
    if tier_shown:
        fix_rows = []
        for s in tier_shown:
            fix_rows.append(
                {
                    "Tier": s.tier,
                    "Issue": s.name,
                    "Criticality": round(s.criticality, 2),
                    "Presence %": round(s.lap_presence_pct, 1),
                    "Per lap": round(s.events_per_lap, 2),
                    "Severity": round(s.mean_severity, 2),
                    "Hot spots (m)": ", ".join(f"{h:.0f}" for h in s.hot_corners_m[:3]),
                }
            )
        fix_df = pd.DataFrame(fix_rows)
        st.dataframe(fix_df, use_container_width=True, hide_index=True)
        s_issues = [s for s in tier_shown if s.tier == "S"]
        a_issues = [s for s in tier_shown if s.tier == "A"]
        if s_issues:
            st.error(
                "**Tier S — fix first:** "
                + "; ".join(f"{s.name} (crit {s.criticality:.2f})" for s in s_issues[:5])
            )
        elif a_issues:
            st.warning(
                "**Tier A — next focus:** "
                + "; ".join(f"{s.name} (crit {s.criticality:.2f})" for s in a_issues[:5])
            )
        else:
            st.success("No S/A tier issues after filters — session looks relatively clean.")
    else:
        st.info("No issues to tier yet.")

    st.subheader("Current setup (from telemetry)")
    setup_table = setup.get("table") or []
    if setup_table:
        st.dataframe(
            pd.DataFrame(setup_table),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "ARB: `arb_setup_0` = Front ARB, `arb_setup_1` = Rear ARB. "
            "Wings: `wing_setup_0` = Front (0–50), `wing_setup_1` = Rear (0–50). "
            "Brake bias shows **label + percent**: "
            "**more forward 70%** … **more rearward 50%** (% is front share). "
            "If a Current value disagrees with the in-game menu, tell me both numbers."
        )
        # Highlight ARB + wings explicitly so they are never buried
        n = setup.get("named") or {}
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Front wing", format_setup_value("Front wing", n.get("Front wing")))
        a2.metric("Rear wing", format_setup_value("Rear wing", n.get("Rear wing")))
        a3.metric("Front ARB", format_setup_value("Front ARB", n.get("Front ARB")))
        a4.metric("Rear ARB", format_setup_value("Rear ARB", n.get("Rear ARB")))
    else:
        st.warning("No setup channels found in this file.")

    # ----- Suggested full setup (Conservative / Aggressive) -----
    st.subheader(f"Suggested setup ({suggested['mode'].title()})")
    st.caption(
        f"From your current car · tiers {', '.join(suggested['tiers_used'])} · "
        f"max {suggested['max_changes']} knobs · no conflicting directions · "
        "clamped to confirmed limits. Not a meta guarantee — a structured starting point."
    )
    if suggested.get("applied"):
        st.markdown("**Changes applied to build this suggestion**")
        for a in suggested["applied"]:
            st.markdown(
                f"- **{a['parameter']}**: {format_setup_value(a['parameter'], a['from'])} → "
                f"**{format_setup_value(a['parameter'], a['to'])}** "
                f"(`{a['direction']}` ×{a['steps']}) — *[{a['tier']}] {a['issue']}*: {a['reason']}"
            )
    else:
        st.info(
            "No numeric setup clicks applied (no feasible levers, or session looks clean). "
            "Try **Aggressive**, lower issue filters, or check Tier S/A issues above."
        )

    if suggested.get("rows"):
        # Highlight changed rows first in a compact "delta only" view
        changed_rows = [r for r in suggested["rows"] if r.get("Changed")]
        if changed_rows:
            st.markdown("**Delta only**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Parameter": r["Parameter"],
                            "Current": r["Current"],
                            "Suggested": r["Suggested"],
                            "Delta": r["Delta"],
                        }
                        for r in changed_rows
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Full setup card (current vs suggested)"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Parameter": r["Parameter"],
                            "Current": r["Current"],
                            "Suggested": r["Suggested"],
                            "Delta": r["Delta"],
                        }
                        for r in suggested["rows"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
    if suggested.get("skipped"):
        with st.expander(f"Skipped issues ({len(suggested['skipped'])})"):
            for line in suggested["skipped"]:
                st.write(f"- {line}")

    # ----- Driver coach -----
    st.subheader("Driver coach (vs best lap)")
    for n in driver.get("notes") or []:
        st.markdown(f"- {n}")
    z = driver.get("time_loss_zones") or []
    if z:
        st.markdown("**Top time-loss zones (speed proxy)**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Distance (m)": int(x["distance_m"]),
                        "Proxy loss (s)": round(x["time_loss_s"], 3),
                        "Best lap kph": round(x["speed_ref_kph"], 1),
                        "Other kph": round(x["speed_cmp_kph"], 1),
                        "Δ kph": round(x["speed_delta_kph"], 1),
                    }
                    for x in z
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        m = driver.get("delta_series")
        if m is not None and len(m) > 5 and "time_loss_pos" in m.columns:
            fig_tl = go.Figure()
            fig_tl.add_trace(
                go.Scatter(
                    x=m["bin"],
                    y=m["time_loss_pos"],
                    name="Proxy time loss (s)",
                    fill="tozeroy",
                )
            )
            fig_tl.update_layout(
                height=280,
                xaxis_title="Distance (m)",
                yaxis_title="Seconds slower than best (proxy)",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_tl, use_container_width=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.markdown("**Braking**")
        for t in driver.get("brake_notes") or []:
            st.write(t)
    with dc2:
        st.markdown("**Throttle / exit**")
        for t in driver.get("throttle_notes") or []:
            st.write(t)
    with dc3:
        st.markdown("**Scrub / balance feel**")
        for t in driver.get("scrub_notes") or []:
            st.write(t)

    # ----- Recommendations (multiple options per issue; limits enforced) -----
    st.subheader("Recommended setup changes")
    st.caption(
        "Ordered by issue criticality when possible. Each issue lists **multiple options**. "
        "Suggestions that hit a min/max limit are **blocked** so you can use the next lever."
    )
    if not changes:
        st.success("No strong setup signals — check thresholds or drive more representative laps.")
    else:
        # Order issue blocks by tier of matching summary
        tier_by_name = {s.name: s.tier for s in shown}
        crit_by_name = {s.name: s.criticality for s in shown}
        by_issue = recommendations_by_issue(changes)
        ordered_names = sorted(
            by_issue.keys(),
            key=lambda n: (
                {"S": 0, "A": 1, "B": 2, "C": 3}.get(tier_by_name.get(n, "C"), 9),
                -crit_by_name.get(n, 0),
            ),
        )
        for issue_name in ordered_names:
            opts = by_issue[issue_name]
            feasible_opts = [o for o in opts if o.feasible]
            blocked_opts = [o for o in opts if not o.feasible]
            tier = tier_by_name.get(issue_name, "?")
            with st.container(border=True):
                st.markdown(f"### [{tier}] {issue_name}")
                if not feasible_opts:
                    st.warning(
                        "All primary levers are at limits or unavailable. Review blocked options below."
                    )
                for ch in feasible_opts:
                    st.markdown(
                        f"**{ch.option_label}: {ch.parameter}** — `{ch.direction}` · {ch.amount_hint}"
                    )
                    st.write(ch.reason)
                    st.caption(f"Validate: {ch.validation_metric}")
                if blocked_opts:
                    with st.expander(f"Blocked options ({len(blocked_opts)}) — at limit or impossible"):
                        for ch in blocked_opts:
                            st.markdown(
                                f"**{ch.option_label}: {ch.parameter}** — `{ch.direction}` · ~~{ch.amount_hint}~~"
                            )
                            st.caption(ch.blocked_reason or "Not feasible with current setup.")

    # ----- Issues table -----
    st.subheader("Issues (with frequency & criticality)")
    if not any(
        s.count >= min_count and s.lap_presence_pct >= min_presence for s in summaries
    ):
        st.warning(
            "No issues passed the frequency filters (showing top raw issues). "
            "Lower 'Min event count' / 'Min lap presence' in the sidebar if needed."
        )

    if shown:
        table = pd.DataFrame(
            [
                {
                    "Tier": s.tier,
                    "Issue": s.name,
                    "Criticality": round(s.criticality, 2),
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
            color="Tier",
            title="Issue frequency (clustered events) by tier",
            category_orders={"Tier": ["S", "A", "B", "C"]},
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
                    fig2.add_hline(y=us_alpha, line_dash="dot", annotation_text="US")
                    fig2.add_hline(y=-os_alpha, line_dash="dot", annotation_text="OS")
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
`α_balance = |α_f| − |α_r|` with thresholds {us_alpha} / {os_alpha} rad.

**Frequency:** raw samples are clustered by **lap + 50 m distance bin** so one long push corner
counts as one event, not hundreds of rows. Presence % = share of laps where the issue appeared.

**Criticality:** `events/lap × severity × confidence × issue weight × presence factor` → tiers S/A/B/C.

**Driver coach:** speed vs best lap by distance bin; proxy time loss `ds×(1/v−1/v_best)`.

**Session grade:** weighted pace (40) + consistency (20) + cleanliness (20) + balance (10) + tires (10).

**Setup advice:** multiple levers per issue; blocked at user-confirmed setup limits.

**Axes:** lateral/longitudinal G auto-assigned from correlation with steering when possible.
            """
        )

    with st.expander("All issue IDs (debug)"):
        st.write([asdict(s) for s in summaries])


if __name__ == "__main__":
    main()
