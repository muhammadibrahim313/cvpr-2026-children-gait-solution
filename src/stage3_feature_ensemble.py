"""Stage 3 CGC-inspired feature ensemble for CVPR 2026 Children Gait.

Purpose
-------
This is a focused next experiment after the current best public submission:

    submission_stage1best_t1_stage2_t2.csv -> 0.53195

It adapts the strongest immediately runnable idea from the public 5th-place
CGC notebook: clinical pose features + tree ensemble for Track 1. It keeps the
already proven Stage 2 Track 2 rows, because Track 2 was responsible for the
jump from ~0.49 to ~0.532.

Run this in Kaggle after copying/restoring the V1 artifacts into
/kaggle/working. It does not overwrite /kaggle/working/submission.csv.

Outputs
-------
- /kaggle/working/submission_stage3_cgc_t1_stage2t2.csv
- /kaggle/working/submission_stage3_cgc_t1_stage2t2_lesspos005.csv
- /kaggle/working/submission_stage3_cgc_t1_stage2t2_morepos005.csv
- /kaggle/working/tables/<RUN_ID>/stage3_track1_oof_prob.csv
- /kaggle/working/versioning/<RUN_ID>/run_summary.json

Submit only the base file first if the printed CV and totals look sane.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:
    joblib = None


# =========================
# Config
# =========================

SEED = 2026
MAX_FRAMES_PER_VIEW = 500
CV_SEEDS = [42, 123, 456, 789, 321]
N_SPLITS = 5
N_JOBS = -1

KAGGLE_COMP_DIR = Path("/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge")
LOCAL_COMP_DIR = Path("data/kaggle_hosted")
KAGGLE_TRACK1 = KAGGLE_COMP_DIR / "track1_train.json"
KAGGLE_TRACK2 = KAGGLE_COMP_DIR / "track2_train.json"
LOCAL_TRACK1 = LOCAL_COMP_DIR / "track1_train.json"
LOCAL_TRACK2 = LOCAL_COMP_DIR / "track2_train.json"
PATH1 = KAGGLE_TRACK1 if KAGGLE_TRACK1.exists() else LOCAL_TRACK1
PATH2 = KAGGLE_TRACK2 if KAGGLE_TRACK2.exists() else LOCAL_TRACK2

KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/qasminumber/cvpr-2026-data/dataset")
LOCAL_DATA_ROOT = Path("data/full_dataset/dataset")
DEFAULT_DATA_ROOT = KAGGLE_DATA_ROOT if KAGGLE_DATA_ROOT.exists() else LOCAL_DATA_ROOT
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_working")
RUN_ID = datetime.now(UTC).strftime("stage3_cgc_%Y%m%d_%H%M%S")
TABLE_DIR = WORK_DIR / "tables" / RUN_ID
VERSION_DIR = WORK_DIR / "versioning" / RUN_ID
MODEL_DIR = VERSION_DIR / "models"

T1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
T2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]
LABEL_COLS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
SUBMISSION_COLUMNS = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)
VIEWS = ["backward", "forward", "left", "right"]
GAIT_IDXS = [5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
VIEW_RE = re.compile(r"_(forward|backward|left|right)_", re.IGNORECASE)


# =========================
# Utilities
# =========================

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def section(title: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def ensure_dirs() -> None:
    for d in [TABLE_DIR, VERSION_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_data_root() -> Path:
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    for d in Path("/kaggle/input").rglob("dataset"):
        if d.is_dir() and any(d.glob("*/**/frame_*.json")):
            return d
    return DEFAULT_DATA_ROOT


def parse_view(seq_dir: Path) -> str:
    m = VIEW_RE.search(seq_dir.name)
    if m:
        return m.group(1).lower()
    lower = seq_dir.name.lower()
    for view in VIEWS:
        if view in lower:
            return view
    return "unknown"


def parse_patient_id(seq_dir: Path) -> int:
    digits = re.sub(r"\D", "", seq_dir.parent.name)
    return int(digits)


def sample_evenly(files: list[Path], max_count: int) -> list[Path]:
    if len(files) <= max_count:
        return files
    idx = np.linspace(0, len(files) - 1, max_count).round().astype(int)
    return [files[int(i)] for i in idx]


def flatten_track1(track1: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in track1:
        row = {"patient_id": int(item["patient_id"])}
        for i in range(1, 18):
            row[f"L{i}"] = int(item["left"][str(i)])
            row[f"R{i}"] = int(item["right"][str(i)])
        row["Total"] = int(sum(row[c] for c in LABEL_COLS))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def build_patient_files(data_root: Path) -> dict[int, dict[str, list[Path]]]:
    patient_files: dict[int, dict[str, list[Path]]] = {}
    sequence_dirs = sorted({p.parent for p in data_root.rglob("frame_*.json")})
    for i, seq_dir in enumerate(sequence_dirs, start=1):
        try:
            pid = parse_patient_id(seq_dir)
        except Exception:
            continue
        view = parse_view(seq_dir)
        if view not in VIEWS:
            continue
        files = sorted(seq_dir.glob("frame_*.json"))
        patient_files.setdefault(pid, {v: [] for v in VIEWS})
        patient_files[pid][view].extend(files)
        if i % 250 == 0:
            log(f"Indexed {i}/{len(sequence_dirs)} sequence dirs")
    return patient_files


# =========================
# Pose feature extraction
# =========================

def load_kpts(files: list[Path], max_frames: int = MAX_FRAMES_PER_VIEW) -> tuple[np.ndarray | None, float]:
    chosen = sample_evenly(sorted(files), max_frames)
    rows = []
    fps = 30.0
    for fp in chosen:
        try:
            data = load_json(fp)
            fps = float(data.get("video_info", {}).get("fps", fps) or fps)
            inst = data.get("instance_info", [])
            if not inst:
                continue
            kp = np.asarray(inst[0].get("keypoints", []), dtype=np.float32)
            if kp.ndim == 2 and kp.shape[0] >= 23 and kp.shape[1] >= 2:
                rows.append(kp[:23, :2])
        except Exception:
            continue
    if not rows:
        return None, fps
    return np.stack(rows).astype(np.float32), fps


def ang3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8
    cosv = np.sum(ba * bc, axis=-1) / denom
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))


def stats(arr: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        x = np.array([0.0], dtype=np.float32)
    return {
        f"{prefix}_m": float(np.mean(x)),
        f"{prefix}_s": float(np.std(x)),
        f"{prefix}_mn": float(np.min(x)),
        f"{prefix}_mx": float(np.max(x)),
        f"{prefix}_r": float(np.ptp(x)),
        f"{prefix}_md": float(np.median(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p25": float(np.percentile(x, 25)),
        f"{prefix}_p75": float(np.percentile(x, 75)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
    }


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    val = float(np.corrcoef(a, b)[0, 1])
    return val if np.isfinite(val) else 0.0


def interp_seq(seq: np.ndarray, target: int = 60) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32).reshape(-1)
    seq = seq[np.isfinite(seq)]
    if len(seq) == 0:
        return np.zeros(target, dtype=np.float32)
    if len(seq) == 1:
        return np.full(target, float(seq[0]), dtype=np.float32)
    x = np.linspace(0, 1, len(seq))
    xn = np.linspace(0, 1, target)
    return interp1d(x, seq, kind="linear", fill_value="extrapolate")(xn).astype(np.float32)


def add_cadence_fft_features(f: dict[str, float], kpts: np.ndarray, fps: float) -> None:
    dt = 1.0 / max(float(fps), 1.0)
    for side, ankle_idx in [("L", 15), ("R", 16)]:
        y = kpts[:, ankle_idx, 1].astype(np.float32)
        yc = y - np.mean(y)
        if len(yc) > 20:
            ac = np.correlate(yc, yc, mode="full")[len(yc) - 1 :]
            ac = ac / (float(ac[0]) + 1e-8)
            peaks = [
                j
                for j in range(1, len(ac) - 1)
                if ac[j] > ac[j - 1] and ac[j] > ac[j + 1] and ac[j] > 0.2
            ]
            if peaks:
                period = peaks[0] * dt
                f[f"{side}_cad"] = float(60.0 / period) if period > 0 else 0.0
                f[f"{side}_step_t"] = float(period)
                f[f"{side}_step_ac"] = float(ac[peaks[0]])

            fft = np.abs(np.fft.rfft(yc))
            freqs = np.fft.rfftfreq(len(yc), d=dt)
            if len(fft) > 1:
                dom = int(np.argmax(fft[1:]) + 1)
                f[f"{side}_dom_freq"] = float(freqs[dom])
                f[f"{side}_dom_amp"] = float(fft[dom] / max(len(yc), 1))
                mask = (freqs >= 0.5) & (freqs <= 2.0)
                f[f"{side}_walk_power"] = float(np.sum(fft[mask] ** 2) / (np.sum(fft**2) + 1e-8))


def add_phase_features(
    f: dict[str, float],
    kpts: np.ndarray,
    knee: np.ndarray,
    hip: np.ndarray,
    ankle: np.ndarray,
    fps: float,
) -> None:
    dt = 1.0 / max(float(fps), 1.0)
    if len(kpts) < 12:
        return
    vel = np.diff(kpts, axis=0) / dt
    for side, ankle_idx, knee_arr, hip_arr, ankle_arr in [
        ("L", 15, knee["L"], hip["L"], ankle["L"]),
        ("R", 16, knee["R"], hip["R"], ankle["R"]),
    ]:
        speed = np.linalg.norm(vel[:, ankle_idx], axis=-1)
        if len(speed) < 5:
            continue
        speed = np.r_[speed, speed[-1]]
        lo = np.percentile(speed, 40)
        hi = np.percentile(speed, 60)
        stance = speed <= lo
        swing = speed >= hi
        for phase_name, mask in [("stance", stance), ("swing", swing)]:
            if int(mask.sum()) < 3:
                continue
            f.update(stats(knee_arr[mask], f"{side}_{phase_name}_knee"))
            f.update(stats(hip_arr[mask], f"{side}_{phase_name}_hip"))
            f.update(stats(ankle_arr[mask], f"{side}_{phase_name}_ankle"))


def extract_features(kpts: np.ndarray | None, fps: float = 30.0) -> dict[str, float] | None:
    if kpts is None or len(kpts) < 10:
        return None
    kpts = np.asarray(kpts, dtype=np.float32)
    n = len(kpts)
    f: dict[str, float] = {"n_frames": float(n), "fps": float(fps)}

    hip_center = (kpts[:, 11] + kpts[:, 12]) / 2.0
    ankle_mid = (kpts[:, 15] + kpts[:, 16]) / 2.0
    body_height = float(np.nanmean(np.linalg.norm(kpts[:, 0] - ankle_mid, axis=-1))) + 1e-8
    norm = (kpts - hip_center[:, None, :]) / body_height

    angle_defs = {
        "Lkn": (11, 13, 15),
        "Rkn": (12, 14, 16),
        "Lhp": (5, 11, 13),
        "Rhp": (6, 12, 14),
        "Lak": (13, 15, 17),
        "Rak": (14, 16, 20),
        "Lakh": (13, 15, 19),
        "Rakh": (14, 16, 22),
    }
    angle_series: dict[str, np.ndarray] = {}
    for name, (a, b, c) in angle_defs.items():
        arr = ang3(kpts[:, a], kpts[:, b], kpts[:, c])
        angle_series[name] = arr
        f.update(stats(arr, name))

    shoulder_mid = (kpts[:, 5] + kpts[:, 6]) / 2.0
    hip_mid = (kpts[:, 11] + kpts[:, 12]) / 2.0
    trunk = np.degrees(np.arctan2((shoulder_mid - hip_mid)[:, 0], -(shoulder_mid - hip_mid)[:, 1] + 1e-8))
    f.update(stats(trunk, "trunk"))
    f.update(stats(kpts[:, 5, 1] - kpts[:, 6, 1], "shoulder_tilt_y"))
    f.update(stats(kpts[:, 11, 1] - kpts[:, 12, 1], "pelvis_tilt_y"))
    f.update(stats(np.abs(kpts[:, 5, 0] - kpts[:, 6, 0]), "shoulder_width"))
    f.update(stats(np.abs(kpts[:, 11, 0] - kpts[:, 12, 0]), "hip_width"))
    step_width = np.abs(kpts[:, 15, 0] - kpts[:, 16, 0])
    f.update(stats(step_width, "step_width"))
    f["step_width_cv"] = float(np.std(step_width) / (np.mean(step_width) + 1e-8))

    for side, ankle_idx, knee_idx, hip_idx, toe_idx, heel_idx in [
        ("L", 15, 13, 11, 17, 19),
        ("R", 16, 14, 12, 20, 22),
    ]:
        f.update(stats(kpts[:, ankle_idx, 1], f"{side}_ankle_y"))
        f.update(stats(kpts[:, ankle_idx, 0], f"{side}_ankle_x"))
        thigh = np.linalg.norm(kpts[:, hip_idx] - kpts[:, knee_idx], axis=-1)
        shank = np.linalg.norm(kpts[:, knee_idx] - kpts[:, ankle_idx], axis=-1)
        f.update(stats(thigh / (shank + 1e-8), f"{side}_thigh_shank"))
        knee_ankle_x = (kpts[:, knee_idx, 0] - kpts[:, hip_idx, 0]) - 0.5 * (kpts[:, ankle_idx, 0] - kpts[:, hip_idx, 0])
        f.update(stats(knee_ankle_x, f"{side}_knee_ankle_x"))
        toe_heel = kpts[:, toe_idx] - kpts[:, heel_idx]
        foot_angle = np.degrees(np.arctan2(toe_heel[:, 1], toe_heel[:, 0] + 1e-8))
        f.update(stats(foot_angle, f"{side}_foot_angle"))
        f.update(stats(kpts[:, toe_idx, 1] - kpts[:, heel_idx, 1], f"{side}_toe_heel_y"))
        f.update(stats(kpts[:, toe_idx, 0] - kpts[:, heel_idx, 0], f"{side}_toe_heel_x"))

    f["ankle_y_range_asym"] = abs(f.get("L_ankle_y_r", 0.0) - f.get("R_ankle_y_r", 0.0))
    f["step_time_asym"] = abs(f.get("L_step_t", 0.0) - f.get("R_step_t", 0.0))

    if n >= 3:
        dt = 1.0 / max(float(fps), 1.0)
        vel = np.diff(kpts, axis=0) / dt
        for name, idx in [("La", 15), ("Ra", 16), ("Lk", 13), ("Rk", 14), ("Lt", 17), ("Rt", 20)]:
            speed = np.linalg.norm(vel[:, idx], axis=-1)
            f.update(stats(speed, f"{name}_speed"))
            if len(speed) > 1:
                f[f"{name}_acc_abs_m"] = float(np.mean(np.abs(np.diff(speed) / dt)))

    add_cadence_fft_features(f, kpts, fps)

    for idx in GAIT_IDXS:
        f[f"n{idx}_xm"] = float(np.mean(norm[:, idx, 0]))
        f[f"n{idx}_ym"] = float(np.mean(norm[:, idx, 1]))
        f[f"n{idx}_xs"] = float(np.std(norm[:, idx, 0]))
        f[f"n{idx}_ys"] = float(np.std(norm[:, idx, 1]))
        f[f"n{idx}_xr"] = float(np.ptp(norm[:, idx, 0]))
        f[f"n{idx}_yr"] = float(np.ptp(norm[:, idx, 1]))

    for left, right, out_name in [
        ("Lkn", "Rkn", "knee"),
        ("Lhp", "Rhp", "hip"),
        ("Lak", "Rak", "ankle"),
        ("Lakh", "Rakh", "ankle_heel"),
    ]:
        for suffix in ["_m", "_s", "_r", "_p25", "_p75"]:
            lk = f"{left}{suffix}"
            rk = f"{right}{suffix}"
            if lk in f and rk in f:
                f[f"sym_{out_name}{suffix}"] = abs(f[lk] - f[rk]) / (abs(f[lk]) + abs(f[rk]) + 1e-8)

    f["cc_knee_y"] = safe_corr(kpts[:, 13, 1], kpts[:, 14, 1])
    f["cc_ankle_y"] = safe_corr(kpts[:, 15, 1], kpts[:, 16, 1])
    f["cc_ankle_x"] = safe_corr(kpts[:, 15, 0], kpts[:, 16, 0])

    head_y = kpts[:, 0, 1]
    f.update(stats(head_y, "head_y"))
    f.update(stats(kpts[:, 0, 0], "head_x"))
    f["head_bob_over_body"] = float(np.ptp(head_y) / body_height)
    f["head_sway_over_body"] = float(np.ptp(kpts[:, 0, 0]) / body_height)

    q1 = int(max(0, min(n - 1, round(0.15 * n))))
    q2 = int(max(q1 + 1, min(n, round(0.25 * n))))
    mid = slice(q1, q2)
    for name in ["Lkn", "Rkn", "Lhp", "Rhp", "Lak", "Rak"]:
        f.update(stats(angle_series[name][mid], f"midstance_{name}"))

    add_phase_features(
        f,
        kpts,
        knee={"L": angle_series["Lkn"], "R": angle_series["Rkn"]},
        hip={"L": angle_series["Lhp"], "R": angle_series["Rhp"]},
        ankle={"L": angle_series["Lak"], "R": angle_series["Rak"]},
        fps=fps,
    )

    for key, value in list(f.items()):
        if not np.isfinite(value):
            f[key] = 0.0
    return f


def build_features(patient_files: dict[int, dict[str, list[Path]]], patient_ids: list[int]) -> dict[int, dict[str, float]]:
    features: dict[int, dict[str, float]] = {}
    total = len(patient_ids)
    for i, pid in enumerate(sorted(patient_ids), start=1):
        per_view = []
        row: dict[str, float] = {}
        for view in VIEWS:
            files = patient_files.get(pid, {}).get(view, [])
            if not files:
                continue
            kpts, fps = load_kpts(files)
            ft = extract_features(kpts, fps)
            if ft is None:
                continue
            per_view.append(ft)
            for k, v in ft.items():
                row[f"{view}__{k}"] = float(v)
        if per_view:
            keys = sorted(set().union(*[set(d.keys()) for d in per_view]))
            for key in keys:
                vals = np.array([d[key] for d in per_view if key in d and np.isfinite(d[key])], dtype=np.float32)
                if vals.size:
                    row[f"avg__{key}"] = float(np.mean(vals))
                    if vals.size >= 2:
                        row[f"viewstd__{key}"] = float(np.std(vals))
            features[pid] = row
        if i % 10 == 0 or i == total:
            log(f"Built CGC features {i}/{total}; usable={len(features)}")
    return features


# =========================
# Track 1 model
# =========================

def make_matrix(pids: list[int], features: dict[int, dict[str, float]], feature_names: list[str] | None = None) -> tuple[np.ndarray, list[int], list[str]]:
    valid = [int(p) for p in pids if int(p) in features]
    if not valid:
        raise ValueError("No patient features available")
    if feature_names is None:
        feature_names = sorted(set().union(*[set(features[p].keys()) for p in valid]))
    X = np.array([[features[p].get(name, 0.0) for name in feature_names] for p in valid], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, valid, feature_names


def make_models(seed: int) -> list[tuple[str, MultiOutputClassifier]]:
    return [
        (
            "gbm",
            MultiOutputClassifier(
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    min_samples_leaf=3,
                    random_state=seed,
                ),
                n_jobs=N_JOBS,
            ),
        ),
        (
            "et",
            MultiOutputClassifier(
                ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=4,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=N_JOBS,
                ),
                n_jobs=1,
            ),
        ),
        (
            "rf",
            MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=4,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=N_JOBS,
                ),
                n_jobs=1,
            ),
        ),
    ]


def multioutput_positive_proba(model: MultiOutputClassifier, X: np.ndarray) -> np.ndarray:
    cols = []
    for est in model.estimators_:
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X)
            classes = list(getattr(est, "classes_", []))
            if 1 in classes:
                cols.append(proba[:, classes.index(1)])
            elif len(classes) == 1:
                cols.append(np.ones(len(X), dtype=np.float32) if int(classes[0]) == 1 else np.zeros(len(X), dtype=np.float32))
            else:
                cols.append(np.zeros(len(X), dtype=np.float32))
        else:
            cols.append(est.predict(X).astype(np.float32))
    return np.vstack(cols).T.astype(np.float32)


def s1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float((y_true == y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_pred.sum(axis=1) - y_true.sum(axis=1)) ** 2)))
    score = float((acc + 1.0 - rmse / 34.0) / 2.0)
    return {"accuracy": acc, "rmse": rmse, "score": score}


def split_iterator(X: np.ndarray, y: np.ndarray, seed: int):
    totals = y.sum(axis=1).astype(int)
    strat = np.clip(totals, 0, 6)
    counts = pd.Series(strat).value_counts()
    if len(counts) > 1 and int(counts.min()) >= N_SPLITS:
        return StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X, strat)
    return KFold(N_SPLITS, shuffle=True, random_state=seed).split(X)


def tune_thresholds(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    thresholds = np.full(prob.shape[1], 0.5, dtype=np.float32)
    for c in range(prob.shape[1]):
        best_acc = -1.0
        best_t = 0.5
        for t in np.arange(0.30, 0.701, 0.05):
            acc = accuracy_score(y[:, c], (prob[:, c] >= t).astype(int))
            if acc > best_acc:
                best_acc = float(acc)
                best_t = float(t)
        thresholds[c] = best_t
    return thresholds


def train_track1_ensemble(X_raw: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[str, MultiOutputClassifier]], VarianceThreshold, StandardScaler, dict[str, float]]:
    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_vt)
    log(f"Track1 matrix after variance filter: {X.shape}")

    oof = np.zeros_like(y, dtype=np.float32)
    counts = np.zeros(len(y), dtype=np.int32)
    start = time.time()
    for seed_index, seed in enumerate(CV_SEEDS, start=1):
        for fold, (tr, va) in enumerate(split_iterator(X, y, seed), start=1):
            fold_prob = np.zeros((len(va), y.shape[1]), dtype=np.float32)
            for name, model in make_models(seed):
                model.fit(X[tr], y[tr])
                fold_prob += multioutput_positive_proba(model, X[va]) / 3.0
            oof[va] += fold_prob
            counts[va] += 1
            fold_score = s1_metric(y[va], (fold_prob >= 0.5).astype(int))
            log(
                f"seed {seed_index}/{len(CV_SEEDS)} fold {fold}: "
                f"S1@0.5={fold_score['score']:.5f} acc={fold_score['accuracy']:.5f} rmse={fold_score['rmse']:.5f}"
            )
            gc.collect()
        log(f"Completed CV seed {seed_index}/{len(CV_SEEDS)} in {(time.time() - start) / 60.0:.1f} min")

    oof = oof / np.maximum(counts[:, None], 1)
    thresholds = tune_thresholds(y, oof)
    tuned = (oof >= thresholds[None, :]).astype(int)
    cv_score = s1_metric(y, tuned)
    log(f"Track1 OOF tuned: S1={cv_score['score']:.5f} acc={cv_score['accuracy']:.5f} rmse={cv_score['rmse']:.5f}")

    final_models = make_models(SEED)
    for name, model in final_models:
        log(f"Fitting final {name}")
        model.fit(X, y)
    return oof, thresholds, final_models, vt, scaler, cv_score


def predict_track1(final_models: list[tuple[str, MultiOutputClassifier]], vt: VarianceThreshold, scaler: StandardScaler, X_raw: np.ndarray, thresholds: np.ndarray, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    X = scaler.transform(vt.transform(X_raw))
    prob = np.zeros((len(X), len(LABEL_COLS)), dtype=np.float32)
    for name, model in final_models:
        prob += multioutput_positive_proba(model, X) / len(final_models)
    th = np.clip(thresholds + float(shift), 0.05, 0.95)
    pred = (prob >= th[None, :]).astype(int)
    return pred, prob


# =========================
# Submission
# =========================

def load_best_track2_rows() -> pd.DataFrame:
    candidates = [
        WORK_DIR / "submission_stage1best_t1_stage2_t2.csv",
        WORK_DIR / "submission_stage1_t1_lesspos_01250_stage2t2_recreated.csv",
        WORK_DIR / "submission_stage2_pose_tcn.csv",
        WORK_DIR / "submission.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "ID" in df.columns and df["ID"].astype(str).str.startswith("track2-").any():
                log(f"Using Track2 rows from {path}")
                return df[df["ID"].astype(str).str.startswith("track2-")].copy()
    raise FileNotFoundError("No prior submission with Track2 rows found in /kaggle/working")


def load_current_best_for_compare() -> pd.DataFrame | None:
    for path in [
        WORK_DIR / "submission_stage1best_t1_stage2_t2.csv",
        WORK_DIR / "submission_stage1_t1_lesspos_01250_stage2t2_recreated.csv",
        WORK_DIR / "submission.csv",
    ]:
        if path.exists():
            return pd.read_csv(path)
    return None


def make_submission(t1_pred: np.ndarray, t2_rows: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    rows = []
    for pid, labels in zip(T1_TEST_IDS, t1_pred.astype(int)):
        labels = labels.tolist()
        rows.append([f"track1-{pid}", *labels[:17], *labels[17:], int(sum(labels)), -1, -1])
    t1_df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    out = pd.concat([t1_df, t2_rows[SUBMISSION_COLUMNS]], ignore_index=True)
    out = out[SUBMISSION_COLUMNS]
    assert out.shape == (25, 38), out.shape
    assert list(out["ID"][:16]) == [f"track1-{p}" for p in T1_TEST_IDS]
    assert list(out["ID"][16:]) == [f"track2-{p}" for p in T2_TEST_IDS]
    out.to_csv(out_path, index=False)
    log(f"Wrote {out_path}")
    return out


def compare_to_current_best(candidate: pd.DataFrame, current: pd.DataFrame | None) -> dict[str, Any]:
    if current is None:
        return {}
    cand = candidate[candidate["ID"].astype(str).str.startswith("track1-")].reset_index(drop=True)
    best = current[current["ID"].astype(str).str.startswith("track1-")].reset_index(drop=True)
    if len(cand) != len(best):
        return {}
    diff = cand[LABEL_COLS].astype(int).values - best[LABEL_COLS].astype(int).values
    changed = np.argwhere(diff != 0)
    by_patient: dict[str, int] = {}
    for row_idx, col_idx in changed:
        pid = str(cand.loc[row_idx, "ID"])
        by_patient[pid] = by_patient.get(pid, 0) + 1
    return {
        "changed_labels_vs_current_best": int(len(changed)),
        "changes_by_patient": by_patient,
        "candidate_totals": dict(zip(cand["ID"].tolist(), cand["Total"].astype(int).tolist())),
        "current_best_totals": dict(zip(best["ID"].tolist(), best["Total"].astype(int).tolist())),
    }


def main() -> None:
    start = time.time()
    seed_everything()
    ensure_dirs()

    section("Stage 3 CGC Feature Ensemble")
    data_root = find_data_root()
    log(f"RUN_ID={RUN_ID}")
    log(f"PATH1={PATH1}")
    log(f"PATH2={PATH2}")
    log(f"DATA_ROOT={data_root}")
    log(f"WORK_DIR={WORK_DIR}")

    if not PATH1.exists():
        raise FileNotFoundError(PATH1)
    if not data_root.exists():
        raise FileNotFoundError(data_root)

    config = {
        "run_id": RUN_ID,
        "seed": SEED,
        "max_frames_per_view": MAX_FRAMES_PER_VIEW,
        "cv_seeds": CV_SEEDS,
        "n_splits": N_SPLITS,
        "path1": str(PATH1),
        "path2": str(PATH2),
        "data_root": str(data_root),
    }
    dump_json(config, VERSION_DIR / "run_config.json")

    section("Load labels and build patient files")
    track1_df = flatten_track1(load_json(PATH1))
    patient_files = build_patient_files(data_root)
    all_pids = sorted(set(track1_df["patient_id"].astype(int).tolist()) | set(T1_TEST_IDS) | set(T2_TEST_IDS))
    log(f"Track1 train={track1_df.shape}, patient_file_count={len(patient_files)}, feature_pids={len(all_pids)}")
    track1_df.to_csv(TABLE_DIR / "track1_train_flat.csv", index=False)

    section("Extract CGC-inspired features")
    features = build_features(patient_files, all_pids)
    feature_dump = pd.DataFrame([{"patient_id": pid, **features[pid]} for pid in sorted(features)])
    feature_dump.to_csv(TABLE_DIR / "stage3_patient_features.csv", index=False)
    log(f"Feature table: {feature_dump.shape}")

    section("Train Track 1 ensemble")
    X_raw, train_pids, feature_names = make_matrix(track1_df["patient_id"].astype(int).tolist(), features)
    y_df = track1_df.set_index("patient_id")
    y = np.array([[int(y_df.loc[pid, col]) for col in LABEL_COLS] for pid in train_pids], dtype=np.int32)
    log(f"Track1 X_raw={X_raw.shape}, y={y.shape}")

    oof, thresholds, final_models, vt, scaler, cv_score = train_track1_ensemble(X_raw, y)
    pd.DataFrame(oof, columns=LABEL_COLS).assign(patient_id=train_pids).to_csv(TABLE_DIR / "stage3_track1_oof_prob.csv", index=False)
    pd.DataFrame({"label": LABEL_COLS, "threshold": thresholds}).to_csv(TABLE_DIR / "stage3_track1_thresholds.csv", index=False)
    with (VERSION_DIR / "feature_names.txt").open("w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")

    if joblib is not None:
        joblib.dump({"models": final_models, "vt": vt, "scaler": scaler, "thresholds": thresholds, "feature_names": feature_names}, MODEL_DIR / "track1_stage3_cgc.joblib")
        log(f"Saved model bundle to {MODEL_DIR / 'track1_stage3_cgc.joblib'}")

    section("Create submissions with Stage 2 Track 2")
    X_test_raw, test_pids, _ = make_matrix(T1_TEST_IDS, features, feature_names)
    if test_pids != T1_TEST_IDS:
        raise ValueError(f"Missing test features. Got {test_pids}, expected {T1_TEST_IDS}")
    t2_rows = load_best_track2_rows()
    current_best = load_current_best_for_compare()

    outputs: dict[str, dict[str, Any]] = {}
    for shift, name in [
        (0.0, "submission_stage3_cgc_t1_stage2t2.csv"),
        (0.05, "submission_stage3_cgc_t1_stage2t2_lesspos005.csv"),
        (-0.05, "submission_stage3_cgc_t1_stage2t2_morepos005.csv"),
    ]:
        pred, prob = predict_track1(final_models, vt, scaler, X_test_raw, thresholds, shift=shift)
        out = make_submission(pred, t2_rows, WORK_DIR / name)
        info = compare_to_current_best(out, current_best)
        outputs[name] = {
            "threshold_shift": shift,
            "track1_totals": dict(zip([f"track1-{p}" for p in T1_TEST_IDS], pred.sum(axis=1).astype(int).tolist())),
            **info,
        }
        print("\nCandidate:", name, flush=True)
        print("threshold_shift:", shift, flush=True)
        print("Track1 totals:", outputs[name]["track1_totals"], flush=True)
        if info:
            print("changed_labels_vs_current_best:", info["changed_labels_vs_current_best"], flush=True)
            print("changes_by_patient:", info["changes_by_patient"], flush=True)

    summary = {
        "run_id": RUN_ID,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "cv": {"track1": cv_score},
        "feature_table": str(TABLE_DIR / "stage3_patient_features.csv"),
        "outputs": outputs,
        "recommendation": (
            "Submit submission_stage3_cgc_t1_stage2t2.csv first only if Track1 OOF S1 is clearly above "
            "the old Stage1 OOF 0.7196 and changed labels are not extreme. Keep current best selected otherwise."
        ),
    }
    dump_json(summary, VERSION_DIR / "run_summary.json")
    log(f"Saved summary to {VERSION_DIR / 'run_summary.json'}")
    log(f"Elapsed minutes: {(time.time() - start) / 60.0:.2f}")
    section("Done")


if __name__ == "__main__":
    main()
