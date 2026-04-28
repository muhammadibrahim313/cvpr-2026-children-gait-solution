"""Stage 1 Kaggle baseline for CVPR 2026 AI for Children Challenge.

Copy this whole file into one Kaggle notebook cell, or upload it as a script and
run it. It is designed for collaboration:

- prints progress and useful summaries;
- saves tables to /kaggle/working/tables/<RUN_ID>/;
- saves config/summary/models to /kaggle/working/versioning/<RUN_ID>/;
- writes /kaggle/working/submission.csv for direct Kaggle submission.

This first pass is CPU-safe. It samples frames from each sequence and builds
patient-level clinical/pose features. Later GPU models can reuse the saved
tables.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:
    joblib = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =========================
# Config
# =========================

SEED = 42
MAX_FRAMES_PER_SEQUENCE = 96  # lower to 48 if the first run is too slow
LOW_SCORE_THR = 0.30

KAGGLE_TRACK1 = Path("/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track1_train.json")
KAGGLE_TRACK2 = Path("/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track2_train.json")
KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/qasminumber/cvpr-2026-data/dataset")

LOCAL_TRACK1 = Path("data/kaggle_hosted/track1_train.json")
LOCAL_TRACK2 = Path("data/kaggle_hosted/track2_train.json")
LOCAL_DATA_ROOT = Path("data/full_dataset/dataset")

PATH1 = KAGGLE_TRACK1 if KAGGLE_TRACK1.exists() else LOCAL_TRACK1
PATH2 = KAGGLE_TRACK2 if KAGGLE_TRACK2.exists() else LOCAL_TRACK2
DATA_ROOT = KAGGLE_DATA_ROOT if KAGGLE_DATA_ROOT.exists() else LOCAL_DATA_ROOT

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_working")
RUN_ID = datetime.utcnow().strftime("stage1_%Y%m%d_%H%M%S")
TABLE_DIR = WORK_DIR / "tables" / RUN_ID
VERSION_DIR = WORK_DIR / "versioning" / RUN_ID
FIG_DIR = WORK_DIR / "figures" / RUN_ID
MODEL_DIR = VERSION_DIR / "models"

T1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
T2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]
SUBMISSION_COLUMNS = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)

VIEWS = ["backward", "forward", "left", "right"]
VIEW_RE = re.compile(r"_(forward|backward|left|right)_", re.IGNORECASE)

JOINT_NAMES = {
    0: "nose",
    1: "eye_l",
    2: "eye_r",
    3: "ear_l",
    4: "ear_r",
    5: "shoulder_l",
    6: "shoulder_r",
    7: "elbow_l",
    8: "elbow_r",
    9: "wrist_l",
    10: "wrist_r",
    11: "hip_l",
    12: "hip_r",
    13: "knee_l",
    14: "knee_r",
    15: "ankle_l",
    16: "ankle_r",
    17: "bigtoe_l",
    18: "smalltoe_l",
    19: "heel_l",
    20: "bigtoe_r",
    21: "smalltoe_r",
    22: "heel_r",
}


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
    for d in [TABLE_DIR, VERSION_DIR, FIG_DIR, MODEL_DIR]:
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


def parse_view(sequence_dir: Path) -> str:
    m = VIEW_RE.search(sequence_dir.name)
    return m.group(1).lower() if m else "unknown"


def parse_patient_id(sequence_dir: Path) -> int:
    # Expected: DATA_ROOT / "0001" / "0001-...._filtered_pose"
    patient_part = sequence_dir.parent.name
    digits = re.sub(r"\D", "", patient_part)
    return int(digits)


def sample_evenly(files: list[Path], max_count: int) -> list[Path]:
    if len(files) <= max_count:
        return files
    idx = np.linspace(0, len(files) - 1, max_count).round().astype(int)
    return [files[i] for i in idx]


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def basic_stats(values: list[float], prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
        return np.nan
    u = a - b
    v = c - b
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return np.nan
    cosv = float(np.dot(u, v) / (nu * nv))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))


def signed_angle_to_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    if np.any(~np.isfinite(top)) or np.any(~np.isfinite(bottom)):
        return np.nan
    v = top - bottom
    if np.linalg.norm(v) < 1e-6:
        return np.nan
    # image coordinates: y grows downward; vertical upright vector is [0, -1]
    return math.degrees(math.atan2(float(v[0]), float(-v[1])))


# =========================
# Data indexing
# =========================

def build_dataset_index(data_root: Path) -> pd.DataFrame:
    rows = []
    sequence_dirs = sorted({p.parent for p in data_root.rglob("frame_*.json")})
    for n, seq_dir in enumerate(sequence_dirs, start=1):
        files = sorted(seq_dir.glob("frame_*.json"))
        first = load_json(files[0]) if files else {}
        video_info = first.get("video_info", {}) if isinstance(first, dict) else {}
        rows.append(
            {
                "patient_id": parse_patient_id(seq_dir),
                "view": parse_view(seq_dir),
                "sequence_dir": str(seq_dir),
                "frame_count": len(files),
                "video_name": video_info.get("video_name", ""),
                "fps": video_info.get("fps", np.nan),
                "total_frames": video_info.get("total_frames", np.nan),
                "width": video_info.get("width", np.nan),
                "height": video_info.get("height", np.nan),
            }
        )
        if n % 200 == 0:
            log(f"Indexed {n}/{len(sequence_dirs)} sequence folders")
    return pd.DataFrame(rows)


def flatten_track1(track1: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in track1:
        row = {"patient_id": int(item["patient_id"])}
        for i in range(1, 18):
            row[f"L{i}"] = int(item["left"][str(i)])
            row[f"R{i}"] = int(item["right"][str(i)])
        row["Total"] = sum(row[f"L{i}"] + row[f"R{i}"] for i in range(1, 18))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def flatten_track2(track2: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in track2:
        rows.append(
            {
                "patient_id": int(item["patient_id"]),
                "Left_gait_subtype": item["left"]["gait_subtype"],
                "Right_gait_subtype": item["right"]["gait_subtype"],
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


# =========================
# Feature extraction
# =========================

def frame_measurements(frame_path: Path) -> dict[str, float]:
    data = load_json(frame_path)
    video_info = data.get("video_info", {})
    width = safe_float(video_info.get("width", 1920), 1920.0)
    height = safe_float(video_info.get("height", 1080), 1080.0)
    instances = data.get("instance_info", [])
    if not instances:
        return {}
    inst = instances[0]
    kp = np.asarray(inst.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(inst.get("keypoint_scores", []), dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < 23 or kp.shape[1] < 2:
        return {}
    if scores.shape[0] < kp.shape[0]:
        scores = np.ones(kp.shape[0], dtype=np.float32)

    bbox = inst.get("gt_bbox_xywh_px")
    if bbox and len(bbox) >= 4:
        bx, by, bw, bh = [safe_float(v) for v in bbox[:4]]
        if not np.isfinite(bw) or bw <= 0:
            bw = width
        if not np.isfinite(bh) or bh <= 0:
            bh = height
    else:
        bx, by, bw, bh = 0.0, 0.0, width, height

    norm = kp.copy()
    norm[:, 0] = (norm[:, 0] - bx) / max(bw, 1.0)
    norm[:, 1] = (norm[:, 1] - by) / max(bh, 1.0)

    out: dict[str, float] = {
        "bbox_w": bw / max(width, 1.0),
        "bbox_h": bh / max(height, 1.0),
        "bbox_area": (bw * bh) / max(width * height, 1.0),
        "body_score_mean": float(np.mean(scores[:23])),
        "body_score_min": float(np.min(scores[:23])),
        "body_low_score_rate": float(np.mean(scores[:23] < LOW_SCORE_THR)),
    }

    # Keep core normalized coordinates and scores.
    for idx, name in JOINT_NAMES.items():
        out[f"{name}_x"] = float(norm[idx, 0])
        out[f"{name}_y"] = float(norm[idx, 1])
        out[f"{name}_score"] = float(scores[idx])

    # Clinical angles and geometry.
    out["knee_angle_l"] = angle_deg(norm[11], norm[13], norm[15])
    out["knee_angle_r"] = angle_deg(norm[12], norm[14], norm[16])
    out["hip_angle_l"] = angle_deg(norm[5], norm[11], norm[13])
    out["hip_angle_r"] = angle_deg(norm[6], norm[12], norm[14])
    out["ankle_angle_l_bigtoe"] = angle_deg(norm[13], norm[15], norm[17])
    out["ankle_angle_r_bigtoe"] = angle_deg(norm[14], norm[16], norm[20])
    out["ankle_angle_l_heel"] = angle_deg(norm[13], norm[15], norm[19])
    out["ankle_angle_r_heel"] = angle_deg(norm[14], norm[16], norm[22])

    shoulder_mid = (norm[5] + norm[6]) / 2.0
    hip_mid = (norm[11] + norm[12]) / 2.0
    out["trunk_angle"] = signed_angle_to_vertical(shoulder_mid, hip_mid)
    out["pelvis_tilt_y"] = float(norm[11, 1] - norm[12, 1])
    out["shoulder_tilt_y"] = float(norm[5, 1] - norm[6, 1])
    out["hip_width"] = float(abs(norm[11, 0] - norm[12, 0]))
    out["shoulder_width"] = float(abs(norm[5, 0] - norm[6, 0]))

    # Foot clearance/contact proxies in image-normalized coordinates.
    for side, ankle, bigtoe, smalltoe, heel in [
        ("l", 15, 17, 18, 19),
        ("r", 16, 20, 21, 22),
    ]:
        foot_y = [norm[ankle, 1], norm[bigtoe, 1], norm[smalltoe, 1], norm[heel, 1]]
        foot_x = [norm[ankle, 0], norm[bigtoe, 0], norm[smalltoe, 0], norm[heel, 0]]
        out[f"foot_{side}_y_min"] = float(np.nanmin(foot_y))
        out[f"foot_{side}_y_max"] = float(np.nanmax(foot_y))
        out[f"foot_{side}_y_range"] = float(np.nanmax(foot_y) - np.nanmin(foot_y))
        out[f"foot_{side}_x_range"] = float(np.nanmax(foot_x) - np.nanmin(foot_x))
        out[f"toe_heel_y_diff_{side}"] = float(norm[bigtoe, 1] - norm[heel, 1])
        out[f"toe_heel_x_diff_{side}"] = float(norm[bigtoe, 0] - norm[heel, 0])

    return out


def sequence_features(sequence_dir: Path, max_frames: int) -> dict[str, Any]:
    files = sorted(sequence_dir.glob("frame_*.json"))
    chosen = sample_evenly(files, max_frames)
    frame_rows = []
    for fp in chosen:
        row = frame_measurements(fp)
        if row:
            frame_rows.append(row)

    base = {
        "patient_id": parse_patient_id(sequence_dir),
        "view": parse_view(sequence_dir),
        "sequence_dir": str(sequence_dir),
        "frame_count": len(files),
        "sampled_frames": len(chosen),
        "valid_sampled_frames": len(frame_rows),
    }
    if not frame_rows:
        return base

    keys = sorted(frame_rows[0].keys())
    for key in keys:
        values = [row.get(key, np.nan) for row in frame_rows]
        base.update(basic_stats(values, key))
        # Motion feature from sampled time series.
        arr = np.asarray(values, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            diffs = np.diff(arr)
            base[f"{key}_diff_abs_mean"] = float(np.mean(np.abs(diffs)))
            base[f"{key}_diff_std"] = float(np.std(diffs))
        else:
            base[f"{key}_diff_abs_mean"] = np.nan
            base[f"{key}_diff_std"] = np.nan

    return base


def build_sequence_features(index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(index_df)
    for i, row in enumerate(index_df.itertuples(index=False), start=1):
        seq_dir = Path(row.sequence_dir)
        rows.append(sequence_features(seq_dir, MAX_FRAMES_PER_SEQUENCE))
        if i % 50 == 0 or i == total:
            log(f"Extracted sequence features {i}/{total}")
    return pd.DataFrame(rows)


def build_patient_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in seq_df.columns
        if c not in {"patient_id", "view", "sequence_dir"} and pd.api.types.is_numeric_dtype(seq_df[c])
    ]
    parts = []
    for view in VIEWS:
        sub = seq_df[seq_df["view"] == view].copy()
        grouped = sub.groupby("patient_id")[numeric_cols].agg(["mean", "std", "min", "max"])
        grouped.columns = [f"{view}__{col}__{agg}" for col, agg in grouped.columns]
        parts.append(grouped)

    all_view = seq_df.groupby("patient_id")[numeric_cols].agg(["mean", "std", "min", "max"])
    all_view.columns = [f"all__{col}__{agg}" for col, agg in all_view.columns]
    parts.append(all_view)

    patient_df = pd.concat(parts, axis=1).reset_index()
    patient_df = patient_df.sort_values("patient_id").reset_index(drop=True)
    return patient_df


# =========================
# Metrics and models
# =========================

def make_binary_model(y: pd.Series) -> Any:
    if y.nunique() < 2:
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_multiclass_model(y: pd.Series) -> Any:
    if y.nunique() < 2:
        return DummyClassifier(strategy="most_frequent")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=SEED,
                ),
            ),
        ]
    )


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(len(X), dtype=np.float32)


def compute_s1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float((y_true == y_pred).mean())
    true_total = y_true.sum(axis=1)
    pred_total = y_pred.sum(axis=1)
    rmse = float(np.sqrt(np.mean((pred_total - true_total) ** 2)))
    score = float((acc + 1.0 - rmse / 34.0) / 2.0)
    return {"track": "track1", "accuracy": acc, "rmse": rmse, "score": score}


def compute_s2(left_true: np.ndarray, right_true: np.ndarray, left_pred: np.ndarray, right_pred: np.ndarray) -> dict[str, float]:
    acc = float(np.mean(np.concatenate([left_true == left_pred, right_true == right_pred])))
    f1_left = float(f1_score(left_true, left_pred, average="macro", zero_division=0))
    f1_right = float(f1_score(right_true, right_pred, average="macro", zero_division=0))
    f1 = (f1_left + f1_right) / 2.0
    score = (acc + f1) / 2.0
    return {
        "track": "track2",
        "accuracy": acc,
        "macro_f1_left": f1_left,
        "macro_f1_right": f1_right,
        "macro_f1_mean": f1,
        "score": score,
    }


def cv_track1(X: pd.DataFrame, y: pd.DataFrame, label_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    kfold = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=SEED)
    prob_oof = pd.DataFrame(index=y.index, columns=label_cols, dtype=float)
    pred_oof = pd.DataFrame(index=y.index, columns=label_cols, dtype=int)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        fold_pred = []
        fold_true = y.iloc[va_idx][label_cols].values.astype(int)
        for col in label_cols:
            model = make_binary_model(y.iloc[tr_idx][col])
            model.fit(X_tr, y.iloc[tr_idx][col])
            prob = positive_probability(model, X_va)
            pred = (prob >= 0.5).astype(int)
            prob_oof.loc[y.index[va_idx], col] = prob
            pred_oof.loc[y.index[va_idx], col] = pred
            fold_pred.append(pred)
        fold_pred_arr = np.vstack(fold_pred).T
        score = compute_s1(fold_true, fold_pred_arr)
        score["fold"] = fold
        fold_rows.append(score)
        log(f"Track1 fold {fold}: S1={score['score']:.5f} acc={score['accuracy']:.5f} rmse={score['rmse']:.5f}")

    overall = compute_s1(y[label_cols].values.astype(int), pred_oof[label_cols].values.astype(int))
    log(f"Track1 OOF: S1={overall['score']:.5f} acc={overall['accuracy']:.5f} rmse={overall['rmse']:.5f}")
    return prob_oof, pd.DataFrame(fold_rows), overall


def tune_track1_thresholds(y_true: pd.DataFrame, prob_oof: pd.DataFrame, label_cols: list[str]) -> dict[str, float]:
    thresholds = {}
    for col in label_cols:
        best_t = 0.5
        best_acc = -1.0
        yt = y_true[col].values.astype(int)
        prob = prob_oof[col].values.astype(float)
        for t in np.linspace(0.2, 0.8, 25):
            pred = (prob >= t).astype(int)
            acc = accuracy_score(yt, pred)
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)
        # Shrink slightly toward 0.5 to reduce CV overfit.
        thresholds[col] = float(0.75 * best_t + 0.25 * 0.5)
    tuned_pred = np.column_stack([(prob_oof[c].values.astype(float) >= thresholds[c]).astype(int) for c in label_cols])
    tuned_score = compute_s1(y_true[label_cols].values.astype(int), tuned_pred)
    log(
        f"Track1 OOF tuned thresholds: S1={tuned_score['score']:.5f} "
        f"acc={tuned_score['accuracy']:.5f} rmse={tuned_score['rmse']:.5f}"
    )
    return thresholds


def cv_track2(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    loo = LeaveOneOut()
    left_true, right_true, left_pred, right_pred, patient_ids = [], [], [], [], []
    for fold, (tr_idx, va_idx) in enumerate(loo.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        lm = make_multiclass_model(y.iloc[tr_idx]["Left_gait_subtype"])
        rm = make_multiclass_model(y.iloc[tr_idx]["Right_gait_subtype"])
        lm.fit(X_tr, y.iloc[tr_idx]["Left_gait_subtype"])
        rm.fit(X_tr, y.iloc[tr_idx]["Right_gait_subtype"])
        lp = lm.predict(X_va)[0]
        rp = rm.predict(X_va)[0]
        left_pred.append(lp)
        right_pred.append(rp)
        left_true.append(y.iloc[va_idx]["Left_gait_subtype"].values[0])
        right_true.append(y.iloc[va_idx]["Right_gait_subtype"].values[0])
        patient_ids.append(int(y.iloc[va_idx]["patient_id"].values[0]))
        if fold % 5 == 0 or fold == len(X):
            log(f"Track2 LOO progress {fold}/{len(X)}")

    pred_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "left_true": left_true,
            "right_true": right_true,
            "left_pred": left_pred,
            "right_pred": right_pred,
        }
    )
    score = compute_s2(
        pred_df["left_true"].values,
        pred_df["right_true"].values,
        pred_df["left_pred"].values,
        pred_df["right_pred"].values,
    )
    log(
        f"Track2 LOO: S2={score['score']:.5f} acc={score['accuracy']:.5f} "
        f"macroF1={score['macro_f1_mean']:.5f}"
    )
    return pred_df, score


def fit_track1_models(X: pd.DataFrame, y: pd.DataFrame, label_cols: list[str]) -> dict[str, Any]:
    models = {}
    for col in label_cols:
        model = make_binary_model(y[col])
        model.fit(X, y[col])
        models[col] = model
    return models


def fit_track2_models(X: pd.DataFrame, y: pd.DataFrame) -> dict[str, Any]:
    left = make_multiclass_model(y["Left_gait_subtype"])
    right = make_multiclass_model(y["Right_gait_subtype"])
    left.fit(X, y["Left_gait_subtype"])
    right.fit(X, y["Right_gait_subtype"])
    return {"left": left, "right": right}


def predict_track1(models: dict[str, Any], thresholds: dict[str, float], X: pd.DataFrame, label_cols: list[str]) -> np.ndarray:
    preds = []
    for col in label_cols:
        prob = positive_probability(models[col], X)
        preds.append((prob >= thresholds.get(col, 0.5)).astype(int))
    return np.vstack(preds).T


# =========================
# Submission and plots
# =========================

def build_submission(
    patient_features: pd.DataFrame,
    feature_cols: list[str],
    t1_models: dict[str, Any],
    t1_thresholds: dict[str, float],
    t2_models: dict[str, Any],
    label_cols: list[str],
) -> pd.DataFrame:
    rows = []
    patient_features = patient_features.set_index("patient_id", drop=False)

    X_t1 = patient_features.loc[T1_TEST_IDS, feature_cols]
    t1_pred = predict_track1(t1_models, t1_thresholds, X_t1, label_cols)
    for pid, pred in zip(T1_TEST_IDS, t1_pred):
        pred = pred.astype(int).tolist()
        left = pred[:17]
        right = pred[17:]
        rows.append([f"track1-{pid}", *left, *right, sum(left) + sum(right), -1, -1])

    X_t2 = patient_features.loc[T2_TEST_IDS, feature_cols]
    left_pred = t2_models["left"].predict(X_t2)
    right_pred = t2_models["right"].predict(X_t2)
    for pid, lp, rp in zip(T2_TEST_IDS, left_pred, right_pred):
        rows.append([f"track2-{pid}", *([-1] * 34), -1, lp, rp])

    sub = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    assert sub.shape == (len(T1_TEST_IDS) + len(T2_TEST_IDS), len(SUBMISSION_COLUMNS))
    return sub


def save_basic_plots(index_df: pd.DataFrame, track1_df: pd.DataFrame, track2_df: pd.DataFrame, cv_scores: pd.DataFrame) -> None:
    if plt is None:
        log("matplotlib unavailable; skipping plots")
        return

    view_counts = index_df["view"].value_counts().reindex(VIEWS)
    ax = view_counts.plot(kind="bar", title="Sequence count by view")
    ax.set_xlabel("view")
    ax.set_ylabel("sequence count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sequence_count_by_view.png", dpi=140)
    plt.close()

    ax = index_df["frame_count"].hist(bins=40)
    ax.set_title("Frame count per sequence")
    ax.set_xlabel("frames")
    ax.set_ylabel("sequences")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "frame_count_hist.png", dpi=140)
    plt.close()

    label_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
    pos_rates = track1_df[label_cols].mean().sort_values()
    ax = pos_rates.plot(kind="bar", figsize=(12, 4), title="Track 1 positive rate by label")
    ax.set_ylabel("positive rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "track1_label_positive_rates.png", dpi=140)
    plt.close()

    subtype_counts = pd.concat(
        [
            track2_df["Left_gait_subtype"].value_counts().rename("left"),
            track2_df["Right_gait_subtype"].value_counts().rename("right"),
        ],
        axis=1,
    ).fillna(0)
    ax = subtype_counts.plot(kind="bar", title="Track 2 subtype counts")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "track2_subtype_counts.png", dpi=140)
    plt.close()

    if not cv_scores.empty:
        ax = cv_scores.set_index("track")["score"].plot(kind="bar", title="OOF CV score")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "cv_scores.png", dpi=140)
        plt.close()


# =========================
# Main
# =========================

def main() -> None:
    start = time.time()
    seed_everything()
    ensure_dirs()

    section("CVPR 2026 Children Gait - Stage 1 Feature Baseline")
    log(f"RUN_ID={RUN_ID}")
    log(f"PATH1={PATH1}")
    log(f"PATH2={PATH2}")
    log(f"DATA_ROOT={DATA_ROOT}")
    log(f"WORK_DIR={WORK_DIR}")

    if not PATH1.exists() or not PATH2.exists() or not DATA_ROOT.exists():
        raise FileNotFoundError("One or more input paths do not exist. Check Kaggle inputs.")

    config = {
        "run_id": RUN_ID,
        "seed": SEED,
        "max_frames_per_sequence": MAX_FRAMES_PER_SEQUENCE,
        "low_score_thr": LOW_SCORE_THR,
        "path1": str(PATH1),
        "path2": str(PATH2),
        "data_root": str(DATA_ROOT),
        "table_dir": str(TABLE_DIR),
        "version_dir": str(VERSION_DIR),
    }
    dump_json(config, VERSION_DIR / "run_config.json")

    section("Load labels and index dataset")
    track1_raw = load_json(PATH1)
    track2_raw = load_json(PATH2)
    track1_df = flatten_track1(track1_raw)
    track2_df = flatten_track2(track2_raw)
    log(f"Track1 labels: {track1_df.shape}")
    log(f"Track2 labels: {track2_df.shape}")
    log(f"Track2 subtype counts left={track2_df['Left_gait_subtype'].value_counts().to_dict()}")
    log(f"Track2 subtype counts right={track2_df['Right_gait_subtype'].value_counts().to_dict()}")

    index_df = build_dataset_index(DATA_ROOT)
    log(f"Dataset index shape: {index_df.shape}")
    log(f"Patients={index_df['patient_id'].nunique()} sequences={len(index_df)} frames={int(index_df['frame_count'].sum())}")
    log(f"View counts={index_df['view'].value_counts().to_dict()}")

    index_df.to_csv(TABLE_DIR / "dataset_index.csv", index=False)
    track1_df.to_csv(TABLE_DIR / "track1_train_flat.csv", index=False)
    track2_df.to_csv(TABLE_DIR / "track2_train_flat.csv", index=False)

    section("Extract sampled sequence features")
    seq_df = build_sequence_features(index_df)
    seq_df.to_csv(TABLE_DIR / "sequence_features.csv", index=False)
    log(f"Sequence feature table: {seq_df.shape}")

    patient_features = build_patient_features(seq_df)
    patient_features.to_csv(TABLE_DIR / "patient_features.csv", index=False)
    log(f"Patient feature table: {patient_features.shape}")

    feature_cols = [c for c in patient_features.columns if c != "patient_id"]
    with (VERSION_DIR / "feature_columns.txt").open("w", encoding="utf-8") as f:
        for c in feature_cols:
            f.write(c + "\n")

    section("Train/CV Track 1")
    label_cols = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
    t1_train = track1_df.merge(patient_features, on="patient_id", how="inner")
    X1 = t1_train[feature_cols]
    y1 = t1_train[["patient_id"] + label_cols].copy()
    log(f"Track1 merged train: X={X1.shape}, y={y1[label_cols].shape}")
    t1_prob_oof, t1_fold_scores, t1_score = cv_track1(X1, y1, label_cols)
    t1_thresholds = tune_track1_thresholds(y1, t1_prob_oof, label_cols)
    t1_prob_oof.insert(0, "patient_id", y1["patient_id"].values)
    t1_prob_oof.to_csv(TABLE_DIR / "track1_oof_probabilities.csv", index=False)
    t1_fold_scores.to_csv(TABLE_DIR / "track1_fold_scores.csv", index=False)
    dump_json(t1_thresholds, VERSION_DIR / "track1_thresholds.json")

    section("Train/CV Track 2")
    t2_train = track2_df.merge(patient_features, on="patient_id", how="inner")
    X2 = t2_train[feature_cols]
    y2 = t2_train[["patient_id", "Left_gait_subtype", "Right_gait_subtype"]].copy()
    log(f"Track2 merged train: X={X2.shape}, y={y2.shape}")
    t2_oof, t2_score = cv_track2(X2, y2)
    t2_oof.to_csv(TABLE_DIR / "track2_oof_predictions.csv", index=False)

    section("Fit final models and create submission")
    t1_models = fit_track1_models(X1, y1, label_cols)
    t2_models = fit_track2_models(X2, y2)
    if joblib is not None:
        joblib.dump(t1_models, MODEL_DIR / "track1_models.joblib")
        joblib.dump(t2_models, MODEL_DIR / "track2_models.joblib")
        log(f"Saved models to {MODEL_DIR}")
    else:
        log("joblib unavailable; models not saved")

    submission = build_submission(patient_features, feature_cols, t1_models, t1_thresholds, t2_models, label_cols)
    submission_path = TABLE_DIR / "submission_stage1.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    log(f"Saved submission to {submission_path}")
    log(f"Saved Kaggle submission copy to {WORK_DIR / 'submission.csv'}")
    print(submission.head(25).to_string(index=False))

    cv_scores = pd.DataFrame([t1_score, t2_score])
    cv_scores.to_csv(TABLE_DIR / "cv_scores.csv", index=False)
    save_basic_plots(index_df, track1_df, track2_df, cv_scores)

    elapsed = time.time() - start
    summary = {
        "run_id": RUN_ID,
        "elapsed_seconds": elapsed,
        "dataset": {
            "patients": int(index_df["patient_id"].nunique()),
            "sequences": int(len(index_df)),
            "frames": int(index_df["frame_count"].sum()),
            "view_counts": {str(k): int(v) for k, v in index_df["view"].value_counts().to_dict().items()},
        },
        "tables": {
            "dataset_index": str(TABLE_DIR / "dataset_index.csv"),
            "sequence_features": str(TABLE_DIR / "sequence_features.csv"),
            "patient_features": str(TABLE_DIR / "patient_features.csv"),
            "submission_stage1": str(submission_path),
            "submission_copy": str(WORK_DIR / "submission.csv"),
        },
        "cv": {
            "track1": t1_score,
            "track2": t2_score,
            "mean_proxy": float((t1_score["score"] + t2_score["score"]) / 2.0),
        },
        "notes": [
            "This is a patient-level sampled-feature baseline, not the final model.",
            "Use generated patient_features.csv for faster GPU experiments.",
            "Validation is patient-level only.",
        ],
    }
    dump_json(summary, VERSION_DIR / "run_summary.json")
    log(f"Saved run summary to {VERSION_DIR / 'run_summary.json'}")
    log(f"Elapsed minutes: {elapsed / 60:.2f}")
    section("Done")

    gc.collect()


if __name__ == "__main__":
    main()
