"""Stage 2 pose-tensor TCN model for CVPR 2026 AI for Children.

Run this in Kaggle with GPU enabled (P100 is enough). It builds fixed-length
multi-view pose tensors directly from JSON, trains compact PyTorch models, and
creates submission candidates. It does not overwrite your current best unless
you submit one of the new CSVs manually.

Expected inputs:
- /kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track1_train.json
- /kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track2_train.json
- /kaggle/input/datasets/qasminumber/cvpr-2026-data/dataset

Main outputs:
- /kaggle/working/tables/<RUN_ID>/pose_tensor_T*.npz
- /kaggle/working/tables/<RUN_ID>/stage2_oof_*.csv
- /kaggle/working/versioning/<RUN_ID>/run_summary.json
- /kaggle/working/submission_stage2_pose_tcn.csv
- /kaggle/working/submission_stage2_t1_stage1best_t2.csv
- /kaggle/working/submission_stage1best_t1_stage2_t2.csv
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
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# Config
# =========================

SEED = 123
T = 128
VIEWS = ["backward", "forward", "left", "right"]
N_JOINTS = 23
N_CHANNELS = 6  # bbox x/y, hip-centered x/y, score, mask
LOW_SCORE_THR = 0.20

EPOCHS_T1 = 70
EPOCHS_T2 = 90
BATCH_SIZE_T1 = 16
BATCH_SIZE_T2 = 8
LR = 2e-3
WEIGHT_DECAY = 2e-3
DROPOUT = 0.35

KAGGLE_TRACK1 = Path("/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track1_train.json")
KAGGLE_TRACK2 = Path("/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge/track2_train.json")
KAGGLE_DATA_ROOT = Path("/kaggle/input/datasets/qasminumber/cvpr-2026-data/dataset")

PATH1 = KAGGLE_TRACK1
PATH2 = KAGGLE_TRACK2
DATA_ROOT = KAGGLE_DATA_ROOT

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_working")
RUN_ID = datetime.now(UTC).strftime("stage2_%Y%m%d_%H%M%S")
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
CLASS_NAMES = ["WNL", "type1", "type2", "type3", "type4"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}

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


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for d in [TABLE_DIR, VERSION_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_view(sequence_dir: Path) -> str:
    m = VIEW_RE.search(sequence_dir.name)
    return m.group(1).lower() if m else "unknown"


def parse_patient_id(sequence_dir: Path) -> int:
    digits = re.sub(r"\D", "", sequence_dir.parent.name)
    return int(digits)


def sample_evenly(items: list[Path], n: int) -> list[Path]:
    if not items:
        return []
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def flatten_track1(track1: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in track1:
        row = {"patient_id": int(item["patient_id"])}
        for i in range(1, 18):
            row[f"L{i}"] = int(item["left"][str(i)])
            row[f"R{i}"] = int(item["right"][str(i)])
        row["Total"] = sum(row[c] for c in LABEL_COLS)
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
                "left_id": CLASS_TO_ID[item["left"]["gait_subtype"]],
                "right_id": CLASS_TO_ID[item["right"]["gait_subtype"]],
            }
        )
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


def build_index(data_root: Path) -> pd.DataFrame:
    rows = []
    sequence_dirs = sorted({p.parent for p in data_root.rglob("frame_*.json")})
    for i, seq_dir in enumerate(sequence_dirs, start=1):
        files = sorted(seq_dir.glob("frame_*.json"))
        first = load_json(files[0]) if files else {}
        vi = first.get("video_info", {}) if isinstance(first, dict) else {}
        rows.append(
            {
                "patient_id": parse_patient_id(seq_dir),
                "view": parse_view(seq_dir),
                "sequence_dir": str(seq_dir),
                "frame_count": len(files),
                "fps": vi.get("fps", np.nan),
                "total_frames": vi.get("total_frames", np.nan),
                "video_name": vi.get("video_name", ""),
            }
        )
        if i % 250 == 0:
            log(f"Indexed {i}/{len(sequence_dirs)} sequence dirs")
    return pd.DataFrame(rows)


# =========================
# Tensor building
# =========================

def frame_to_tensor(path: Path) -> np.ndarray:
    data = load_json(path)
    vi = data.get("video_info", {})
    width = float(vi.get("width", 1920))
    height = float(vi.get("height", 1080))
    insts = data.get("instance_info", [])
    out = np.zeros((N_JOINTS, N_CHANNELS), dtype=np.float32)
    if not insts:
        return out

    inst = insts[0]
    kp = np.asarray(inst.get("keypoints", []), dtype=np.float32)
    scores = np.asarray(inst.get("keypoint_scores", []), dtype=np.float32)
    if kp.ndim != 2 or kp.shape[0] < N_JOINTS or kp.shape[1] < 2:
        return out
    if scores.shape[0] < kp.shape[0]:
        scores = np.ones((kp.shape[0],), dtype=np.float32)

    bbox = inst.get("gt_bbox_xywh_px")
    if bbox and len(bbox) >= 4:
        bx, by, bw, bh = [float(v) for v in bbox[:4]]
        if bw <= 1 or bh <= 1:
            bx, by, bw, bh = 0.0, 0.0, width, height
    else:
        bx, by, bw, bh = 0.0, 0.0, width, height

    coords = kp[:N_JOINTS, :2].copy()
    bbox_xy = coords.copy()
    bbox_xy[:, 0] = (bbox_xy[:, 0] - bx) / max(bw, 1.0)
    bbox_xy[:, 1] = (bbox_xy[:, 1] - by) / max(bh, 1.0)

    hip_mid = (bbox_xy[11] + bbox_xy[12]) / 2.0
    centered = bbox_xy - hip_mid[None, :]
    scale = max(np.linalg.norm(bbox_xy[5] - bbox_xy[11]), np.linalg.norm(bbox_xy[6] - bbox_xy[12]), 0.05)
    centered = centered / scale

    sc = np.clip(scores[:N_JOINTS], 0.0, 1.5)
    mask = (sc >= LOW_SCORE_THR).astype(np.float32)

    out[:, 0:2] = bbox_xy
    out[:, 2:4] = centered
    out[:, 4] = sc
    out[:, 5] = mask
    return out


def collect_view_files(index_df: pd.DataFrame, patient_id: int, view: str) -> list[Path]:
    sub = index_df[(index_df["patient_id"] == patient_id) & (index_df["view"] == view)]
    all_files: list[Path] = []
    for seq_dir in sub["sequence_dir"].tolist():
        all_files.extend(sorted(Path(seq_dir).glob("frame_*.json")))
    return all_files


def build_pose_tensor(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cache_path = TABLE_DIR / f"pose_tensor_T{T}.npz"
    if cache_path.exists():
        log(f"Loading cached tensor: {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["patient_ids"]

    patient_ids = np.array(sorted(index_df["patient_id"].unique()), dtype=np.int32)
    X = np.zeros((len(patient_ids), len(VIEWS), T, N_JOINTS, N_CHANNELS), dtype=np.float16)
    for pi, pid in enumerate(patient_ids):
        for vi, view in enumerate(VIEWS):
            files = collect_view_files(index_df, int(pid), view)
            chosen = sample_evenly(files, T)
            if not chosen:
                continue
            frames = [frame_to_tensor(p) for p in chosen]
            # If fewer than T, pad by repeating the last valid frame.
            while len(frames) < T:
                frames.append(frames[-1].copy())
            X[pi, vi] = np.stack(frames[:T]).astype(np.float16)
        if (pi + 1) % 10 == 0 or pi + 1 == len(patient_ids):
            log(f"Built pose tensor for {pi + 1}/{len(patient_ids)} patients")

    np.savez_compressed(cache_path, X=X, patient_ids=patient_ids)
    log(f"Saved tensor: {cache_path} shape={X.shape}")
    return X, patient_ids


# =========================
# Models
# =========================

class PoseEncoder(nn.Module):
    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        in_ch = N_JOINTS * N_CHANNELS
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,V,T,J,C
        b, v, t, j, c = x.shape
        x = x.reshape(b * v, t, j * c).permute(0, 2, 1)
        h = self.net(x)
        avg = h.mean(dim=-1)
        mx = h.amax(dim=-1)
        h = torch.cat([avg, mx], dim=1).reshape(b, v, -1)
        return h.reshape(b, -1)


class Track1Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PoseEncoder()
        self.head = nn.Sequential(
            nn.Linear(len(VIEWS) * 192 * 2, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(384, 34),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class Track2Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PoseEncoder(dropout=0.45)
        self.shared = nn.Sequential(
            nn.Linear(len(VIEWS) * 192 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.45),
        )
        self.left = nn.Linear(256, len(CLASS_NAMES))
        self.right = nn.Linear(256, len(CLASS_NAMES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(self.encoder(x))
        return self.left(h), self.right(h)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(X: np.ndarray, y: np.ndarray | list[np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    tx = torch.tensor(X, dtype=torch.float32)
    if isinstance(y, list):
        tensors = [tx] + [torch.tensor(v) for v in y]
    else:
        tensors = [tx, torch.tensor(y)]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def s1_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc = float((y_true == y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_pred.sum(axis=1) - y_true.sum(axis=1)) ** 2)))
    score = float((acc + 1.0 - rmse / 34.0) / 2.0)
    return {"accuracy": acc, "rmse": rmse, "score": score}


def s2_metric(lt: np.ndarray, rt: np.ndarray, lp: np.ndarray, rp: np.ndarray) -> dict[str, float]:
    acc = float(np.mean(np.concatenate([lt == lp, rt == rp])))
    f1_l = float(f1_score(lt, lp, average="macro", zero_division=0))
    f1_r = float(f1_score(rt, rp, average="macro", zero_division=0))
    f1 = (f1_l + f1_r) / 2.0
    return {"accuracy": acc, "macro_f1": f1, "score": (acc + f1) / 2.0}


def tune_thresholds(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    thresholds = np.full(prob.shape[1], 0.5, dtype=np.float32)
    for c in range(prob.shape[1]):
        best_t, best_acc = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 25):
            acc = accuracy_score(y_true[:, c], (prob[:, c] >= t).astype(int))
            if acc > best_acc:
                best_t, best_acc = float(t), float(acc)
        thresholds[c] = 0.75 * best_t + 0.25 * 0.5
    return thresholds


# =========================
# Training
# =========================

def train_track1_fold(X_train, y_train, X_valid, y_valid, fold: int) -> tuple[np.ndarray, Track1Net]:
    dev = device()
    model = Track1Net().to(dev)
    pos = np.clip(y_train.sum(axis=0), 1, None)
    neg = np.clip(len(y_train) - y_train.sum(axis=0), 1, None)
    pos_weight = torch.tensor(np.clip(neg / pos, 0.5, 6.0), dtype=torch.float32, device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(X_train, y_train.astype(np.float32), BATCH_SIZE_T1, True)
    valid_loader = make_loader(X_valid, y_valid.astype(np.float32), BATCH_SIZE_T1, False)

    best_score = -999.0
    best_state = None
    best_prob = None

    for epoch in range(1, EPOCHS_T1 + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            # Soft total loss helps RMSE without forcing hard labels.
            loss = loss + 0.025 * ((torch.sigmoid(logits).sum(dim=1) - yb.sum(dim=1)) ** 2).mean() / 34.0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch % 5 == 0 or epoch == EPOCHS_T1:
            model.eval()
            probs = []
            with torch.no_grad():
                for xb, _ in valid_loader:
                    logits = model(xb.to(dev))
                    probs.append(torch.sigmoid(logits).cpu().numpy())
            prob = np.vstack(probs)
            pred = (prob >= 0.5).astype(int)
            score = s1_metric(y_valid.astype(int), pred)["score"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_prob = prob.copy()
        if epoch in [1, 20, 40, 60, EPOCHS_T1]:
            log(f"T1 fold {fold} epoch {epoch}/{EPOCHS_T1} loss={np.mean(losses):.5f} best_s1={best_score:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_prob if best_prob is not None else np.zeros_like(y_valid, dtype=np.float32), model


def train_track1_oof(X_all: np.ndarray, patient_ids: np.ndarray, track1_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray]:
    pid_to_idx = {int(pid): i for i, pid in enumerate(patient_ids)}
    train_pids = track1_df["patient_id"].astype(int).tolist()
    idx = np.array([pid_to_idx[p] for p in train_pids], dtype=int)
    X = X_all[idx]
    y = track1_df[LABEL_COLS].values.astype(np.float32)

    oof = np.zeros_like(y, dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        prob, _ = train_track1_fold(X[tr], y[tr], X[va], y[va], fold)
        oof[va] = prob
        fold_pred = (prob >= 0.5).astype(int)
        score = s1_metric(y[va].astype(int), fold_pred)
        log(f"T1 fold {fold} final 0.5: S1={score['score']:.5f} acc={score['accuracy']:.5f} rmse={score['rmse']:.5f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    thresholds = tune_thresholds(y.astype(int), oof)
    tuned = (oof >= thresholds[None, :]).astype(int)
    score = s1_metric(y.astype(int), tuned)
    log(f"T1 OOF tuned: S1={score['score']:.5f} acc={score['accuracy']:.5f} rmse={score['rmse']:.5f}")
    return oof, y, score, thresholds


def train_track1_final(X_all: np.ndarray, patient_ids: np.ndarray, track1_df: pd.DataFrame) -> Track1Net:
    pid_to_idx = {int(pid): i for i, pid in enumerate(patient_ids)}
    idx = np.array([pid_to_idx[int(p)] for p in track1_df["patient_id"]], dtype=int)
    X = X_all[idx]
    y = track1_df[LABEL_COLS].values.astype(np.float32)
    # Train final with all data. Use validation equal to train just to reuse function.
    _, model = train_track1_fold(X, y, X, y, fold=99)
    return model


def class_weights(values: np.ndarray) -> torch.Tensor:
    counts = np.bincount(values, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1, None)
    weights = weights / weights.mean()
    return torch.tensor(np.clip(weights, 0.5, 5.0), dtype=torch.float32)


def train_track2_fold(X_train, yl_train, yr_train, X_valid, yl_valid, yr_valid, fold: int) -> tuple[np.ndarray, np.ndarray, Track2Net]:
    dev = device()
    model = Track2Net().to(dev)
    ce_l = nn.CrossEntropyLoss(weight=class_weights(yl_train).to(dev))
    ce_r = nn.CrossEntropyLoss(weight=class_weights(yr_train).to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(X_train, [yl_train.astype(np.int64), yr_train.astype(np.int64)], BATCH_SIZE_T2, True)
    valid_loader = make_loader(X_valid, [yl_valid.astype(np.int64), yr_valid.astype(np.int64)], BATCH_SIZE_T2, False)

    best_score = -999.0
    best_state = None
    best_l = None
    best_r = None
    for epoch in range(1, EPOCHS_T2 + 1):
        model.train()
        losses = []
        for xb, ylb, yrb in train_loader:
            xb = xb.to(dev)
            ylb = ylb.to(dev)
            yrb = yrb.to(dev)
            opt.zero_grad(set_to_none=True)
            log_l, log_r = model(xb)
            loss = ce_l(log_l, ylb) + ce_r(log_r, yrb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch % 5 == 0 or epoch == EPOCHS_T2:
            model.eval()
            lp, rp = [], []
            with torch.no_grad():
                for xb, _, _ in valid_loader:
                    log_l, log_r = model(xb.to(dev))
                    lp.append(torch.softmax(log_l, dim=1).cpu().numpy())
                    rp.append(torch.softmax(log_r, dim=1).cpu().numpy())
            lp_arr = np.vstack(lp)
            rp_arr = np.vstack(rp)
            pred_l = lp_arr.argmax(axis=1)
            pred_r = rp_arr.argmax(axis=1)
            score = s2_metric(yl_valid, yr_valid, pred_l, pred_r)["score"]
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_l = lp_arr.copy()
                best_r = rp_arr.copy()
        if epoch in [1, 30, 60, EPOCHS_T2]:
            log(f"T2 fold {fold} epoch {epoch}/{EPOCHS_T2} loss={np.mean(losses):.5f} best_s2={best_score:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_l, best_r, model


def train_track2_oof(X_all: np.ndarray, patient_ids: np.ndarray, track2_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    pid_to_idx = {int(pid): i for i, pid in enumerate(patient_ids)}
    idx = np.array([pid_to_idx[int(p)] for p in track2_df["patient_id"]], dtype=int)
    X = X_all[idx]
    yl = track2_df["left_id"].values.astype(np.int64)
    yr = track2_df["right_id"].values.astype(np.int64)

    oof_l = np.zeros((len(X), len(CLASS_NAMES)), dtype=np.float32)
    oof_r = np.zeros((len(X), len(CLASS_NAMES)), dtype=np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        pl, pr, _ = train_track2_fold(X[tr], yl[tr], yr[tr], X[va], yl[va], yr[va], fold)
        oof_l[va] = pl
        oof_r[va] = pr
        fold_score = s2_metric(yl[va], yr[va], pl.argmax(axis=1), pr.argmax(axis=1))
        log(f"T2 fold {fold}: S2={fold_score['score']:.5f} acc={fold_score['accuracy']:.5f} f1={fold_score['macro_f1']:.5f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    score = s2_metric(yl, yr, oof_l.argmax(axis=1), oof_r.argmax(axis=1))
    log(f"T2 OOF: S2={score['score']:.5f} acc={score['accuracy']:.5f} f1={score['macro_f1']:.5f}")
    return oof_l, oof_r, yl, yr, score


def train_track2_final(X_all: np.ndarray, patient_ids: np.ndarray, track2_df: pd.DataFrame) -> Track2Net:
    pid_to_idx = {int(pid): i for i, pid in enumerate(patient_ids)}
    idx = np.array([pid_to_idx[int(p)] for p in track2_df["patient_id"]], dtype=int)
    X = X_all[idx]
    yl = track2_df["left_id"].values.astype(np.int64)
    yr = track2_df["right_id"].values.astype(np.int64)
    _, _, model = train_track2_fold(X, yl, yr, X, yl, yr, fold=99)
    return model


def predict_track1(model: Track1Net, X: np.ndarray, thresholds: np.ndarray, shift: float = 0.0) -> np.ndarray:
    dev = device()
    model.eval()
    loader = make_loader(X, np.zeros((len(X), 34), dtype=np.float32), BATCH_SIZE_T1, False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
    prob = np.vstack(probs)
    th = np.clip(thresholds + shift, 0.03, 0.97)
    return (prob >= th[None, :]).astype(int)


def predict_track2(model: Track2Net, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dev = device()
    model.eval()
    loader = make_loader(X, [np.zeros(len(X), dtype=np.int64), np.zeros(len(X), dtype=np.int64)], BATCH_SIZE_T2, False)
    lp, rp = [], []
    with torch.no_grad():
        for xb, _, _ in loader:
            log_l, log_r = model(xb.to(dev))
            lp.append(torch.softmax(log_l, dim=1).cpu().numpy())
            rp.append(torch.softmax(log_r, dim=1).cpu().numpy())
    l = np.vstack(lp).argmax(axis=1)
    r = np.vstack(rp).argmax(axis=1)
    return l, r


def make_submission(t1_pred: np.ndarray, t2_left_ids: np.ndarray, t2_right_ids: np.ndarray, out_path: Path) -> pd.DataFrame:
    rows = []
    for pid, labels in zip(T1_TEST_IDS, t1_pred.astype(int)):
        labels = labels.tolist()
        rows.append([f"track1-{pid}", *labels[:17], *labels[17:], int(sum(labels)), -1, -1])
    for pid, l, r in zip(T2_TEST_IDS, t2_left_ids, t2_right_ids):
        rows.append([f"track2-{pid}", *([-1] * 34), -1, ID_TO_CLASS[int(l)], ID_TO_CLASS[int(r)]])
    df = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    assert df.shape == (25, 38)
    df.to_csv(out_path, index=False)
    log(f"Wrote {out_path}")
    print(df.to_string(index=False), flush=True)
    return df


def main() -> None:
    start = time.time()
    seed_everything()
    ensure_dirs()
    section("Stage 2 Pose TCN")
    log(f"RUN_ID={RUN_ID}")
    log(f"device={device()}")
    log(f"PATH1={PATH1}")
    log(f"PATH2={PATH2}")
    log(f"DATA_ROOT={DATA_ROOT}")

    config = {
        "run_id": RUN_ID,
        "seed": SEED,
        "T": T,
        "views": VIEWS,
        "n_joints": N_JOINTS,
        "n_channels": N_CHANNELS,
        "epochs_t1": EPOCHS_T1,
        "epochs_t2": EPOCHS_T2,
        "batch_size_t1": BATCH_SIZE_T1,
        "batch_size_t2": BATCH_SIZE_T2,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
    }
    dump_json(config, VERSION_DIR / "run_config.json")

    section("Load labels and index")
    track1_df = flatten_track1(load_json(PATH1))
    track2_df = flatten_track2(load_json(PATH2))
    index_df = build_index(DATA_ROOT)
    index_df.to_csv(TABLE_DIR / "dataset_index.csv", index=False)
    track1_df.to_csv(TABLE_DIR / "track1_flat.csv", index=False)
    track2_df.to_csv(TABLE_DIR / "track2_flat.csv", index=False)
    log(f"index={index_df.shape}, patients={index_df.patient_id.nunique()}, frames={int(index_df.frame_count.sum())}")

    section("Build pose tensor")
    X_all, patient_ids = build_pose_tensor(index_df)
    log(f"X_all={X_all.shape}, dtype={X_all.dtype}")

    section("Track 1 OOF and final")
    oof_t1, y1, t1_score, thresholds = train_track1_oof(X_all, patient_ids, track1_df)
    pd.DataFrame(oof_t1, columns=LABEL_COLS).assign(patient_id=track1_df["patient_id"].values).to_csv(TABLE_DIR / "stage2_track1_oof_prob.csv", index=False)
    pd.DataFrame({"label": LABEL_COLS, "threshold": thresholds}).to_csv(TABLE_DIR / "stage2_track1_thresholds.csv", index=False)
    t1_model = train_track1_final(X_all, patient_ids, track1_df)
    torch.save(t1_model.state_dict(), MODEL_DIR / "track1_tcn.pt")

    section("Track 2 OOF and final")
    oof_l, oof_r, yl, yr, t2_score = train_track2_oof(X_all, patient_ids, track2_df)
    t2_oof_df = pd.DataFrame(
        {
            "patient_id": track2_df["patient_id"].values,
            "left_true": [ID_TO_CLASS[int(v)] for v in yl],
            "right_true": [ID_TO_CLASS[int(v)] for v in yr],
            "left_pred": [ID_TO_CLASS[int(v)] for v in oof_l.argmax(axis=1)],
            "right_pred": [ID_TO_CLASS[int(v)] for v in oof_r.argmax(axis=1)],
        }
    )
    t2_oof_df.to_csv(TABLE_DIR / "stage2_track2_oof_pred.csv", index=False)
    t2_model = train_track2_final(X_all, patient_ids, track2_df)
    torch.save(t2_model.state_dict(), MODEL_DIR / "track2_tcn.pt")

    section("Create submissions")
    pid_to_idx = {int(pid): i for i, pid in enumerate(patient_ids)}
    idx_t1 = np.array([pid_to_idx[p] for p in T1_TEST_IDS], dtype=int)
    idx_t2 = np.array([pid_to_idx[p] for p in T2_TEST_IDS], dtype=int)
    t1_pred = predict_track1(t1_model, X_all[idx_t1], thresholds, shift=0.0)
    t2_l, t2_r = predict_track2(t2_model, X_all[idx_t2])
    stage2 = make_submission(t1_pred, t2_l, t2_r, WORK_DIR / "submission_stage2_pose_tcn.csv")

    # Hybrid candidates using current known best Stage 1 threshold file if present.
    best_stage1_path = WORK_DIR / "submission_stage1_t1_lesspos_0125.csv"
    if best_stage1_path.exists():
        best_stage1 = pd.read_csv(best_stage1_path)
        hybrid1 = stage2.copy()
        # Use Stage 1 best Track 2, Stage 2 Track 1.
        hybrid1.loc[hybrid1["ID"].str.startswith("track2-"), :] = best_stage1.loc[best_stage1["ID"].str.startswith("track2-"), :].values
        hybrid1.to_csv(WORK_DIR / "submission_stage2_t1_stage1best_t2.csv", index=False)
        log(f"Wrote {WORK_DIR / 'submission_stage2_t1_stage1best_t2.csv'}")

        hybrid2 = best_stage1.copy()
        # Use Stage 1 best Track 1, Stage 2 Track 2.
        hybrid2.loc[hybrid2["ID"].str.startswith("track2-"), :] = stage2.loc[stage2["ID"].str.startswith("track2-"), :].values
        hybrid2.to_csv(WORK_DIR / "submission_stage1best_t1_stage2_t2.csv", index=False)
        log(f"Wrote {WORK_DIR / 'submission_stage1best_t1_stage2_t2.csv'}")
    else:
        log("No Stage 1 best file found; skipped hybrid submissions")

    summary = {
        "run_id": RUN_ID,
        "elapsed_minutes": (time.time() - start) / 60.0,
        "device": str(device()),
        "dataset": {
            "patients": int(index_df.patient_id.nunique()),
            "sequences": int(len(index_df)),
            "frames": int(index_df.frame_count.sum()),
        },
        "cv": {
            "track1": t1_score,
            "track2": t2_score,
            "mean_proxy": float((t1_score["score"] + t2_score["score"]) / 2.0),
        },
        "outputs": {
            "stage2": str(WORK_DIR / "submission_stage2_pose_tcn.csv"),
            "stage2_t1_stage1best_t2": str(WORK_DIR / "submission_stage2_t1_stage1best_t2.csv"),
            "stage1best_t1_stage2_t2": str(WORK_DIR / "submission_stage1best_t1_stage2_t2.csv"),
        },
    }
    dump_json(summary, VERSION_DIR / "run_summary.json")
    log(f"Saved summary: {VERSION_DIR / 'run_summary.json'}")
    log(f"Elapsed minutes: {(time.time() - start) / 60.0:.2f}")
    section("Done")


if __name__ == "__main__":
    main()
