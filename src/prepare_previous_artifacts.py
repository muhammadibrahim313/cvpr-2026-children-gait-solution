"""Prepare V1 notebook artifacts for follow-up Kaggle scripts.

Use this in a new Kaggle notebook when V1 has been added as an input:

    V1_ROOT = "/kaggle/input/notebooks/qasminumber/cvpr-2026-v1"
    exec(open("kaggle_prepare_v1_artifacts.py").read())

It copies the required V1 output tree from `/kaggle/input/...` into
`/kaggle/working/...`, preserving the paths expected by later scripts.
"""

from __future__ import annotations

import shutil
from pathlib import Path


V1_ROOT = Path(globals().get("V1_ROOT", "/kaggle/input/notebooks/qasminumber/cvpr-2026-v1"))
WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_working")

REQUIRED = [
    "submission_stage1best_t1_stage2_t2.csv",
    "submission_stage1_t1_lesspos_0125.csv",
    "submission_stage2_pose_tcn.csv",
    "tables/stage1_20260425_120648/patient_features.csv",
    "tables/stage1_20260425_120648/track1_oof_probabilities.csv",
    "tables/stage1_20260425_120648/track1_train_flat.csv",
    "versioning/stage1_20260425_120648/feature_columns.txt",
    "versioning/stage1_20260425_120648/track1_thresholds.json",
    "versioning/stage1_20260425_120648/models/track1_models.joblib",
    "versioning/stage1_20260425_120648/run_summary.json",
    "tables/stage2_20260425_125020/pose_tensor_T128.npz",
    "tables/stage2_20260425_125020/stage2_track2_oof_pred.csv",
    "versioning/stage2_20260425_125020/models/track2_tcn.pt",
    "versioning/stage2_20260425_125020/run_summary.json",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def copy_one(rel: str) -> bool:
    src = V1_ROOT / rel
    dst = WORK_DIR / rel
    if not src.exists():
        log(f"MISSING: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f"copied: {rel}")
    return True


def main() -> None:
    log("=" * 80)
    log("Prepare V1 artifacts")
    log("=" * 80)
    log(f"V1_ROOT={V1_ROOT}")
    log(f"WORK_DIR={WORK_DIR}")
    if not V1_ROOT.exists():
        raise FileNotFoundError(f"V1 input root not found: {V1_ROOT}")

    ok = [copy_one(rel) for rel in REQUIRED]

    best_src = WORK_DIR / "submission_stage1best_t1_stage2_t2.csv"
    if best_src.exists():
        shutil.copy2(best_src, WORK_DIR / "submission.csv")
        log("restored current best to /kaggle/working/submission.csv")

    log("=" * 80)
    log(f"Ready: {sum(ok)}/{len(ok)} required artifacts copied.")
    if not all(ok):
        log("Some optional/follow-up artifacts are missing, but Track 1 corrected peak probe may still work if Stage 1 artifacts copied.")
    log("Next: run kaggle_stage1_peak_probe_corrected_stage2t2.py")


if __name__ == "__main__":
    main()
