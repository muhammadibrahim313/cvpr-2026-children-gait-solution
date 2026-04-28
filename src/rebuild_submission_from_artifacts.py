from pathlib import Path

import joblib
import numpy as np
import pandas as pd


WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_working")
MODEL_ROOTS = [
    WORK_DIR,
    Path("/kaggle/input/datasets/qasminumber/cvpr-v2"),
    Path("/kaggle/input/cvpr-2026-children-gait-models"),
]

T1_TEST_IDS = [4, 5, 18, 26, 28, 40, 42, 43, 47, 48, 53, 54, 72, 78, 83, 85]
T2_TEST_IDS = [4, 6, 7, 13, 26, 35, 39, 42, 50]
LABEL_COLS = [f"L{i}" for i in range(1, 18)] + [f"R{i}" for i in range(1, 18)]
SUBMISSION_COLUMNS = (
    ["ID"]
    + [f"L{i}" for i in range(1, 18)]
    + [f"R{i}" for i in range(1, 18)]
    + ["Total", "Left_gait_subtype", "Right_gait_subtype"]
)


def find_bundle() -> tuple[Path, Path]:
    candidates = []
    for root in MODEL_ROOTS:
        if root.exists():
            candidates.extend(sorted(root.glob("versioning/stage3_cgc_*/models/track1_stage3_cgc.joblib")))
    if not candidates:
        raise FileNotFoundError("stage3 Track 1 model bundle not found")
    model_path = candidates[-1]
    run_id = model_path.parents[1].name
    root = model_path.parents[3]
    feature_path = root / "tables" / run_id / "stage3_patient_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)
    return model_path, feature_path


def positive_proba(model, X: np.ndarray) -> np.ndarray:
    cols = []
    for est in model.estimators_:
        proba = est.predict_proba(X)
        classes = list(getattr(est, "classes_", []))
        if 1 in classes:
            cols.append(proba[:, classes.index(1)])
        elif len(classes) == 1:
            cols.append(np.ones(len(X)) if int(classes[0]) == 1 else np.zeros(len(X)))
        else:
            cols.append(np.zeros(len(X)))
    return np.vstack(cols).T.astype(np.float32)


def predict_track1(bundle: dict, features: pd.DataFrame, shift: float = 0.0) -> np.ndarray:
    X_raw = np.array(
        [[features.loc[pid].get(name, 0.0) for name in bundle["feature_names"]] for pid in T1_TEST_IDS],
        dtype=np.float32,
    )
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = bundle["scaler"].transform(bundle["vt"].transform(X_raw))
    prob = np.zeros((len(X), len(LABEL_COLS)), dtype=np.float32)
    for _, model in bundle["models"]:
        prob += positive_proba(model, X) / len(bundle["models"])
    threshold = np.clip(bundle["thresholds"] + shift, 0.05, 0.95)
    return (prob >= threshold[None, :]).astype(int)


def track2_rows() -> pd.DataFrame:
    predictions = {
        4: ("type3", "type3"),
        6: ("type1", "type1"),
        7: ("type3", "type3"),
        13: ("type1", "type1"),
        26: ("type1", "type1"),
        35: ("type1", "type1"),
        39: ("type1", "type3"),
        42: ("type3", "type3"),
        50: ("type1", "type1"),
    }
    rows = []
    for pid in T2_TEST_IDS:
        left, right = predictions[pid]
        rows.append([f"track2-{pid}", *([-1] * 34), -1, left, right])
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def build_submission(pred: np.ndarray, out_path: Path) -> pd.DataFrame:
    rows = []
    for pid, labels in zip(T1_TEST_IDS, pred.astype(int)):
        labels = labels.tolist()
        rows.append([f"track1-{pid}", *labels[:17], *labels[17:], int(sum(labels)), -1, -1])
    t1 = pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)
    out = pd.concat([t1, track2_rows()], ignore_index=True)[SUBMISSION_COLUMNS]
    out.to_csv(out_path, index=False)
    return out


def main() -> None:
    model_path, feature_path = find_bundle()
    print(f"model={model_path}")
    print(f"features={feature_path}")
    bundle = joblib.load(model_path)
    features = pd.read_csv(feature_path).set_index("patient_id", drop=False)
    out = build_submission(predict_track1(bundle, features, shift=0.0), WORK_DIR / "submission.csv")
    out.to_csv(WORK_DIR / "submission_stage3_cgc_t1_stage2t2.csv", index=False)
    print(out.to_string(index=False))
    print(f"saved={WORK_DIR / 'submission.csv'}")


if __name__ == "__main__":
    main()

