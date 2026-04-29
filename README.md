# CVPR 2026 Children Gait Solution

End-to-end Kaggle notebook for **[CVPR 2026] The First AI for Children Challenge**.

Selected public submission:

```text
submission_stage3_cgc_t1_stage2t2.csv
public score: 0.54177
```

## What Is Included

```text
notebooks/cvpr_2026_children_gait_end_to_end.ipynb
requirements.txt
LICENSE
```

The notebook contains the full pipeline:

1. Load competition labels and raw pose JSON files.
2. Train a compact pose-TCN model for Track 2 gait subtype prediction.
3. Extract clinical gait features for Track 1.
4. Train a Gradient Boosting / Extra Trees / Random Forest Track 1 ensemble.
5. Combine Stage 3 Track 1 with Stage 2 Track 2.
6. Save model artifacts and submission CSVs to `/kaggle/working`.

## Required Kaggle Inputs

Attach these two datasets to the Kaggle notebook:

```text
/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge
/kaggle/input/datasets/ibrahimqasimi/cvpr-2026-main-data
```

Only these two inputs are required.

## How To Run

Open:

```text
notebooks/cvpr_2026_children_gait_end_to_end.ipynb
```

Run all cells from top to bottom.

Recommended accelerator:

```text
T4 GPU
```

The Stage 2 pose model uses PyTorch. The Stage 3 feature ensemble is CPU-based and can take roughly two hours.

## Main Output

The selected submission is written to:

```text
/kaggle/working/submission_stage3_cgc_t1_stage2t2.csv
```

The notebook also writes model artifacts under:

```text
/kaggle/working/versioning/
/kaggle/working/tables/
```

## Final Method

Track 2 uses a compact temporal convolution model trained on fixed-length multi-view pose tensors.

Track 1 uses patient-level clinical pose features. These features summarize joint angles, trunk posture, foot motion, cadence proxies, frequency-domain ankle motion, symmetry, and stance/swing behavior across all available views. The final classifier is an ensemble of Gradient Boosting, Extra Trees, and Random Forest models with per-label threshold tuning.

Final Stage 3 Track 1 validation:

```text
S1 = 0.82619
accuracy = 0.77409
RMSE = 4.13856
```

Public leaderboard result:

```text
0.54177
```

## Artifact Publishing

Public artifact dataset:

```text
https://www.kaggle.com/datasets/ibrahimqasimi/cvpr-2026-children-gait-solution-artifacts
```

This dataset contains the generated model files, feature tables, run summaries, and selected submission CSV.

After running the notebook, the important generated artifacts from `/kaggle/working` are:

```text
versioning/stage3_cgc_*/models/track1_stage3_cgc.joblib
tables/stage3_cgc_*/stage3_patient_features.csv
versioning/stage2_*/models/track2_tcn.pt
submission_stage3_cgc_t1_stage2t2.csv
```

Useful optional files to include in the same artifact dataset:

```text
versioning/stage3_cgc_*/run_config.json
versioning/stage3_cgc_*/run_summary.json
tables/stage3_cgc_*/stage3_track1_oof_prob.csv
tables/stage3_cgc_*/stage3_track1_thresholds.csv
tables/stage2_*/stage2_track2_oof_pred.csv
```

Raw competition data is not included in this repository.
