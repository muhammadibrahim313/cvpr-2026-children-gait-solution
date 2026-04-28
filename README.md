# CVPR 2026 Children Gait Solution

Repository for the Kaggle competition **[CVPR 2026] The First AI for Children Challenge**.

Selected public leaderboard submission:

```text
submission_stage3_cgc_t1_stage2t2.csv
public score: 0.53536
```

## Method Summary

The final system is a two-part ensemble:

1. Track 1 EVGS prediction uses patient-level pose features and a multi-output tree ensemble.
2. Track 2 gait subtype prediction uses the strongest saved pose-TCN Track 2 output from our previous run.

The Track 1 model extracts multi-view clinical gait features from pose JSON files, including joint angles, trunk and pelvis posture, foot progression proxies, cadence proxies, frequency-domain ankle motion, symmetry, and stance/swing summaries. A Gradient Boosting, Extra Trees, and Random Forest ensemble predicts the 34 EVGS labels. Per-label thresholds are tuned on repeated patient-level cross-validation.

The selected Stage 3 Track 1 cross-validation result was:

```text
Track1 OOF S1 = 0.82619
accuracy = 0.77409
RMSE = 4.13856
```

## Repository Layout

```text
src/
  stage1_feature_baseline.py
  stage2_pose_tcn.py
  stage3_feature_ensemble.py
  rebuild_submission_from_artifacts.py
  prepare_previous_artifacts.py
  run_all_kaggle.py

notebooks/
  cvpr_2026_children_gait_solution.ipynb

models/
  README.md

outputs/
  README.md

docs/
  experiment_notes.md
  public_release_note.md
```

## Kaggle Inputs

The notebook expects these Kaggle inputs:

```text
/kaggle/input/competitions/cvpr-2026-the-first-ai-children-challenge
/kaggle/input/datasets/qasminumber/cvpr-2026-data
```

For rebuilding the exact selected CSV from artifacts, add the public model-artifact dataset after publishing it. Expected structure:

```text
versioning/stage3_cgc_*/models/track1_stage3_cgc.joblib
tables/stage3_cgc_*/stage3_patient_features.csv
```

## Reproduce Training Submission

Run in a Kaggle notebook:

```python
exec(open("/kaggle/input/<repo-or-code-dataset>/src/stage3_feature_ensemble.py").read())
```

This writes:

```text
/kaggle/working/submission_stage3_cgc_t1_stage2t2.csv
/kaggle/working/submission_stage3_cgc_t1_stage2t2_lesspos005.csv
/kaggle/working/submission_stage3_cgc_t1_stage2t2_morepos005.csv
```

The selected file is:

```text
submission_stage3_cgc_t1_stage2t2.csv
```

## Rebuild From Artifacts

After publishing model artifacts as a Kaggle dataset:

```python
exec(open("/kaggle/input/<repo-or-code-dataset>/src/rebuild_submission_from_artifacts.py").read())
```

This writes:

```text
/kaggle/working/submission.csv
/kaggle/working/submission_stage3_cgc_t1_stage2t2.csv
```

## Environment

The code was developed for the Kaggle Python image.

Main packages:

```text
numpy
pandas
scipy
scikit-learn
torch
joblib
matplotlib
```

## Notes

- Raw competition data is not included in this repository.
- Large model artifacts are intended to be shared through a public Kaggle dataset or release asset.
- The public code link should be posted in the Kaggle discussion comment section as requested by the host.

