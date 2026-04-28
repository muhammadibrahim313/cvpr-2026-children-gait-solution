# Experiment Notes

## Starting Point

The first stable baseline used sampled pose features and independent logistic models for Track 1. Public score was around `0.486`.

The first large improvement came from combining:

- Stage 1 Track 1 thresholded feature model;
- Stage 2 pose-TCN Track 2 predictions.

That hybrid reached:

```text
0.53195
```

## Stage 3 Change

The final improvement focused on Track 1. Track 2 was left unchanged because previous Track 2 attempts showed the pose-TCN Track 2 rows were already the strongest component.

Stage 3 replaced the Track 1 logistic model with a clinical pose feature ensemble:

- joint angle summaries;
- trunk, pelvis, shoulder posture summaries;
- foot and toe-heel proxies;
- ankle/knee velocity and acceleration;
- cadence and FFT motion summaries;
- left/right symmetry;
- stance/swing proxies from ankle velocity;
- repeated patient-level cross-validation.

The Stage 3 Track 1 cross-validation score was:

```text
S1 = 0.82619
accuracy = 0.77409
RMSE = 4.13856
```

The selected public submission was:

```text
submission_stage3_cgc_t1_stage2t2.csv
public score = 0.53536
```

## Ablation Notes

Threshold probes around the selected Stage 3 file:

```text
base       0.53536
more +0.05 0.53527
less +0.05 0.52935
```

The base threshold was kept selected.

