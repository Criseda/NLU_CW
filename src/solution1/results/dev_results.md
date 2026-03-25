# Final Dev Set Evaluation Results

## Visualisations

### Confusion Matrix
![Confusion Matrix](./plots/confusion_matrix.png)

### Calibration Curve
![Calibration Curve](./plots/calibration_curve.png)

## Performance Metrics

### Core Metrics

| Metric            |    Score |   95% CI Limit (Lower) |   95% CI Limit (Upper) |
|:------------------|---------:|-----------------------:|-----------------------:|
| accuracy          | 0.708493 |               0.69781  |               0.72034  |
| balanced_accuracy | 0.709217 |               0.698156 |               0.721304 |
| f1_macro          | 0.708347 |               0.697523 |               0.72016  |
| f1_weighted       | 0.708218 |               0.697498 |               0.72001  |
| precision         | 0.7335   |               0.716149 |               0.749651 |
| recall            | 0.672775 |               0.656977 |               0.689054 |
| roc_auc           | 0.793202 |               0.782748 |               0.804077 |
| brier_score       | 0.185912 |               0.181234 |               0.190428 |

### Base Model Comparison

| Model | accuracy | f1_macro | roc_auc |
|:---|---:|---:|---:|
| a1_logreg | 0.627732 | 0.624731 | 0.688719 |
| a2_lgbm | 0.707659 | 0.707368 | 0.793548 |
| stacked_meta | 0.708493 | 0.708347 | 0.793202 |
