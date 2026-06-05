# Model Comparison

## Evaluation Metric

The competition scoring expression appears to use R2:

```text
score = max(0, 100 * r2_score(actual, predicted))
```

The local evaluation reports:

- R2
- RMSE
- MAE

## Compared Models

| Model | Strength | Risk |
|---|---|---|
| Global mean | Simple baseline | Cannot capture road/location variance |
| Road-context mean | Strong structural prior | Misses time/location dynamics |
| Exact daily lag | Captures repeated location-time pattern | Can be noisy and overfit exact slots |
| Calibrated daily lag | Adds day49 correction | Early-day correction may not generalize all morning |
| Tuned blend forecaster | Balances lag and priors | Needs careful decay tuning |
| Rolling-CV road/geohash blend | Best practical public performance | Still limited without true future labels |

## Internal Validation

| Model | R2 | RMSE | MAE |
|---|---:|---:|---:|
| rolling_cv_road_geo_blend | 0.92721 | 0.03846 | 0.02809 |
| tuned_blend_forecaster | 0.91671 | 0.04114 | 0.02982 |
| calibrated_daily_lag | 0.84007 | 0.05700 | 0.03906 |

## Public Leaderboard Feedback

Submitted variants showed the road/geohash blend family outperformed the earlier tuned blend:

| Submission | Public Score |
|---|---:|
| tuned_blend_9167 | 86.86874 |
| road50_geo45_lag05_decay120 | 88.07249 |
| road50_geo45_lag05_decay240 | 88.55802 |
| road50_geo45_lag05_decay360 | 88.81985 |
| rolling_cv_road_geo_9272_local | 88.92648 |

## Final Recommendation

Use the rolling-CV road/geohash blend as the final packaged solution. For further improvement, install LightGBM/CatBoost and train a residual model on top of the current prediction features.
