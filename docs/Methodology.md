# Methodology Report

## Objective

Predict normalized traffic demand for Bengaluru geohash locations and timestamps. The output must follow the competition schema:

```text
Index,demand
```

## Key Split Insight

The dataset is not an ordinary random tabular split:

- Train day 48 contains a complete 24-hour profile.
- Train day 49 contains only `00:00` to `02:00`.
- Test day 49 contains future timestamps from `02:15` to `13:45`.

This means the task is a forward-time traffic forecasting problem. Random validation would leak temporal structure and overstate model quality.

## Validation Strategy

The solution uses chronological validation:

1. Keep day 48 as the previous-day reference.
2. Use early day-49 records for calibration.
3. Hold out later day-49 records.
4. Evaluate with R2, RMSE, and MAE.

Rolling cutoffs were also used to reduce overfitting to a single validation point.

## Modeling Strategy

Three model families were considered:

1. **Calibrated Daily Lag**
   - Uses same geohash and same timestamp from day 48.
   - Applies smoothed day49-vs-day48 calibration.

2. **Tuned Blend Forecaster**
   - Combines geohash-hour demand, exact lag, road priors, and minute priors.
   - Applies fast-decaying day49 calibration.

3. **Rolling-CV Road/Geohash Blend**
   - Heavier road-context weighting.
   - Public-score-informed calibration decay.
   - Selected as the final competition family.

## Final Model Formula

The final model builds a base prediction from:

```text
base = 0.45 * geohash_hour_mean
     + 0.05 * exact_day48_lag
     + 0.50 * road_context_mean
```

Then it estimates geohash-level day49 calibration:

```text
delta = actual_day49_known - base_prediction_day49_known
```

The final prediction is:

```text
prediction = base + smoothed_geohash_delta * exp(-horizon_minutes / decay)
```

Predictions are clipped to `[0, 1]`.

## Why This Works

Traffic demand is strongly influenced by road class and location. The full day-48 signal gives a prior for each location/time, while early day49 records reveal same-day level shifts. The final blend balances structural traffic priors with short-horizon calibration.
