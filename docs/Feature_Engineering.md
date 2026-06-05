# Feature Engineering

## Temporal Features

- `minute`: parsed timestamp as minutes from midnight.
- `hour`: hour of day.
- `quarter`: 15-minute bucket inside each hour.
- `slot`: 15-minute slot index.
- `sin_time`, `cos_time`: cyclic representation of daily time.

## Lag Features

- `lag_demand`: day48 demand at the same geohash and timestamp.
- `geo_hour`: day48 mean demand for the same geohash and hour.
- `minute_mean`: global demand mean by minute.
- `hour_mean`: global demand mean by hour.

## Road and Context Features

- Road type priors.
- Number of lanes.
- Large vehicle permission.
- Landmark indicator.
- Weather and temperature.
- Group-level target statistics for road-context combinations.

## Geohash Features

- Geohash-level historical demand mean.
- Geohash-hour demand mean.
- Optional decoded latitude/longitude in the script for richer models.

## Calibration Features

- Per-geohash day49 residual between known early labels and base predictions.
- Smoothed global fallback for sparse geohashes.
- Exponential decay by forecast horizon.

## Leakage Controls

- `Index` is never used as a predictive feature.
- Validation statistics are fit only on the training side of each chronological split.
- Test labels are never used.
