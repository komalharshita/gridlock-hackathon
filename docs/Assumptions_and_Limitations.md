# Assumptions and Limitations

## Assumptions

- The target `demand` is normalized to the range `[0, 1]`.
- The evaluation metric is R2-based.
- Day48 is a valid previous-day reference for day49 forecasting.
- Known early day49 labels can be used for same-day calibration.
- Road attributes and weather fields are available for both train and test at prediction time.

## Limitations

- Public leaderboard feedback suggests local validation can be optimistic.
- The final model is tuned for the specific revealed time split.
- The solution avoids using `Index`, but if `Index` encodes hidden ordering beyond row identity, that signal is intentionally ignored to reduce leakage risk.
- Full train/test files are not included in the GitHub package by default.
- Advanced gradient boosting models may improve performance but require additional packages and tuning time.

## Future Improvements

- Train LightGBM/CatBoost residual models using the current predictions as strong base features.
- Add spatial neighborhood features between nearby geohashes.
- Use recursive forecast calibration by timestamp blocks.
- Blend public-score-informed variants with constrained optimization.
- Build a robust private/public split simulator if more leaderboard feedback is available.
