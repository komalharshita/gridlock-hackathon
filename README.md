# Gridlock Hackathon 2.0 - Traffic Demand Prediction

Team: **APIcalypse Now**

## Project Overview

This repository contains a complete hackathon-ready solution for the Gridlock Hackathon 2.0 traffic demand prediction challenge. The objective is to forecast normalized traffic demand for Bengaluru geohash locations across future time slots using road attributes, weather information, temporal patterns, and historical demand behavior.

The final solution uses a leaderboard-informed, time-aware forecasting strategy. The strongest signal is the revealed chronology of the data: day 48 contains a full daily traffic pattern, day 49 contains the first two hours of labels, and the test set asks for future day-49 demand. The model combines road-context priors, geohash-hour behavior, exact previous-day lag, and calibrated day-49 uplift.

## Team Roles

| Member | Role | Responsibility |
|---|---|---|
| Komal Harshita | Data & EDA Lead | Dataset audit, missing-value analysis, target distribution, traffic-pattern exploration |
| Dipshikha Soni| Feature Engineering Lead | Time features, geohash/location statistics, road-context features, calibration features |
| Simran Sethi | Modeling & Validation Lead | Chronological validation, model comparison, leaderboard-informed tuning |
| Abhijna | Documentation & Delivery Lead | Notebook, reports, presentation, reproducibility, final packaging |

All team members share equal responsibility for final solution quality, experimentation, review, and submission decisions.

## Repository Structure

```text
gridlock-hackathon-apicalypse-now/
├── data/
│   ├── README.md
│   └── sample_submission.csv
├── docs/
│   ├── Assumptions_and_Limitations.md
│   ├── Data_Dictionary.md
│   ├── Feature_Engineering.md
│   ├── Methodology.md
│   └── Model_Comparison.md
├── notebooks/
│   └── Gridlock_Hackathon_End_to_End.ipynb
├── outputs/
│   └── submission.csv
├── presentation/
│   └── APIcalypse_Now_Gridlock_Hackathon.pptx
├── reports/
│   └── PROJECT_STRUCTURE.md
├── src/
│   └── gridlock_solution.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone or unzip this repository.
2. Place the competition files in the `data/` directory:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
3. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the solution:

```bash
python src/gridlock_solution.py --data-dir data --output-dir outputs
```

5. Upload `outputs/submission.csv` to the competition platform.

## Methodology Summary

- Used chronological validation instead of random validation.
- Treated `Index` as an identifier only, never as a model feature.
- Engineered temporal, road-context, geohash, lag, and calibration features.
- Compared calibrated daily lag, tuned blend forecaster, and rolling-CV road/geohash blend.
- Selected the road/geohash blend family based on public leaderboard feedback and rolling chronological validation.

## Results Summary

Best observed public leaderboard score during iteration:

- `submission_rolling_cv_road_geo_9272_local.csv`: **88.92648**

Internal validation metrics for the final model family:

| Model | Chronological R2 | RMSE | MAE |
|---|---:|---:|---:|
| rolling_cv_road_geo_blend | 0.92721 | 0.03846 | 0.02809 |
| tuned_blend_forecaster | 0.91671 | 0.04114 | 0.02982 |
| calibrated_daily_lag | 0.84007 | 0.05700 | 0.03906 |

## Reproducibility

The core solution is deterministic and uses fixed parameters. To reproduce:

```bash
python src/gridlock_solution.py --data-dir data --output-dir outputs
```

The notebook provides a full walkthrough from loading data through submission generation.

## Notes

Competition train/test data is ignored by `.gitignore` to avoid accidental redistribution. Only the sample submission and generated prediction file are included.
