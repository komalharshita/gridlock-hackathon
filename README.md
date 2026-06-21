# Gridlock Hackathon 2.0 - Traffic Demand Prediction

Team: **APIcalypse Now**

## Project Overview

This repository contains a complete hackathon-ready solution for the Gridlock Hackathon 2.0 traffic demand prediction challenge. The objective is to forecast normalized traffic demand for Bengaluru geohash locations across future time slots using road attributes, weather information, temporal patterns, and historical demand behavior.

The final solution uses a leaderboard-informed, time-aware forecasting strategy. The strongest signal is the revealed chronology of the data: day 48 contains a full daily traffic pattern, day 49 contains the first two hours of labels, and the test set asks for future day-49 demand. The model combines road-context priors, geohash-hour behavior, exact previous-day lag, and calibrated day-49 uplift.

## Phase 2 Prototype: AI Traffic Command Center

For the event-driven congestion round, the Streamlit prototype extends the traffic model into a decision-support workflow for Bengaluru Traffic Police:

- planned event approval mode for rallies, processions, public events, VIP movement, and construction,
- unplanned incident mode for accidents, breakdowns, debris, tree falls, waterlogging, and sudden congestion,
- operational risk adjustment using preparation lead time, blocked lanes, weather watch, and peak-hour context,
- visual traffic mood indicator and situation chips for fast severity reading during demos,
- ready-made scenario presets for planned rallies, rainy accidents, and VIP movement,
- resource allocation recommendations for officers, barricades, patrol vehicles, medical support, surveillance, and public advisory needs,
- diversion scenario comparison with estimated delay savings,
- downloadable command brief with risk drivers, resource plan, and response timeline.

Run the prototype with:

```bash
streamlit run app.py
```

See `docs/Phase2_Command_Center_Strategy.md` for the demo script and pitch positioning.

## Prototype Submission & Judging Criteria Mapping

The **Gridlock Sentinel Command Center** prototype has been engineered to directly address the hackathon's core judging criteria:

### 1. Technical Rigor (Algorithm & Modeling Depth)
* **GBDT Predictive Risk Engine**: Combines the offline machine learning model (trained on historical Bengaluru event records) with real-time operational risk adjustments (blocked lanes, rain watch, peak hour, preparation lead time) to predict situational severity.
* **SHAP Explainability**: Integrates SHAP TreeExplainer values directly into the "Event Digital Twin" UI, allowing command operators to see exactly which features (Time of Day, Event Type, Location, etc.) are driving the risk prediction.
* **Real-Street Dijkstra Routing Engine**: Leverages a Python `networkx` spatial road graph of Bengaluru (20 main nodes, 25 connecting arterial edges) to calculate congested travel times vs. optimal diversion detours when an incident occurs.
* **Signal Preemption Scheduler**: Auto-snaps hospital coordinates to the road network, calculating emergency corridor ETA windows and generating a detailed green-light preemption override schedule for traffic police controllers.
* **Police Proximity Allocator**: Solves a proximity-based resource allocation problem to dispatch officers and vehicles from nearby stations under capacity constraints.

### 2. Visual Quality & User Experience (CoreUI Aesthetics)
* **CoreUI Flat Dark Skin**: Built using a premium flat dark palette (App background: `#181924`, Cards: `#222437`, Sidebar: `#1d2030`) utilizing modern typography (Inter, IBM Plex Mono) and consistent components.
* **Zero-Emoji Professional Standard**: Completely replaced all emojis with high-quality Icons8 SVG/PNG markers and icons to deliver an enterprise-grade visual experience.
* **Dynamic, Responsive Reruns**: Removed all button-click gating. The dashboard, metrics, Plotly sparkline charts, and interactive map update in real-time instantly as the operator changes sidebar sliders.
* **Dynamic Hospital Recommendation**: Automatically computes the closest medical facility based on geocoded input location and sets it as the default, simplifying operator workflows.

### 3. Real-World Impact & Operational Feasibility
* **BMTC Commuter Transit Advisor**: Scans routes disrupted by the incident zone, calculates detour delay times, and suggests shifted boarding stop coordinates to prevent commuter isolation.
* **Downloadable Command Brief**: Generates a standardized markdown report (`gridlock_command_brief.md`) containing the complete scenario data, risk drivers, dispatched resource plans, and emergency timelines for offline sharing.

### 4. Demo Link & Setup Stability
* **Demo URL**: [https://gridlock-sentinel.streamlit.app/](https://gridlock-sentinel.streamlit.app/)
* **Hybrid Map Loading (Zero-Failure)**: The map automatically uses the **Mappls (MapmyIndia) Vector Map SDK** if the `MAPPLS_API_KEY` is present in secrets. If the key is not set, it instantly falls back to a custom **Folium Dark Matter** map, preventing script load freezes and providing a flawless evaluation flow.

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
