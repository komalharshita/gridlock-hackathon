# Data Dictionary

| Column | Type | Description | Usage |
|---|---|---|---|
| `Index` | Integer | Unique row identifier | Submission ID only; not used as a feature |
| `geohash` | String | Encoded geographic area | Location grouping, lag, geohash-hour priors |
| `day` | Integer | Recorded day number | Chronological split and calibration |
| `timestamp` | String | Time of day | Parsed into minute/hour/slot features |
| `demand` | Float | Traffic demand target | Prediction target |
| `RoadType` | Categorical | Nearby road type | Strong road-context prior |
| `NumberofLanes` | Integer | Number of nearby lanes | Road capacity signal |
| `LargeVehicles` | Categorical | Whether large vehicles are allowed | Mobility context |
| `Landmarks` | Categorical | Landmark presence | Activity/demand context |
| `Temperature` | Float | Local temperature | Weather/context feature |
| `Weather` | Categorical | Weather condition | Context feature |
