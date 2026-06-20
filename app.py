import shap
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))
import traffic_network as tn

# Configure Page
st.set_page_config(
    page_title="Gridlock Sentinel",
    page_icon="https://img.icons8.com/color/48/traffic-light.png",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    src = Path("src")
    clf = joblib.load(src / "risk_model.pkl")
    encoders = joblib.load(src / "risk_encoders.pkl")
    return clf, encoders

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_events.csv")

clf, encoders = load_model()
df = load_data()
G = tn.create_bengaluru_graph()

# Inject Modern CSS for clean, premium styling (no emojis)
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 13px;
        color: #6c757d;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #212529;
    }
    .alert-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }
    .custom-icon {
        vertical-align: middle;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# CORE FUNCTIONS
# -------------------------------------------------------
def predict_event_risk(event_cause, crowd_size, zone, hour, day_of_week):
    if crowd_size < 2000:
        crowd_level = "Low"
    elif crowd_size < 10000:
        crowd_level = "Medium"
    else:
        crowd_level = "High"

    is_weekend = day_of_week in ["Saturday", "Sunday"]
    feature_cols = ["event_cause", "crowd_proxy", "zone", "hour", "day_of_week", "is_weekend"]

    row = pd.DataFrame([{
        "event_cause": encoders["event_cause"].transform([event_cause])[0]
            if event_cause in encoders["event_cause"].classes_ else 0,
        "crowd_proxy": encoders["crowd_proxy"].transform([crowd_level])[0]
            if crowd_level in encoders["crowd_proxy"].classes_ else 0,
        "zone": encoders["zone"].transform([zone])[0]
            if zone in encoders["zone"].classes_ else 0,
        "hour": hour,
        "day_of_week": encoders["day_of_week"].transform([day_of_week])[0]
            if day_of_week in encoders["day_of_week"].classes_ else 0,
        "is_weekend": int(is_weekend),
    }])[feature_cols]

    pred = clf.predict(row)[0]
    proba = clf.predict_proba(row)[0]
    proba_dict = dict(zip(clf.classes_, proba.round(3)))

    risk_score = int(
        proba_dict.get("Minor", 0) * 20 +
        proba_dict.get("Moderate", 0) * 60 +
        proba_dict.get("Severe", 0) * 100
    )

    return {
        "risk_level": pred,
        "risk_score": risk_score,
        "confidence": proba_dict,
        "crowd_level": crowd_level,
    }

def explain_prediction(event_cause, crowd_size, zone, hour, day_of_week, crowd_level):
    is_weekend = day_of_week in ["Saturday", "Sunday"]
    feature_cols = ["event_cause", "crowd_proxy", "zone", "hour", "day_of_week", "is_weekend"]

    row = pd.DataFrame([{
        "event_cause": encoders["event_cause"].transform([event_cause])[0]
            if event_cause in encoders["event_cause"].classes_ else 0,
        "crowd_proxy": encoders["crowd_proxy"].transform([crowd_level])[0]
            if crowd_level in encoders["crowd_proxy"].classes_ else 0,
        "zone": encoders["zone"].transform([zone])[0]
            if zone in encoders["zone"].classes_ else 0,
        "hour": hour,
        "day_of_week": encoders["day_of_week"].transform([day_of_week])[0]
            if day_of_week in encoders["day_of_week"].classes_ else 0,
        "is_weekend": int(is_weekend),
    }])[feature_cols]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(row)
    pred_idx = list(clf.classes_).index(clf.predict(row)[0])

    if isinstance(shap_values, list):
        values = np.array(shap_values[pred_idx][0]).flatten()
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            values = shap_values[0, :, pred_idx].flatten()
        elif shap_values.ndim == 2:
            values = shap_values[0].flatten()
        else:
            values = shap_values.flatten()

    readable_names = {
        "event_cause": "Event Type",
        "crowd_proxy": "Crowd Size",
        "zone": "Zone/Location",
        "hour": "Time of Day",
        "day_of_week": "Day of Week",
        "is_weekend": "Weekend?",
    }
    result = pd.DataFrame({
        "Factor": [readable_names[c] for c in feature_cols],
        "Impact": values
    }).sort_values("Impact", key=abs, ascending=False)
    return result

def get_past_similar_events(event_cause, zone, limit=5):
    similar = df[df["event_cause"] == event_cause].copy()
    if zone != "Unknown":
        zone_match = similar[similar["zone"] == zone]
        if len(zone_match) >= 2:
            similar = zone_match

    similar = similar[["event_cause", "address", "zone", "risk_level",
                        "hour", "day_of_week", "crowd_proxy"]].dropna(subset=["address"])
    similar = similar[~similar["address"].str.contains("ಲಿಂಕ್|ರಸ್ತೆ", na=False)]
    return similar.head(limit).reset_index(drop=True)

def predict_duration(event_cause):
    defaults = {
        "public_event": 180, "procession": 150, "protest": 120,
        "vip_movement": 60,  "accident": 45,    "vehicle_breakdown": 30,
        "congestion": 90,    "construction": 240,"tree_fall": 60,
        "water_logging": 120,"pot_holes": 30,    "road_conditions": 60,
        "Debris": 45,        "debris": 45,       "others": 60,
        "Fog / Low Visibility": 90, "test_demo": 30,
    }
    known = df[(df["event_cause"] == event_cause) & df["resolution_minutes"].notna()]
    if len(known) >= 3:
        median = known["resolution_minutes"].median()
        min_t = known["resolution_minutes"].min()
        max_t = known["resolution_minutes"].max()
        return int(median), int(min_t), int(max_t), len(known)
    else:
        default = defaults.get(event_cause, 60)
        return default, int(default * 0.5), int(default * 2), 0

def recommend_resources(crowd_size, risk_level, event_cause):
    multiplier = {"Minor": 1.0, "Moderate": 1.3, "Severe": 1.7}.get(risk_level, 1.0)
    base = {
        "Traffic Officers":      max(2,  crowd_size // 500),
        "Barricades":            max(2,  crowd_size // 1200),
        "Patrol Vehicles":       max(1,  crowd_size // 3000),
        "Ambulances on Standby": max(1,  crowd_size // 5000),
        "Fire Brigade Units":    1 if crowd_size > 5000 or risk_level == "Severe" else 0,
        "CCTV/Surveillance":     max(2,  crowd_size // 4000),
        "First Aid Posts":       max(1,  crowd_size // 8000),
        "Bus Route Diversions":  max(1,  crowd_size // 6000),
        "PA System Units":       max(1,  crowd_size // 10000),
    }

    if event_cause in ["protest", "procession", "public_event"]:
        base["Traffic Officers"] = int(base["Traffic Officers"] * 1.3)
        base["Patrol Vehicles"] = int(base["Patrol Vehicles"] * 1.5)
    if event_cause in ["accident", "tree_fall", "Debris", "debris"]:
        base["Ambulances on Standby"] = max(2, base["Ambulances on Standby"])
        base["Fire Brigade Units"] = max(1, base["Fire Brigade Units"])
    if event_cause == "construction":
        base["Barricades"] = int(base["Barricades"] * 2)

    return {k: int(v * multiplier) for k, v in base.items() if v > 0}

def level_from_score(score):
    if score >= 75:
        return "Severe"
    if score >= 45:
        return "Moderate"
    return "Minor"

def adjust_operational_risk(base_score, event_cause, hour, mode, lead_time_min,
                            lanes_blocked, rain_watch):
    score = int(base_score)
    reasons = []

    if mode == "Unplanned Incident":
        score += 12
        reasons.append("Unplanned incident requires faster response")

    if event_cause in ["accident", "tree_fall", "Debris", "debris", "water_logging"]:
        score += 8
        reasons.append("Incident type can block lanes or slow clearance")

    if lanes_blocked >= 2:
        score += 15
        reasons.append("Two or more lanes blocked")
    elif lanes_blocked == 1:
        score += 8
        reasons.append("One lane blocked")

    if rain_watch:
        score += 10
        reasons.append("Rain or waterlogging watch active")

    if mode == "Planned Event":
        if lead_time_min < 60:
            score += 10
            reasons.append("Less than one hour of preparation time")
        elif lead_time_min >= 180:
            score -= 5
            reasons.append("Three or more hours available for preparation")

    if hour in list(range(8, 11)) + list(range(17, 22)):
        score += 8
        reasons.append("Peak traffic window")

    score = max(0, min(100, score))
    return score, level_from_score(score), reasons

def build_response_timeline(mode, risk_level, best_strategy, lead_time_min):
    if mode == "Planned Event":
        prep = max(30, min(lead_time_min, 180))
        return [
            f"T-{prep} min: Confirm event footprint, control room owner, and junction list",
            "T-60 min: Pre-position officers, barricades, patrol vehicles, and advisory messages",
            f"T-30 min: Activate {best_strategy} and verify emergency corridor",
            "T+0 min onward: Monitor CCTV density every 15 minutes and update diversion status",
        ]

    severe_step = "Open emergency corridor and notify ambulance/fire control"
    if risk_level != "Severe":
        severe_step = "Keep emergency corridor ready if queue length increases"
    return [
        "0-5 min: Verify incident location, lanes blocked, and nearest junction impact",
        "5-10 min: Dispatch patrol vehicles and traffic officers to the affected approach",
        f"10-20 min: Activate {best_strategy} and publish public advisory",
        f"20+ min: {severe_step}",
    ]

def get_command_mood(risk_level, mode):
    if risk_level == "Severe":
        return {
            "icon": "https://img.icons8.com/color/48/high-importance.png",
            "title": "High Alert",
            "caption": "Control room should treat this as a priority disruption.",
            "color": "#C62828",
        }
    if risk_level == "Moderate":
        return {
            "icon": "https://img.icons8.com/color/48/warning-shield.png",
            "title": "Watch Closely",
            "caption": "Extra deployment and active monitoring are recommended.",
            "color": "#EF6C00",
        }
    if mode == "Unplanned Incident":
        return {
            "icon": "https://img.icons8.com/color/48/info--v1.png",
            "title": "Verify Fast",
            "caption": "Risk is low, but live incidents still need quick confirmation.",
            "color": "#1565C0",
        }
    return {
        "icon": "https://img.icons8.com/color/48/checked.png",
        "title": "Manageable",
        "caption": "Standard deployment should be enough with routine monitoring.",
        "color": "#2E7D32",
    }

def get_situation_chips(inputs, risk_level):
    chips = []
    if inputs["mode"] == "Planned Event":
        chips.append(("https://img.icons8.com/color/48/calendar.png", "Planned approval"))
        if inputs["lead_time"] >= 180:
            chips.append(("https://img.icons8.com/flat-round/24/checkmark.png", "Strong prep window"))
        elif inputs["lead_time"] < 60:
            chips.append(("https://img.icons8.com/color/48/clock.png", "Short prep window"))
    else:
        chips.append(("https://img.icons8.com/color/48/siren.png", "Live incident"))
        if inputs["lanes_blocked"] > 0:
            chips.append(("https://img.icons8.com/color/48/barricade.png", f"{inputs['lanes_blocked']} lane(s) blocked"))

    if inputs["rain_watch"]:
        chips.append(("https://img.icons8.com/color/48/rain.png", "Rain watch active"))

    if inputs["hour"] in list(range(8, 11)) + list(range(17, 22)):
        chips.append(("https://img.icons8.com/color/48/alarm-clock.png", "Peak hour"))

    status_icon = {
        "Minor": "https://img.icons8.com/color/48/checked.png",
        "Moderate": "https://img.icons8.com/color/48/warning-shield.png",
        "Severe": "https://img.icons8.com/color/48/high-importance.png"
    }[risk_level]
    chips.append((status_icon, f"{risk_level} risk"))
    return chips

def make_command_brief(inputs, risk_score, risk_level, duration_min, busy_until,
                       best_strategy, savings, resources, timeline,
                       operational_reasons):
    resource_lines = "\n".join(
        f"- {name}: {count}" for name, count in resources.items()
    )
    timeline_lines = "\n".join(f"- {item}" for item in timeline)
    reason_lines = "\n".join(f"- {item}" for item in operational_reasons) or "- No extra operational risk factors"

    return f"""# Gridlock Command Brief

## Scenario
- Mode: {inputs['mode']}
- Event/Incident Type: {inputs['event_cause']}
- Zone: {inputs['zone']}
- Crowd / impact estimate: {inputs['crowd_size']:,}
- Start / report hour: {inputs['hour']:02d}:00
- Day: {inputs['day']}
- Lanes blocked: {inputs['lanes_blocked']}
- Rain / waterlogging watch: {'Yes' if inputs['rain_watch'] else 'No'}

## Predicted Impact
- Operational risk score: {risk_score}/100
- Risk level: {risk_level}
- Typical disruption duration: {duration_min} minutes
- Area busy until: ~{busy_until:02d}:00
- Best diversion strategy: {best_strategy}
- Delay saved versus no action: ~{savings} minutes

## Operational Risk Drivers
{reason_lines}

## Resource Plan
{resource_lines}

## Response Timeline
{timeline_lines}
"""

def get_demo_presets():
    return [
        {
            "name": "Evening Rally Surge",
            "mode": "Planned Event",
            "event_cause": "public_event",
            "zone": "Central Zone 1",
            "crowd_size": 20000,
            "hour": 18,
            "day": "Saturday",
            "lead_time": 120,
            "lanes_blocked": 0,
            "rain_watch": False,
        },
        {
            "name": "Rainy Accident Response",
            "mode": "Unplanned Incident",
            "event_cause": "accident",
            "zone": "East Zone 1",
            "crowd_size": 6000,
            "hour": 9,
            "day": "Monday",
            "lead_time": 0,
            "lanes_blocked": 2,
            "rain_watch": True,
        },
        {
            "name": "VIP Movement Prep",
            "mode": "Planned Event",
            "event_cause": "vip_movement",
            "zone": "Central Zone 2",
            "crowd_size": 5000,
            "hour": 17,
            "day": "Friday",
            "lead_time": 240,
            "lanes_blocked": 0,
            "rain_watch": False,
        },
    ]

# Map Builder using real network logic
def build_network_map(zone, risk_level, risk_score, routing, corridor, dispatch):
    lat, lng = tn.NODE_COORDS.get(zone, (12.9716, 77.5946))
    m = folium.Map(location=[lat, lng], zoom_start=13, tiles="CartoDB positron")

    # 1. Draw the underlying road network graph (grey lines)
    for u, v in tn.ROAD_EDGES:
        c1, c2 = tn.NODE_COORDS[u], tn.NODE_COORDS[v]
        folium.PolyLine(
            [c1, c2],
            color="#bdc3c7",
            weight=2,
            opacity=0.6,
            tooltip=f"Road segment: {u} to {v}"
        ).add_to(m)

    # 2. Draw incident zone circle & marker
    radius_meters = {
        "Minor": 400, "Moderate": 750, "Severe": 1200
    }.get(risk_level, 500)
    folium.Circle(
        location=[lat, lng],
        radius=radius_meters,
        color="#e74c3c",
        fill=True,
        fill_opacity=0.15,
        popup=f"Incident Buffer: {risk_level} Risk",
        tooltip="Affected Area Buffer"
    ).add_to(m)

    folium.Marker(
        location=[lat, lng],
        popup=f"Incident Point: {zone}<br>Risk Score: {risk_score}/100",
        icon=folium.Icon(color="red", icon="exclamation-sign")
    ).add_to(m)

    # 3. Draw Commuter routing (if available)
    if routing:
        std_coords = [tn.NODE_COORDS[n] for n in routing["std_path"]]
        div_coords = [tn.NODE_COORDS[n] for n in routing["congested_path"]]
        
        # Standard route (dashed orange/red)
        folium.PolyLine(
            std_coords,
            color="#e67e22",
            weight=3,
            opacity=0.75,
            dash_array="5, 10",
            tooltip="Standard Path (Stuck in traffic)"
        ).add_to(m)

        # Diversion route (solid green)
        folium.PolyLine(
            div_coords,
            color="#2ecc71",
            weight=5,
            opacity=0.9,
            tooltip="Optimized Diversion Path"
        ).add_to(m)
        
        # Draw origin and destination markers
        orig = routing["std_path"][0]
        dest = routing["std_path"][-1]
        folium.Marker(
            location=tn.NODE_COORDS[orig],
            popup=f"Commuter Origin: {orig}",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        folium.Marker(
            location=tn.NODE_COORDS[dest],
            popup=f"Commuter Destination: {dest}",
            icon=folium.Icon(color="purple", icon="stop")
        ).add_to(m)

    # 4. Draw Emergency Corridor (if available)
    if corridor:
        folium.PolyLine(
            corridor["coords_path"],
            color="#3498db",
            weight=6,
            opacity=0.95,
            tooltip=f"Emergency Green Corridor to {corridor['path'][-1]}"
        ).add_to(m)
        
        # Hospital marker
        folium.Marker(
            location=corridor["hospital_coords"],
            popup=f"Destination Medical Center: {corridor['schedule'][-1]['node']}",
            icon=folium.Icon(color="blue", icon="plus")
        ).add_to(m)

    # 5. Draw Police Stations Dispatched
    if dispatch:
        for station in dispatch:
            folium.Marker(
                location=station["coords"],
                popup=f"Station: {station['station']}<br>Dispatched Officers: {station['officers_dispatched']}<br>Dispatched Vehicles: {station['cars_dispatched']}",
                icon=folium.Icon(color="cadetblue", icon="user")
            ).add_to(m)

    # Emoji-Free legend using HTML/CSS
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:10px 15px; border-radius:8px;
                border:1px solid #ccc; font-size:12px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
        <b style="font-size:13px;">Map Legend</b><br>
        <span style="display:inline-block; width:12px; height:12px; background:#bdc3c7; margin-right:5px;"></span>Road Network<br>
        <span style="display:inline-block; width:12px; height:12px; background:#e74c3c; opacity:0.3; margin-right:5px; border-radius:50%;"></span>Affected Radius<br>
        <span style="display:inline-block; width:12px; height:12px; border-top: 2px dashed #e67e22; margin-right:5px; vertical-align: middle;"></span>Standard Route (Congested)<br>
        <span style="display:inline-block; width:12px; height:12px; background:#2ecc71; margin-right:5px;"></span>Diversion Route<br>
        <span style="display:inline-block; width:12px; height:12px; background:#3498db; margin-right:5px;"></span>Emergency Green Corridor<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------

# Custom Header (No emojis, uses Icons8 icon)
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
        <img src="https://img.icons8.com/color/48/traffic-light.png" width="36" height="36" class="custom-icon"/>
        <h1 style="margin:0; font-size:32px;">Gridlock Sentinel Command Center</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Proactive Event Impact Simulator & Mitigation Engine — Team APIcalypse Now")
st.divider()

# Load preset configurations
EVENT_CAUSES = sorted([
    'accident', 'congestion', 'construction', 'Debris',
    'Fog / Low Visibility', 'others', 'pot_holes', 'procession',
    'protest', 'public_event', 'road_conditions', 'tree_fall',
    'vehicle_breakdown', 'vip_movement', 'water_logging'
])
ZONES = sorted(list(tn.NODE_COORDS.keys()))
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
PRESETS = {preset["name"]: preset for preset in get_demo_presets()}

# SIDEBAR INPUT PANEL
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px;">
            <img src="https://img.icons8.com/color/48/graph.png" width="24" height="24"/>
            <h3 style="margin:0;">Control Panel</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Setup the event simulation constraints")
    
    preset_name = st.selectbox(
        "Demo Scenario Preset",
        ["Custom"] + list(PRESETS.keys()),
        help="Use a ready-made situation for fast judging demos"
    )
    preset = PRESETS.get(preset_name, {})

    mode = st.radio(
        "Command Mode",
        ["Planned Event", "Unplanned Incident"],
        index=["Planned Event", "Unplanned Incident"].index(preset.get("mode", "Planned Event")),
        horizontal=True
    )
    event_cause = st.selectbox("Event Type", EVENT_CAUSES,
                                index=EVENT_CAUSES.index(preset.get("event_cause", "public_event")))
    zone = st.selectbox("Location (Zone)", ZONES,
                        index=ZONES.index(preset.get("zone", "Central Zone 1")))
    crowd_size = st.number_input(
        "Expected Crowd / Impact Size",
        min_value=10, max_value=500000,
        value=preset.get("crowd_size", 20000), step=500,
        help="For incidents, use estimated affected road users or queue impact"
    )
    hour = st.slider("Event Start Time (24hr)", 0, 23, preset.get("hour", 18),
                     help="18 = 6:00 PM")
    day = st.selectbox("Day of Week", DAYS, index=DAYS.index(preset.get("day", "Saturday")))
    
    lead_time = preset.get("lead_time", 120)
    lanes_blocked = 0
    if mode == "Planned Event":
        lead_time = st.slider(
            "Preparation Lead Time (minutes)",
            min_value=0, max_value=360, value=lead_time, step=15
        )
    else:
        lanes_blocked = st.slider("Lanes Blocked", 0, 4, preset.get("lanes_blocked", 1))

    rain_watch = st.checkbox("Rain / waterlogging watch", value=preset.get("rain_watch", False))

    st.markdown("---")
    # Dynamic Hospital Selector (Recommends closest but allows override)
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 5px;">
            <img src="https://img.icons8.com/color/48/hospital.png" width="20" height="20"/>
            <b style="font-size:14px;">Green Corridor Settings</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Auto-recommend closest hospital based on zone distance
    zone_coord = tn.NODE_COORDS.get(zone, (12.9716, 77.5946))
    closest_hospital = min(
        tn.HOSPITAL_COORDS.keys(),
        key=lambda h: tn.haversine_distance(zone_coord, tn.HOSPITAL_COORDS[h])
    )
    
    hosp_options = list(tn.HOSPITAL_COORDS.keys())
    target_hospital = st.selectbox(
        "Emergency Hospital Destination",
        options=hosp_options,
        index=hosp_options.index(closest_hospital),
        help="System auto-recommends the closest facility, but you can override it."
    )
    
    st.markdown("---")
    
    # Simulate Button
    if st.button("Simulate Mitigation Strategy", use_container_width=True, type="primary"):
        st.session_state["submitted"] = True
        st.session_state["inputs"] = {
            "mode": mode,
            "event_cause": event_cause,
            "zone": zone,
            "crowd_size": crowd_size,
            "hour": hour,
            "day": day,
            "lead_time": lead_time,
            "lanes_blocked": lanes_blocked,
            "rain_watch": rain_watch,
            "target_hospital": target_hospital
        }

# MAIN CONTENT PANEL
predict_btn = st.session_state.get("submitted", False)

if predict_btn and "inputs" in st.session_state:
    inp = st.session_state["inputs"]
    
    with st.spinner("Running system simulation models..."):
        # 1. Base ML predictions
        risk_result = predict_event_risk(
            inp["event_cause"], inp["crowd_size"],
            inp["zone"], inp["hour"], inp["day"]
        )
        model_risk = risk_result["risk_level"]
        model_risk_score = risk_result["risk_score"]
        
        # 2. Risk Adjustment based on field operations
        risk_score, risk, operational_reasons = adjust_operational_risk(
            model_risk_score,
            inp["event_cause"],
            inp["hour"],
            inp["mode"],
            inp["lead_time"],
            inp["lanes_blocked"],
            inp["rain_watch"],
        )
        crowd_level = risk_result["crowd_level"]

        # 3. Predict Resolution duration
        duration_median, duration_min, duration_max, data_points = predict_duration(inp["event_cause"])
        busy_until = (inp["hour"] + duration_median // 60) % 24

        # 4. Standard recommended resources
        resources_base = recommend_resources(inp["crowd_size"], risk, inp["event_cause"])
        
        # 5. Dynamic Routing & Diversion solver
        # We will route from West Zone 1 (Vijayanagar) to East Zone 2 (Whitefield) by default
        # to show a commuter traversing the city across the incident zone
        commuter_origin = "West Zone 1"
        commuter_destination = "East Zone 2"
        if inp["zone"] == "West Zone 1":
            commuter_origin = "Central Zone 2"
        elif inp["zone"] == "East Zone 2":
            commuter_destination = "East Zone 1"
            
        routing_res = tn.get_routing_scenarios(
            G, 
            source=commuter_origin, 
            target=commuter_destination, 
            incident_node=inp["zone"], 
            risk_score=risk_score
        )
        
        # 6. Emergency Green Corridor calculations
        corridor_res = tn.get_emergency_corridor(
            G, 
            incident_zone=inp["zone"], 
            hospital_name=inp["target_hospital"], 
            risk_score=risk_score
        )
        
        # 7. Police Resource Dispatch optimizer
        officers_needed = resources_base.get("Traffic Officers", 5)
        cars_needed = resources_base.get("Patrol Vehicles", 2)
        dispatch_res, unmet_off, unmet_cars = tn.optimize_police_dispatch(
            required_officers=officers_needed,
            required_cars=cars_needed,
            incident_zone=inp["zone"],
            G=G
        )
        
        # 8. BMTC transit checker
        transit_res = tn.check_bmtc_transit(G, incident_node=inp["zone"], risk_score=risk_score)
        
        # Timeline and brief generation
        best_strat = "Active Diversion Plan" if routing_res and routing_res["savings_mins"] > 0 else "Standard Monitoring"
        timeline = build_response_timeline(inp["mode"], risk, best_strat, inp["lead_time"])
        command_brief = make_command_brief(
            inp, risk_score, risk, duration_median, busy_until, best_strat,
            routing_res["savings_mins"] if routing_res else 0, resources_base, timeline, operational_reasons
        )
        command_mood = get_command_mood(risk, inp["mode"])
        situation_chips = get_situation_chips(inp, risk)

    # RENDER KPI HEADER
    st.markdown(
        f"""
        <div style="border-left: 8px solid {command_mood['color']};
                    background: #ffffff; padding: 18px 20px; border-radius: 8px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 14px;">
            <div style="display:flex; align-items:center; gap:16px;">
                <img src="{command_mood['icon']}" width="48" height="48"/>
                <div>
                    <div style="font-size:22px; font-weight:700; color:{command_mood['color']}; margin:0;">
                        {command_mood['title']}
                    </div>
                    <div style="font-size:14px; color:#6c757d;">{command_mood['caption']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Situation Chips Rendering (No emojis)
    chip_html = ""
    for icon_url, label in situation_chips:
        chip_html += (
            f"<span style='display:inline-flex; align-items:center; background:#f1f3f5; border:1px solid #dee2e6;"
            f"border-radius:999px; padding:6px 12px; margin:0 8px 8px 0; font-size:13px; font-weight:500; color:#495057;'>"
            f"<img src='{icon_url}' width='16' height='16' style='margin-right:6px;'/>{label}</span>"
        )
    st.markdown(chip_html, unsafe_allow_html=True)

    # 3 Metrics Layout
    col_score, col_level, col_time = st.columns(3)
    with col_score:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Operational Risk Score</div>
                <div class="metric-value">{risk_score} <span style="font-size:14px; font-weight:normal; color:#6c757d;">/ 100</span></div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_level:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Adjusted Risk Level</div>
                <div class="metric-value">{risk}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_time:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Area Busy Until</div>
                <div class="metric-value">~{busy_until:02d}:00</div>
            </div>
            """, unsafe_allow_html=True
        )

    # TABBED INTERFACE LAYOUT
    t1, t2, t3, t4, t5 = st.tabs([
        "🎭 Event Digital Twin",
        "🔀 Dynamic Routing",
        "🚑 Emergency Corridor",
        "👮 Police Dispatch Planner",
        "🚌 BMTC Transit Advisor"
    ])

    # TAB 1: DIGITAL TWIN & RISK PROFILE
    with t1:
        st.subheader("Event Risk Assessment Profile")
        
        st.caption(
            f"ML Model Base Prediction: {model_risk} Risk ({model_risk_score}/100 score). "
            "Operational adjustments applied on top based on current field constraints."
        )

        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("**Core Action Recommendations:**")
            actions = {
                "Minor": [
                    "Deploy standard traffic officers at key junctions surrounding the zone.",
                    "Keep patrol vehicles on standard regional patrol rotation.",
                    "Monitor intersection density feeds via local surveillance feeds."
                ],
                "Moderate": [
                    "Pre-position road barriers at secondary road entrances 1 hour before scheduled time.",
                    "Activate alternate route signage at adjacent network junctions.",
                    "Position an ambulance on regional standby.",
                    "Broadcast traffic advisory details through regional warning channels."
                ],
                "Severe": [
                    "Execute complete road closure around target zone 2 hours prior to event.",
                    "Position emergency fire brigade and ambulance services at nearby nodes.",
                    "Initiate detour/diversion routing across the local network.",
                    "Broadcast alert advisory updates to local travel applications.",
                    "Activate Green Corridor protocols for medical response."
                ]
            }
            
            for act in actions[risk]:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:flex-start; gap:8px; margin-bottom:8px;">
                        <img src="https://img.icons8.com/flat-round/24/checkmark.png" width="16" height="16" style="margin-top:3px;"/>
                        <span style="font-size:15px; color:#212529;">{act}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            if operational_reasons:
                st.markdown("<br>**Operational adjustment factors:**", unsafe_allow_html=True)
                for reason in operational_reasons:
                    st.write(f"- {reason}")
                    
        with col_right:
            st.markdown("**Expected Disruption Timeline:**")
            d1, d2 = st.columns(2)
            d1.metric("Typical Duration", f"{duration_median} min")
            d2.metric("Min/Max Horizon", f"{duration_min} - {duration_max} min")
            st.caption(f"Calculated from {data_points} historical events. Defaults applied if sample was sparse.")

        st.markdown("---")
        st.markdown("**SHAP Feature Importance (Decision Drivers)**")
        explain_df = explain_prediction(
            inp["event_cause"], inp["crowd_size"], inp["zone"],
            inp["hour"], inp["day"], crowd_level
        )
        fig_explain = go.Figure(go.Bar(
            x=explain_df["Impact"],
            y=explain_df["Factor"],
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in explain_df["Impact"]],
        ))
        fig_explain.update_layout(
            xaxis_title="Impact on Risk Probability (Right increases risk, Left decreases)",
            height=280, 
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_explain, use_container_width=True)

        st.markdown("---")
        st.markdown("**Past Similar Events in Database:**")
        past_events = get_past_similar_events(inp["event_cause"], inp["zone"])
        if len(past_events) > 0:
            st.dataframe(past_events, use_container_width=True)
        else:
            st.info("No matching historical records found for this subset.")

    # TAB 2: DYNAMIC ROUTING & DIVERSIONS
    with t2:
        st.subheader("Dynamic Diversion Routing Engine")
        st.caption(
            "Models a commuter journey traversing the city through the affected zone. "
            "Computes alternate paths dynamically using current demand weights."
        )
        
        # Let user customize origin and destination for routing
        rc1, rc2 = st.columns(2)
        with rc1:
            route_origin = st.selectbox("Commuter Origin Point", ZONES, index=ZONES.index(commuter_origin))
        with rc2:
            route_destination = st.selectbox("Commuter Destination Point", ZONES, index=ZONES.index(commuter_destination))
            
        # Re-run routing if inputs changed
        if route_origin != commuter_origin or route_destination != commuter_destination:
            routing_res = tn.get_routing_scenarios(
                G, 
                source=route_origin, 
                target=route_destination, 
                incident_node=inp["zone"], 
                risk_score=risk_score
            )
            
        if routing_res:
            col_times, col_paths = st.columns([1, 1])
            with col_times:
                st.markdown("**Travel Statistics Comparison:**")
                st.metric("Standard Path (Stuck Time)", f"{routing_res['stuck_time_mins']} mins", 
                          delta=f"+{routing_res['stuck_time_mins'] - routing_res['std_time_mins']} mins due to jam", delta_color="inverse")
                st.metric("Diversion Path Time", f"{routing_res['congested_time_mins']} mins")
                
                savings = routing_res['savings_mins']
                if savings > 0:
                    st.success(f"Best Strategy: **Route Diversion** — saves approx **{savings} minutes** versus sitting in gridlock.")
                else:
                    st.info("Congestion level allows standard path routing without bypass delays.")
                    
            with col_paths:
                st.markdown("**Routing Details:**")
                st.markdown(f"**Standard Routing Path:**<br>{' &rarr; '.join(routing_res['std_path'])}", unsafe_allow_html=True)
                st.markdown(f"**Congestion Bypass Path:**<br>{' &rarr; '.join(routing_res['congested_path'])}", unsafe_allow_html=True)
        else:
            st.error("Origin and Destination cannot be the same node.")

    # TAB 3: EMERGENCY CORRIDOR GUARDIAN
    with t3:
        st.subheader("Active Emergency Green Corridor Planner")
        st.caption("Secures emergency ambulance transport paths with automated signal scheduling.")

        if corridor_res:
            c_left, c_right = st.columns([1, 1.5])
            with c_left:
                st.markdown(
                    f"""
                    <div style="background-color: #ebf5fb; border-left: 5px solid #3498db; padding: 15px; border-radius: 4px; margin-bottom:15px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
                            <img src="https://img.icons8.com/color/48/ambulance.png" width="24" height="24"/>
                            <b style="color:#2980b9; font-size:16px;">Medical Transit Corridor Active</b>
                        </div>
                        Route: <b>{inp['zone']}</b> to <b>{inp['target_hospital']}</b><br>
                        Total Corridor Distance: <b>{corridor_res['distance_km']:.2f} km</b><br>
                        ETA (Emergency Speed): <b>{corridor_res['eta_mins']} mins</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("**Operational Signal Override Instructions:**")
                st.write("1. Broadcast preemptive signals to all roadside police controllers.")
                st.write("2. Force green state on signal controllers at intersections matching the schedule below.")
                st.write("3. Clear central lane approaches 2 minutes before the ETA window.")
                
            with c_right:
                st.markdown("**Green Corridor Signal Preemption Schedule:**")
                schedule_df = pd.DataFrame(corridor_res["schedule"])
                schedule_df.columns = ["Intersection Node", "Distance (km)", "ETA", "Override Active Window"]
                st.dataframe(schedule_df, use_container_width=True)
        else:
            st.warning("Could not compute emergency corridor routing path.")

    # TAB 4: POLICE DISPATCH PLANNER
    with t4:
        st.subheader("Multi-Station Police Resource Allocation Optimizer")
        st.caption("Allocates officers and patrol units from nearest depots with available capacity.")
        
        # User input override for resource capacities
        ro1, ro2 = st.columns(2)
        with ro1:
            req_off_override = st.number_input("Adjust Required Officers", min_value=1, max_value=100, value=officers_needed)
        with ro2:
            req_cars_override = st.number_input("Adjust Required Patrol Cars", min_value=0, max_value=20, value=cars_needed)
            
        if req_off_override != officers_needed or req_cars_override != cars_needed:
            dispatch_res, unmet_off, unmet_cars = tn.optimize_police_dispatch(
                required_officers=req_off_override,
                required_cars=req_cars_override,
                incident_zone=inp["zone"],
                G=G
            )

        if dispatch_res:
            st.markdown("**Optimized Dispatch Schedule:**")
            disp_data = []
            for d in dispatch_res:
                disp_data.append({
                    "Police Station Depot": d["station"],
                    "Officers Sent": d["officers_dispatched"],
                    "Vehicles Sent": d["cars_dispatched"],
                    "Travel Time ETA": f"{d['travel_time_mins']} mins",
                    "Dispatch Status": d["status"]
                })
            st.dataframe(pd.DataFrame(disp_data), use_container_width=True)
            
            if unmet_off > 0 or unmet_cars > 0:
                st.error(
                    f"Warning: Insufficient resources available. "
                    f"Unmet Deficit: {unmet_off} Officers, {unmet_cars} Patrol Cars. "
                    f"Initiate regional mutual aid callbacks."
                )
            else:
                st.success("Target resource requirements fully covered by adjacent police stations.")
        else:
            st.warning("Police depots are outside network routing limits.")

    # TAB 5: BMTC TRANSIT ADVISORY
    with t5:
        st.subheader("BMTC Public Transit Advisory Portal")
        st.caption("Detects delays and computes detours for public bus commuters intersecting the incident zone.")

        if transit_res:
            for route in transit_res:
                st.markdown(
                    f"""
                    <div style="background-color: #fff9db; border-left: 5px solid #fab005; padding: 15px; border-radius: 4px; margin-bottom: 12px;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:5px;">
                            <img src="https://img.icons8.com/color/48/bus.png" width="22" height="22"/>
                            <b style="color:#f59f00; font-size:15px;">{route['name']}</b>
                            <span style="background-color:#ffe3e3; color:#f03e3e; font-size:11px; font-weight:600; padding:2px 6px; border-radius:4px; margin-left:10px;">
                                {route['status']}
                            </span>
                        </div>
                        Standard Path: <span style="font-size:13px; color:#495057;">{route['standard_stops']}</span><br>
                        Diverted Path: <span style="font-size:13px; color:#495057; font-weight:500;">{route['diverted_stops']}</span><br>
                        Estimated Delay: <b>+{route['estimated_delay_mins']} minutes</b><br>
                        <b style="color:#d9480f;">Commuter advisory:</b> {route['shifted_stop_advise']}<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No active BMTC transit routes are disrupted by this incident.")

    st.markdown("---")
    
    # PDF download brief (Uses clean text download)
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 10px;">
            <img src="https://img.icons8.com/color/48/download.png" width="24" height="24"/>
            <b style="font-size:16px;">Download Command Documentation</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.download_button(
        "Download Brief Report",
        data=command_brief,
        file_name="gridlock_command_brief.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # 🗺️ MAP DISPLAY AT THE BOTTOM
    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 10px;">
            <img src="https://img.icons8.com/color/48/map.png" width="24" height="24"/>
            <b style="font-size:16px;">Affected Network & Active Mitigation Routing</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    event_map = build_network_map(
        inp["zone"], 
        risk, 
        risk_score, 
        routing_res, 
        corridor_res, 
        dispatch_res
    )
    st_folium(event_map, use_container_width=True, height=500)

else:
    # Home default instructions
    st.info("Fill in the event details in the left-hand sidebar control panel and click 'Simulate Mitigation Strategy' to run calculations.")
    st.markdown(
        """
        ### System Capabilities
        This system runs real-time spatial calculations across Bengaluru's road graph.
        - **Digital Twin Risk Profile**: Categorizes incident impacts based on temporal and categorical ML models.
        - **Dynamic Diversion Simulator**: Calculates shortest paths using NetworkX Dijkstra routing based on congestion weights.
        - **Green Corridor Planner**: Calculates preemptive paths to major hospitals and computes intersection ETAs.
        - **Police Dispatch Optimizer**: Dispatches emergency personnel from stations by optimizing travel distances and station capacities.
        - **BMTC Transit Advisor**: Reroutes public transport lines and alerts commuters of bus stops to bypass.
        """
    )
