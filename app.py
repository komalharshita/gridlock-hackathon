
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from pathlib import Path

st.set_page_config(
    page_title="Gridlock Digital Twin",
    page_icon="🚦",
    layout="wide"
)

# -------------------------------------------------------
# LOAD MODEL + DATA
# Why: load once and cache so app stays fast
# -------------------------------------------------------
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

# -------------------------------------------------------
# ZONE CENTER COORDINATES
# Why: when officer picks a zone, we center the map there
# -------------------------------------------------------
ZONE_COORDS = {
    "Central Zone 1": (12.9716, 77.5946),
    "Central Zone 2": (12.9650, 77.5850),
    "North Zone 1":   (13.0500, 77.5900),
    "North Zone 2":   (13.0800, 77.5700),
    "South Zone 1":   (12.9100, 77.5800),
    "South Zone 2":   (12.8800, 77.6000),
    "East Zone 1":    (12.9900, 77.6500),
    "East Zone 2":    (12.9700, 77.7000),
    "West Zone 1":    (12.9600, 77.5200),
    "West Zone 2":    (12.9400, 77.4900),
    "Unknown":        (12.9716, 77.5946),
}

# -------------------------------------------------------
# CORE FUNCTIONS
# -------------------------------------------------------
def predict_event_risk(event_cause, crowd_size, zone, hour, day_of_week):
    """
    Why crowd_size as number: we convert it to Low/Medium/High
    internally but accept exact number from officer for realism
    """
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

    # Convert to 0-100 risk score
    # Why: a number feels more technical and precise than just a label
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

def get_past_similar_events(event_cause, zone, limit=5):
    """
    Why: showing real past events makes the system feel grounded
    in actual Bengaluru data, not made-up predictions
    """
    similar = df[df["event_cause"] == event_cause].copy()
    if zone != "Unknown":
        zone_match = similar[similar["zone"] == zone]
        if len(zone_match) >= 2:
            similar = zone_match

    similar = similar[["event_cause", "address", "zone", "risk_level",
                        "hour", "day_of_week", "crowd_proxy"]].dropna(subset=["address"])
    similar = similar[~similar["address"].str.contains("ಲಿಂಕ್|ರಸ್ತೆ", na=False)]  # skip Kannada addresses for readability
    return similar.head(limit).reset_index(drop=True)

def predict_duration(event_cause):
    """
    Why: police need to know HOW LONG the disruption lasts,
    not just that it will happen. We use median from past data
    where resolution time is known (69 rows).
    For event types with no data, we use domain-knowledge defaults.
    """
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
    """
    Why separate resources per event type: a protest needs more
    police, a construction site needs more barricades,
    a public event might need ambulances on standby.
    """
    multiplier = {"Minor": 1.0, "Moderate": 1.3, "Severe": 1.7}.get(risk_level, 1.0)

    base = {
        "👮 Traffic Officers":      max(2,  crowd_size // 500),
        "🚧 Barricades":            max(2,  crowd_size // 1200),
        "🚔 Patrol Vehicles":       max(1,  crowd_size // 3000),
        "🚑 Ambulances on Standby": max(1,  crowd_size // 5000),
        "🚒 Fire Brigade Units":    1 if crowd_size > 5000 or risk_level == "Severe" else 0,
        "📹 CCTV/Surveillance":     max(2,  crowd_size // 4000),
        "🏥 First Aid Posts":       max(1,  crowd_size // 8000),
        "🚌 Bus Route Diversions":  max(1,  crowd_size // 6000),
        "🔊 PA System Units":       max(1,  crowd_size // 10000),
    }

    # Event-specific additions
    if event_cause in ["protest", "procession", "public_event"]:
        base["👮 Traffic Officers"] = int(base["👮 Traffic Officers"] * 1.3)
        base["🚔 Patrol Vehicles"] = int(base["🚔 Patrol Vehicles"] * 1.5)
    if event_cause in ["accident", "tree_fall", "Debris", "debris"]:
        base["🚑 Ambulances on Standby"] = max(2, base["🚑 Ambulances on Standby"])
        base["🚒 Fire Brigade Units"] = max(1, base["🚒 Fire Brigade Units"])
    if event_cause == "construction":
        base["🚧 Barricades"] = int(base["🚧 Barricades"] * 2)

    return {k: int(v * multiplier) for k, v in base.items() if v > 0}

def get_diversion_scenarios(risk_level, crowd_size, duration_min):
    """
    Why use duration: longer events need more aggressive diversion.
    A 30-min accident vs a 4-hour procession need different strategies.
    """
    base = {"Minor": 15, "Moderate": 35, "Severe": 55}.get(risk_level, 20)
    crowd_factor = min(crowd_size / 10000 * 10, 25)
    duration_factor = min(duration_min / 60 * 5, 20)
    total = base + crowd_factor + duration_factor

    return {
        "No Diversion":        round(total),
        "Route B Diversion":   round(total * 0.55),
        "Early Road Closure":  round(total * 0.35),
    }

def build_map(zone, risk_level, duration_min, hour):
    """
    Why Folium: free, no API key, works offline, renders in Streamlit.
    Shows: event pin, affected radius, two diversion route lines.
    """
    lat, lng = ZONE_COORDS.get(zone, (12.9716, 77.5946))

    m = folium.Map(location=[lat, lng], zoom_start=14, tiles="CartoDB positron")

    # Affected radius — bigger for severe/longer events
    radius_meters = {
        "Minor": 400, "Moderate": 700, "Severe": 1100
    }.get(risk_level, 500) + (duration_min // 30) * 50

    folium.Circle(
        location=[lat, lng],
        radius=radius_meters,
        color="#E24B4A",
        fill=True,
        fill_opacity=0.25,
        popup=f"⚠️ Affected Area — {risk_level} risk, ~{radius_meters}m radius",
        tooltip="Affected zone"
    ).add_to(m)

    folium.Marker(
        location=[lat, lng],
        popup=f"📍 Event Location\nRisk: {risk_level}\nExpected duration: {duration_min} min",
        tooltip="Event location",
        icon=folium.Icon(color="red", icon="exclamation-sign")
    ).add_to(m)

    # Diversion Route A — go slightly north then east
    route_a = [
        [lat + 0.005, lng],
        [lat + 0.005, lng + 0.010],
        [lat,         lng + 0.010],
    ]
    folium.PolyLine(
        route_a, color="#1D9E75", weight=4,
        tooltip="🟢 Route B Diversion"
    ).add_to(m)

    # Diversion Route B — go slightly south then west
    route_b = [
        [lat - 0.005, lng],
        [lat - 0.005, lng - 0.008],
        [lat,         lng - 0.008],
    ]
    folium.PolyLine(
        route_b, color="#7F77DD", weight=4,
        tooltip="🟣 Early Road Closure Route"
    ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;">
        <b>Map Legend</b><br>
        🔴 Affected Area<br>
        🟢 Route B Diversion<br>
        🟣 Early Road Closure Route<br>
        📍 Event Location
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🚦 Gridlock Digital Twin")
st.caption("Pre-Event Impact Simulator for Traffic Management — Team APIcalypse Now")
st.divider()

EVENT_CAUSES = sorted([
    'accident', 'congestion', 'construction', 'Debris',
    'Fog / Low Visibility', 'others', 'pot_holes', 'procession',
    'protest', 'public_event', 'road_conditions', 'tree_fall',
    'vehicle_breakdown', 'vip_movement', 'water_logging'
])
ZONES = [
    'Central Zone 1', 'Central Zone 2',
    'East Zone 1', 'East Zone 2',
    'North Zone 1', 'North Zone 2',
    'South Zone 1', 'South Zone 2',
    'West Zone 1', 'West Zone 2',
    'Unknown'
]
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

# SIDEBAR — input form
with st.sidebar:
    st.header("📋 Event Details")
    st.caption("Fill in details of the event to be approved")

    event_cause = st.selectbox("Event Type", EVENT_CAUSES,
                                index=EVENT_CAUSES.index("public_event"))
    zone = st.selectbox("Location (Zone)", ZONES)
    crowd_size = st.number_input(
        "Expected Crowd Size (exact number)",
        min_value=10, max_value=500000,
        value=20000, step=500,
        help="Enter exact expected headcount"
    )
    hour = st.slider("Event Start Time (24hr)", 0, 23, 18,
                     help="18 = 6:00 PM")
    day = st.selectbox("Day of Week", DAYS, index=5)

    st.divider()
    if st.button("🔍 Simulate Event Impact",
                 use_container_width=True, type="primary"):
        st.session_state["submitted"] = True
        st.session_state["inputs"] = {
            "event_cause": event_cause,
            "zone": zone,
            "crowd_size": crowd_size,
            "hour": hour,
            "day": day,
        }

predict_btn = st.session_state.get("submitted", False)

# MAIN AREA
if predict_btn and "inputs" in st.session_state:
    inp = st.session_state["inputs"]
    with st.spinner("Running simulation..."):
        risk_result = predict_event_risk(
            inp["event_cause"], inp["crowd_size"],
            inp["zone"], inp["hour"], inp["day"]
        )
        risk = risk_result["risk_level"]
        risk_score = risk_result["risk_score"]
        crowd_level = risk_result["crowd_level"]

        duration_median, duration_min, duration_max, data_points = predict_duration(event_cause)
        busy_until = (hour + duration_median // 60) % 24

        resources = recommend_resources(crowd_size, risk, event_cause)
        scenarios = get_diversion_scenarios(risk, crowd_size, duration_median)
        past_events = get_past_similar_events(event_cause, zone)

    # ---- SECTION 1: Digital Twin Impact ----
    st.subheader("🎭 Digital Twin — Predicted Impact")

    col_score, col_level, col_time = st.columns(3)
    col_score.metric("⚡ Risk Score", f"{risk_score}/100")
    col_level.metric("🎯 Risk Level", risk)
    col_time.metric("⏱️ Area Busy Until", f"~{busy_until:02d}:00")

    if risk == "Minor":
        st.success(f"✅ **{risk} Risk ({risk_score}/100)** — Manageable with standard deployment")
    elif risk == "Moderate":
        st.warning(f"⚠️ **{risk} Risk ({risk_score}/100)** — Increased disruption expected, enhanced deployment needed")
    else:
        st.error(f"🚨 **{risk} Risk ({risk_score}/100)** — High disruption, road closure highly probable")

    # Confidence breakdown
    conf = risk_result["confidence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Minor Probability",    f"{conf.get('Minor',0)*100:.1f}%")
    c2.metric("Moderate Probability", f"{conf.get('Moderate',0)*100:.1f}%")
    c3.metric("Severe Probability",   f"{conf.get('Severe',0)*100:.1f}%")

    # Recommended actions
    st.markdown("**📋 Recommended Actions:**")
    actions = {
        "Minor": [
            "✅ Deploy standard traffic officers at key junctions",
            "✅ Keep patrol vehicle on standby",
            "✅ Monitor via CCTV",
        ],
        "Moderate": [
            "⚠️ Pre-position barricades 1 hour before event",
            "⚠️ Activate alternate route signage",
            "⚠️ Deploy ambulance on standby",
            "⚠️ Coordinate with nearest police station",
            "⚠️ Send public advisory via traffic apps",
        ],
        "Severe": [
            "🚨 Initiate road closure 2 hours before event",
            "🚨 Deploy fire brigade and ambulance units",
            "🚨 Activate all diversion routes",
            "🚨 Coordinate with BBMP and BMTC for bus rerouting",
            "🚨 Issue public advisory via media and Bengaluru traffic app",
            "🚨 Station senior officer at event location",
            "🚨 Enable emergency vehicle corridor",
        ],
    }
    for action in actions[risk]:
        st.write(action)

    # Duration prediction
    st.divider()
    st.markdown("**⏱️ Expected Event Duration (based on past similar events):**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Typical Duration", f"{duration_median} min")
    d2.metric("Minimum", f"{duration_min} min")
    d3.metric("Maximum", f"{duration_max} min")
    d4.metric("Based on", f"{data_points} records" if data_points > 0 else "Domain estimate")

    # Past similar events
    st.divider()
    st.markdown("**📂 Past Similar Events in Bengaluru:**")
    if len(past_events) > 0:
        st.dataframe(past_events, use_container_width=True)
    else:
        st.info("No past events found for this combination")

    st.divider()

    # ---- SECTION 2: Resource Allocation ----
    st.subheader("🧠 Resource Allocation Intelligence")
    st.caption(f"For {crowd_size:,} people — {risk} risk — {event_cause.replace('_',' ').title()}")

    res_cols = st.columns(3)
    for i, (resource, count) in enumerate(resources.items()):
        res_cols[i % 3].metric(resource, count)

    st.divider()

    # ---- SECTION 3: Diversion Simulator ----
    st.subheader("🔀 Dynamic Diversion Simulator")

    best = min(scenarios, key=scenarios.get)
    colors = ["#1D9E75" if k == best else "#7F77DD" for k in scenarios]

    fig = go.Figure(go.Bar(
        x=list(scenarios.keys()),
        y=list(scenarios.values()),
        marker_color=colors,
        text=[f"{v} min" for v in scenarios.values()],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Estimated Delay (minutes)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    savings = scenarios["No Diversion"] - scenarios[best]
    st.success(f"✅ Best Strategy: **{best}** — saves ~{savings} minutes vs no action")

    st.divider()

    # ---- SECTION 4: Map ----
    st.subheader("🗺️ Affected Area & Diversion Routes")
    st.caption("Red = affected zone | Green = Route B diversion | Purple = Early closure route")

    event_map = build_map(zone, risk, duration_median, hour)
    st_folium(event_map, use_container_width=True, height=500)

else:
    st.info("👈 Fill in event details on the left and click **Simulate Event Impact**")
    st.markdown("""
    ### What this system does
    Before approving any event, police enter 5 details.
    The system instantly provides:
    - 🎯 **Risk Score** (0–100) with confidence breakdown
    - 📋 **Recommended actions** based on risk level
    - ⏱️ **Predicted event duration** from past similar events
    - 📂 **Past similar events** in Bengaluru
    - 🧠 **Full resource allocation** (officers, ambulance, fire brigade...)
    - 🗺️ **Live map** with affected area and diversion routes
    """)
