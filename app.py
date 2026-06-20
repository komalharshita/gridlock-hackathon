import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import streamlit.components.v1 as components
import requests
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
MAPPLS_KEY = os.getenv("MAPPLS_API_KEY")

st.set_page_config(
    page_title="Gridlock Digital Twin",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.main .block-container { padding-top: 1rem; background-color: #f0f4f8; }
.hero-header {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a6b 50%, #0d47a1 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 4px 20px rgba(10, 22, 40, 0.4);
}
.hero-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; color: white; }
.hero-header p { margin: 0.3rem 0 0; font-size: 0.95rem; color: #90caf9; font-weight: 400; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    color: white;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.risk-badge-severe {
    background: linear-gradient(135deg, #b71c1c, #e53935);
    color: white; padding: 1rem 2rem; border-radius: 10px;
    font-size: 1.5rem; font-weight: 700; text-align: center;
    box-shadow: 0 4px 15px rgba(183,28,28,0.4); margin: 1rem 0;
}
.risk-badge-moderate {
    background: linear-gradient(135deg, #e65100, #f57c00);
    color: white; padding: 1rem 2rem; border-radius: 10px;
    font-size: 1.5rem; font-weight: 700; text-align: center;
    box-shadow: 0 4px 15px rgba(230,81,0,0.4); margin: 1rem 0;
}
.risk-badge-minor {
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
    color: white; padding: 1rem 2rem; border-radius: 10px;
    font-size: 1.5rem; font-weight: 700; text-align: center;
    box-shadow: 0 4px 15px rgba(27,94,32,0.4); margin: 1rem 0;
}
.section-header {
    background: linear-gradient(90deg, #0d47a1, #1565c0);
    color: white; padding: 0.6rem 1.2rem; border-radius: 8px;
    font-size: 1rem; font-weight: 600; margin: 1.5rem 0 1rem; letter-spacing: 0.3px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #1a3a6b 100%);
}
[data-testid="stSidebar"] > div { color: white !important; }
[data-testid="stSidebar"] label { color: #90caf9 !important; font-size: 0.85rem; font-weight: 500; }
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #0a1628 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] { background: rgba(255,255,255,0.1) !important; border-color: rgba(255,255,255,0.2) !important; }
[data-testid="stSidebar"] input { 
    color: #0a1628 !important; 
    background: rgba(255,255,255,0.92) !important; 
    border-color: rgba(255,255,255,0.3) !important;
    caret-color: #0a1628 !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: #888888 !important;
    -webkit-text-fill-color: #888888 !important;
}
[data-testid="stSidebar"] .stTextInput input { 
    color: #0a1628 !important;
    -webkit-text-fill-color: #0a1628 !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    color: #0a1628 !important;
    -webkit-text-fill-color: #0a1628 !important;
}
[data-testid="stMetric"] { background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
[data-testid="stMetricLabel"] { color: #546e7a !important; font-size: 0.8rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { color: #0d47a1 !important; font-weight: 700 !important; }
hr { border-color: #e3eaf5; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL + DATA
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
# GEOCODING — converts address text to lat/lng
# Why: officer types a real landmark, we get exact coordinates
# -------------------------------------------------------
def get_mappls_token():
    """Get OAuth access token using Client ID and Secret."""
    try:
        client_id = os.getenv("MAPPLS_CLIENT_ID")
        client_secret = os.getenv("MAPPLS_CLIENT_SECRET")
        if not client_id or not client_secret:
            print("[mappls] Missing MAPPLS_CLIENT_ID / MAPPLS_CLIENT_SECRET in env")
            return None

        resp = requests.post(
            "https://outpost.mappls.com/api/security/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        print(f"[mappls] Token request failed: {resp.status_code} {resp.text}")
        return None
    except Exception as e:
        print(f"[mappls] Token request exception: {e}")
        return None

def geocode_location(address_text):
    """Converts address text to lat/lng using OpenStreetMap Nominatim (free, no key needed)."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{address_text}, Bengaluru, India",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": "GridlockDigitalTwin/1.0"},  # required by Nominatim's usage policy
            timeout=5
        )
        if resp.status_code != 200:
            print(f"[nominatim] Request failed: {resp.status_code} {resp.text}")
            return None

        results = resp.json()
        if not results:
            print(f"[nominatim] No results for '{address_text}'")
            return None

        result = results[0]
        lat = float(result.get("lat", 0))
        lng = float(result.get("lon", 0))
        formatted = result.get("display_name", address_text)

        if lat and lng:
            return lat, lng, formatted

        return None
    except Exception as e:
        print(f"[nominatim] Exception: {e}")
        return None


def lat_lng_to_zone(lat, lng):
    """
    Converts lat/lng to approximate zone for ML model.
    Why: model was trained on zone names, not raw coordinates.
    We derive zone from coordinates using simple boundary rules.
    """
    if lat > 13.0:
        return "North Zone 2" if lng < 77.58 else "North Zone 1"
    elif lat > 12.98:
        return "North Zone 1"
    elif lat > 12.96:
        return "Central Zone 1" if lng < 77.59 else "East Zone 1"
    elif lat > 12.94:
        return "Central Zone 2" if lng < 77.59 else "East Zone 1"
    elif lat > 12.92:
        return "South Zone 1" if lng < 77.59 else "East Zone 2"
    else:
        return "South Zone 2"

# -------------------------------------------------------
# NEARBY FACILITIES — police stations & hospitals
# Why: Mappls Nearby API doesn't return lat/lng on this account's
# tier (confirmed via testing), so we use the free OSM Overpass API
# instead, which returns real coordinates for these categories.
# -------------------------------------------------------
OVERPASS_TAGS = {
    "police station": '"amenity"="police"',
    "hospital": '"amenity"="hospital"',
}

def get_nearby_facilities(lat, lng, keyword, radius=3000, limit=3):
    """
    Fetches nearby places (police stations, hospitals, etc.) using the
    free OpenStreetMap Overpass API (no key needed). Returns a list of
    dicts: name, lat, lng, address (address is best-effort from OSM tags).
    """
    try:
        tag_filter = OVERPASS_TAGS.get(keyword, '"amenity"="police"')
        query = f"""
        [out:json][timeout:10];
        (
          node[{tag_filter}](around:{radius},{lat},{lng});
          way[{tag_filter}](around:{radius},{lat},{lng});
        );
        out center {limit};
        """
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=10
        )

        if resp.status_code != 200:
            print(f"[overpass] {keyword} request failed: {resp.status_code} {resp.text[:200]}")
            return []

        data = resp.json()
        elements = data.get("elements", [])

        facilities = []
        for el in elements[:limit]:
            tags = el.get("tags", {})
            # 'node' elements have lat/lon directly; 'way' elements have a 'center' object instead
            f_lat = el.get("lat") or el.get("center", {}).get("lat")
            f_lng = el.get("lon") or el.get("center", {}).get("lon")
            if not (f_lat and f_lng):
                continue
            facilities.append({
                "name": tags.get("name", keyword.title()),
                "address": tags.get("addr:full") or tags.get("addr:street", ""),
                "lat": float(f_lat),
                "lng": float(f_lng),
            })

        if not facilities:
            print(f"[overpass] {keyword}: no results with coordinates near ({lat},{lng})")
        return facilities

    except Exception as e:
        print(f"[overpass] {keyword} exception: {e}")
        return []

# -------------------------------------------------------
# CORE FUNCTIONS
# -------------------------------------------------------
def predict_event_risk(event_cause, crowd_size, zone, hour):
    if crowd_size < 2000:
        crowd_level = "Low"
    elif crowd_size < 10000:
        crowd_level = "Medium"
    else:
        crowd_level = "High"

    day_of_week = st.session_state["inputs"].get("day", "Monday")
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
    return {"risk_level": pred, "risk_score": risk_score, "confidence": proba_dict, "crowd_level": crowd_level}

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
        "vip_movement": 60, "accident": 45, "vehicle_breakdown": 30,
        "congestion": 90, "construction": 240, "tree_fall": 60,
        "water_logging": 120, "pot_holes": 30, "road_conditions": 60,
        "Debris": 45, "debris": 45, "others": 60,
        "Fog / Low Visibility": 90, "test_demo": 30,
    }
    known = df[(df["event_cause"] == event_cause) & df["resolution_minutes"].notna()]
    if len(known) >= 3:
        return int(known["resolution_minutes"].median()), int(known["resolution_minutes"].min()), int(known["resolution_minutes"].max()), len(known)
    else:
        d = defaults.get(event_cause, 60)
        return d, int(d * 0.5), int(d * 2), 0

def recommend_resources(crowd_size, risk_level, event_cause):
    multiplier = {"Minor": 1.0, "Moderate": 1.3, "Severe": 1.7}.get(risk_level, 1.0)
    base = {
        "👮 Traffic Officers":      max(2, crowd_size // 500),
        "🚧 Barricades":            max(2, crowd_size // 1200),
        "🚔 Patrol Vehicles":       max(1, crowd_size // 3000),
        "🚑 Ambulances on Standby": max(1, crowd_size // 5000),
        "🚒 Fire Brigade Units":    1 if crowd_size > 5000 or risk_level == "Severe" else 0,
        "📹 CCTV/Surveillance":     max(2, crowd_size // 4000),
        "🏥 First Aid Posts":       max(1, crowd_size // 8000),
        "🚌 Bus Route Diversions":  max(1, crowd_size // 6000),
        "🔊 PA System Units":       max(1, crowd_size // 10000),
    }
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
    base = {"Minor": 15, "Moderate": 35, "Severe": 55}.get(risk_level, 20)
    total = base + min(crowd_size / 10000 * 10, 25) + min(duration_min / 60 * 5, 20)
    return {
        "No Diversion":       round(total),
        "Route B Diversion":  round(total * 0.55),
        "Early Road Closure": round(total * 0.35),
    }

def build_mappls_map(lat, lng, risk_level, duration_min, location_name):
    radius_meters = {
        "Minor": 400, "Moderate": 700, "Severe": 1100
    }.get(risk_level, 500) + (duration_min // 30) * 50

    risk_color = {"Minor": "#1b5e20", "Moderate": "#e65100", "Severe": "#c62828"}.get(risk_level, "#c62828")
    route_b_color = "#00b386"      # bright teal-green, high contrast
    early_closure_color = "#7c3aed"  # bold purple

    dest1_lat = round(lat + 0.012, 6)
    dest1_lng = round(lng + 0.015, 6)
    dest2_lat = round(lat - 0.010, 6)
    dest2_lng = round(lng - 0.012, 6)

    # Fetch nearby police + hospitals (via Overpass, see function above)
    police = get_nearby_facilities(lat, lng, "police station", limit=2)
    hospitals = get_nearby_facilities(lat, lng, "hospital", limit=2)

    # Build marker JS for facilities that have coordinates
    facility_markers_js = ""
    for p in police:
        if p["lat"] and p["lng"]:
            facility_markers_js += f"""
        new mappls.Marker({{
            map: map,
            position: {{ lat: {p['lat']}, lng: {p['lng']} }},
            icon_url: 'https://apis.mapmyindia.com/map_v3/1.png',
            popupHtml: '🚓 <b>{p["name"]}</b><br>{p.get("address","")}',
            fitbounds: false
        }});"""
    for h in hospitals:
        if h["lat"] and h["lng"]:
            facility_markers_js += f"""
        new mappls.Marker({{
            map: map,
            position: {{ lat: {h['lat']}, lng: {h['lng']} }},
            icon_url: 'https://apis.mapmyindia.com/map_v3/2.png',
            popupHtml: '🏥 <b>{h["name"]}</b><br>{h.get("address","")}',
            fitbounds: false
        }});"""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ width: 100%; height: 520px; }}
        .legend {{
            position: absolute; bottom: 30px; left: 10px;
            background: white; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #ccc; font-size: 13px; font-family: Arial;
            z-index: 999; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
            font-weight: 600;
        }}
        .legend-item {{ display:flex; align-items:center; margin: 6px 0; }}
        .dot {{ width:16px; height:16px; border-radius:50%; margin-right:10px; flex-shrink:0; border: 2px solid rgba(0,0,0,0.15); }}
    </style>
</head>
<body>
<div id="map"></div>
<div class="legend">
    <b>🗺️ Map Legend</b><br>
    <div class="legend-item"><div class="dot" style="background:{risk_color}"></div> Affected Zone</div>
    <div class="legend-item"><div class="dot" style="background:{route_b_color}"></div> Route B Diversion</div>
    <div class="legend-item"><div class="dot" style="background:{early_closure_color}"></div> Early Closure Route</div>
    <div class="legend-item"><div class="dot" style="background:#0d47a1"></div> Event Location</div>
    <div class="legend-item"><div class="dot" style="background:#000"></div> 🚓 Police / 🏥 Hospital</div>
</div>
<script>
window.addEventListener('load', function() {{
    var map = new mappls.Map('map', {{
        center: [{lat}, {lng}],
        zoom: 14,
        search: false
    }});
    map.on('load', function() {{
        new mappls.Marker({{
            map: map,
            position: {{ lat: {lat}, lng: {lng} }},
            popupHtml: '<b>📍 {location_name}</b><br>Risk: {risk_level}<br>Duration: ~{duration_min} min',
            fitbounds: false
        }});

        var circle = mappls.Circle({{
            map: map,
            center: {{ lat: {lat}, lng: {lng} }},
            radius: {radius_meters},
            strokeColor: '{risk_color}',
            strokeOpacity: 1,
            strokeWeight: 4,
            fillColor: '{risk_color}',
            fillOpacity: 0.35,
            popupHtml: '⚠️ Affected Zone — {risk_level} risk, ~{radius_meters}m radius'
        }});
        console.log('Circle created:', circle);

        mappls.direction({{
            map: map,
            origin: '{lat},{lng}',
            destination: '{dest1_lat},{dest1_lng}',
            routeColor: '{route_b_color}',
            routeWeight: 6,
            routeOpacity: 1,
            alternatives: false,
            fitbounds: false
        }});

        mappls.direction({{
            map: map,
            origin: '{lat},{lng}',
            destination: '{dest2_lat},{dest2_lng}',
            routeColor: '{early_closure_color}',
            routeWeight: 6,
            routeOpacity: 1,
            alternatives: false,
            fitbounds: false
        }});
        {facility_markers_js}
    }});
}});
</script>
<script src="https://apis.mappls.com/advancedmaps/v1/{MAPPLS_KEY}/map_load?v=1.5&plugins=direction,Circle"></script>
</body>
</html>
"""
    return html


# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🚦 Gridlock Hackathon 2.0</div>
    <h1>Digital Twin — Event Impact Simulator</h1>
    <p>Pre-event traffic intelligence for Bengaluru Police | Team APIcalypse Now</p>
</div>
""", unsafe_allow_html=True)

EVENT_CAUSES = sorted([
    'accident', 'congestion', 'construction', 'Debris',
    'Fog / Low Visibility', 'others', 'pot_holes', 'procession',
    'protest', 'public_event', 'road_conditions', 'tree_fall',
    'vehicle_breakdown', 'vip_movement', 'water_logging'
])
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
HOURS = list(range(0, 24))
MINUTES = ["00", "15", "30", "45"]

with st.sidebar:
    st.header("📋 Event Details")
    st.caption("Fill in details of the event to be approved")

    event_cause = st.selectbox("Event Type", EVENT_CAUSES,
                               index=EVENT_CAUSES.index("public_event"))

    # Location as free text input
    location_input = st.text_input(
        "📍 Event Location",
        placeholder="e.g. Lalbagh Botanical Garden, MG Road, Cubbon Park...",
        help="Type any Bengaluru landmark or address"
    )

    crowd_size = st.number_input(
        "Expected Crowd Size",
        min_value=10, max_value=500000,
        value=20000, step=500,
        help="Enter exact expected headcount"
    )

    # Time as two dropdowns
    st.markdown("**⏰ Event Start Time**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        hour = st.selectbox("Hour", HOURS, index=18,
                            format_func=lambda x: f"{x:02d}")
    with t_col2:
        minute_str = st.selectbox("Min", MINUTES)

    day = st.selectbox("Day of Week", DAYS, index=5)

    st.divider()
    if st.button("🔍 Simulate Event Impact",
                 use_container_width=True, type="primary"):

        if not location_input.strip():
            st.error("Please enter an event location.")
        else:
            with st.spinner("📍 Locating address..."):
                geo_result = geocode_location(location_input.strip())

            if geo_result is None:
                st.error(f"❌ Could not find '{location_input}' on the map. Please check the spelling or try a nearby landmark.")
            else:
                geo_lat, geo_lng, formatted_address = geo_result
                zone = lat_lng_to_zone(geo_lat, geo_lng)
                minute_val = int(minute_str)
                hour_decimal = hour + minute_val / 60

                st.session_state["submitted"] = True
                st.session_state["inputs"] = {
                    "event_cause": event_cause,
                    "location_input": location_input,
                    "formatted_address": formatted_address,
                    "lat": geo_lat,
                    "lng": geo_lng,
                    "zone": zone,
                    "crowd_size": crowd_size,
                    "hour": hour_decimal,
                    "hour_display": f"{hour:02d}:{minute_str}",
                    "day": day,
                }

predict_btn = st.session_state.get("submitted", False)

if predict_btn and "inputs" in st.session_state:
    inp = st.session_state["inputs"]

    with st.spinner("Running simulation..."):
        risk_result = predict_event_risk(
            inp["event_cause"], inp["crowd_size"],
            inp["zone"], inp["hour"]
        )
        risk = risk_result["risk_level"]
        risk_score = risk_result["risk_score"]

        duration_median, duration_min, duration_max, data_points = predict_duration(inp["event_cause"])
        busy_hour = int(inp["hour"] + duration_median // 60) % 24

        resources = recommend_resources(inp["crowd_size"], risk, inp["event_cause"])
        scenarios = get_diversion_scenarios(risk, inp["crowd_size"], duration_median)
        past_events = get_past_similar_events(inp["event_cause"], inp["zone"])

    # ---- SECTION 1: Digital Twin ----
    st.markdown('<div class="section-header">🎭 Digital Twin — Predicted Impact</div>', unsafe_allow_html=True)

    # Show resolved location
    st.info(f"📍 **Location resolved:** {inp['formatted_address']} | 🕐 **Start time:** {inp['hour_display']} {inp['day']} | 🗺️ **Zone:** {inp['zone']}")

    col_score, col_level, col_time = st.columns(3)
    col_score.metric("⚡ Risk Score", f"{risk_score}/100")
    col_level.metric("🎯 Risk Level", risk)
    col_time.metric("⏱️ Area Busy Until", f"~{busy_hour:02d}:00")

    risk_class = {"Minor": "minor", "Moderate": "moderate", "Severe": "severe"}[risk]
    risk_emoji = {"Minor": "🟢", "Moderate": "🟡", "Severe": "🔴"}[risk]
    st.markdown(f'<div class="risk-badge-{risk_class}">{risk_emoji} {risk} RISK — Score: {risk_score}/100</div>',
                unsafe_allow_html=True)

    conf = risk_result["confidence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Minor Probability", f"{conf.get('Minor',0)*100:.1f}%")
    c2.metric("Moderate Probability", f"{conf.get('Moderate',0)*100:.1f}%")
    c3.metric("Severe Probability", f"{conf.get('Severe',0)*100:.1f}%")

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

    st.divider()
    st.markdown("**⏱️ Expected Event Duration (based on past similar events):**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Typical Duration", f"{duration_median} min")
    d2.metric("Minimum", f"{duration_min} min")
    d3.metric("Maximum", f"{duration_max} min")
    d4.metric("Based on", f"{data_points} records" if data_points > 0 else "Domain estimate")

    st.divider()
    st.markdown("**📂 Past Similar Events in Bengaluru:**")
    if len(past_events) > 0:
        st.dataframe(past_events, use_container_width=True)
    else:
        st.info("No past events found for this combination")

    st.divider()

    # ---- SECTION 2: Resources ----
    st.markdown('<div class="section-header">🧠 Resource Allocation Intelligence</div>', unsafe_allow_html=True)
    st.caption(f"For {inp['crowd_size']:,} people — {risk} risk — {inp['event_cause'].replace('_',' ').title()}")

    res_cols = st.columns(3)
    for i, (resource, count) in enumerate(resources.items()):
        res_cols[i % 3].metric(resource, count)

    st.divider()

    # ---- SECTION 3: Diversion Simulator ----
    st.markdown('<div class="section-header">🔀 Dynamic Diversion Simulator</div>', unsafe_allow_html=True)

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
        height=350, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    savings = scenarios["No Diversion"] - scenarios[best]
    st.success(f"✅ Best Strategy: **{best}** — saves ~{savings} minutes vs no action")

    st.divider()

    # ---- SECTION 4: Map ----
    st.markdown('<div class="section-header">🗺️ Affected Area & Diversion Routes</div>', unsafe_allow_html=True)
    st.caption(f"📍 {inp['formatted_address']} | 🔴 Affected zone | 🟢 Route B | 🟣 Early closure")

    map_html = build_mappls_map(
        inp["lat"], inp["lng"], risk,
        duration_median, inp["location_input"]
    )
    components.html(map_html, height=520)

else:
    st.info("👈 Fill in event details on the left and click **Simulate Event Impact**")
    st.markdown("""
    ### What this system does
    Before approving any event, police enter details about the event.
    The system instantly provides:
    - 🎯 **Risk Score** (0–100) with confidence breakdown
    - 📋 **Recommended actions** based on risk level
    - ⏱️ **Predicted event duration** from past similar events
    - 📂 **Past similar events** in Bengaluru
    - 🧠 **Full resource allocation** (officers, ambulance, fire brigade...)
    - 🗺️ **Live Mappls map** with exact location, affected area & real diversion routes
    """)