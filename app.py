import shap
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
import traffic_network as tn
import folium
from streamlit_folium import st_folium


def make_line_sparkline(data_points, bg_color):
    fig = go.Figure(go.Scatter(
        y=data_points,
        mode="lines+markers",
        line=dict(color="#ffffff", width=2.0),
        marker=dict(size=4, color="#ffffff"),
        hoverinfo="none"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=45,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
    )
    return fig

def make_area_sparkline(data_points, bg_color):
    fig = go.Figure(go.Scatter(
        y=data_points,
        mode="lines",
        line=dict(color="#ffffff", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255, 255, 255, 0.15)",
        hoverinfo="none"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=45,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
    )
    return fig

def make_bar_sparkline(data_points, bg_color):
    fig = go.Figure(go.Bar(
        y=data_points,
        marker_color="rgba(255, 255, 255, 0.65)",
        hoverinfo="none"
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=5, b=5),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        height=45,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
    )
    return fig

def make_past_event_card(row):
    risk_level = row.get("risk_level", "Moderate")
    accent = {"Minor": "#2e7d32", "Moderate": "#e65100", "Severe": "#c62828"}.get(risk_level, "#3B5BFF")
    icon = {"Minor": "https://img.icons8.com/color/48/checked.png",
            "Moderate": "https://img.icons8.com/color/48/warning-shield.png",
            "Severe": "https://img.icons8.com/color/48/high-importance.png"}.get(risk_level, "https://img.icons8.com/color/48/info.png")
    if pd.notna(row["hour"]):
        hour = int(row["hour"])
        hour_label = f"{hour % 12 or 12} {'AM' if hour < 12 else 'PM'}"
    else:
        hour_label = "Unknown time"
    day_label = row["day_of_week"] if pd.notna(row["day_of_week"]) else "Unknown day"
    crowd_label = row["crowd_proxy"] if pd.notna(row["crowd_proxy"]) else "Unknown"
    zone_label = row["zone"] if pd.notna(row["zone"]) else "Unknown zone"
    event_label = row["event_cause"].replace("_", " ").title() if pd.notna(row["event_cause"]) else "Unknown event"
    return f"""
    <div style="background-color:#222437;border:1px solid #2f3149;border-left:3px solid {accent};border-radius:10px;padding:16px 18px;height:100%;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <img src="{icon}" width="20" height="20"/>
                <b style="color:#ffffff;font-family:Archivo,Inter,sans-serif;font-size:14px;">{event_label}</b>
            </div>
            <span style="background-color:{accent};color:#ffffff;font-size:10.5px;font-weight:600;padding:2px 9px;border-radius:999px;">{risk_level}</span>
        </div>
        <div style="color:#f8f9fa;font-size:13px;margin-bottom:6px;line-height:1.4;">📍 {row["address"]}</div>
        <div style="color:#a5a6b4;font-size:12.5px;margin-bottom:4px;">Zone: <span style="color:#f8f9fa;font-weight:500;">{zone_label}</span></div>
        <div style="display:flex;gap:14px;margin-top:10px;">
            <div style="color:#a5a6b4;font-size:12px;">🕒 <span style="color:#f8f9fa;">{hour_label}, {day_label}</span></div>
            <div style="color:#a5a6b4;font-size:12px;">👥 <span style="color:#f8f9fa;">{crowd_label} crowd</span></div>
        </div>
    </div>
    """

def make_coreui_card(title, value, percentage, color, is_up=True):
    arrow = "&uarr;" if is_up else "&darr;"
    sign = "+" if is_up else ""
    return f"""
    <div style="background-color: {color}; border-radius: 6px 6px 0px 0px; padding: 15px 15px 5px 15px; color: #ffffff; font-family: 'Inter', sans-serif; height: 75px; display: flex; flex-direction: column; justify-content: space-between; border: 1px solid #2f3149; border-bottom: none;">
        <div style="display: flex; align-items: baseline; gap: 6px; justify-content: space-between;">
            <span style="font-size: 20px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;">{value}</span>
            <span style="font-size: 11px; opacity: 0.9; font-weight: 500;">({sign}{percentage} {arrow})</span>
        </div>
        <div style="font-size: 11px; opacity: 0.85; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;">{title}</div>
    </div>
    """

def make_traffic_chart(df_data, hour_filter, risk_score):
    hours = list(range(24))
    day48_demand = [0.12, 0.08, 0.05, 0.03, 0.06, 0.15, 0.35, 0.62, 0.78, 0.65, 0.55, 0.58, 
                    0.61, 0.59, 0.52, 0.58, 0.72, 0.85, 0.92, 0.81, 0.65, 0.42, 0.25, 0.18]
    
    day49_prediction = [d * (1.0 + (risk_score / 200.0)) if h >= hour_filter else d for h, d in zip(hours, day48_demand)]
    day49_prediction = [min(1.0, d) for d in day49_prediction]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=day48_demand,
        name="Day 48 Baseline (Actual)",
        line=dict(color="#3399ff", width=2.5),
        mode="lines"
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=day49_prediction,
        name="Day 49 Forecast (Predicted)",
        line=dict(color="#2ecc71", width=2.5),
        mode="lines"
    ))
    
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Hour of Day", font=dict(color="#a5a6b4", size=11)),
            tickfont=dict(color="#a5a6b4", size=10),
            gridcolor="#2f3149",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Congestion Index / Demand", font=dict(color="#a5a6b4", size=11)),
            tickfont=dict(color="#a5a6b4", size=10),
            gridcolor="#2f3149",
            zeroline=False
        ),
        legend=dict(
            font=dict(color="#a5a6b4", size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=10, b=40),
        plot_bgcolor="#222437",
        paper_bgcolor="#222437",
        height=280,
    )
    return fig

load_dotenv()
MAPPLS_KEY = st.secrets.get("MAPPLS_API_KEY") or os.getenv("MAPPLS_API_KEY")


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

# Inject Modern CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #FFFFFF;
        --bg: #181924;
        --surface: #222437;
        --accent: #3B5BFF;
        --text: #F8F9FA;
        --muted: #A5A6B4;
        --hairline: #2F3149;
        --minor: #2E7D32;
        --moderate: #E65100;
        --severe: #C62828;
        --radius: 8px;
    }

    /* ---- Base / typography ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--text);
    }
    .stApp {
        background-color: #181924 !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Archivo', 'Inter', sans-serif;
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    h1 { font-weight: 800; }
    h2, h3 { font-weight: 700; }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* Tabular numerals everywhere a number lives */
    .metric-value, [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-feature-settings: "tnum";
    }

    /* ---- Divider ---- */
    hr, [data-testid="stDivider"] {
        border-color: var(--hairline) !important;
    }

    /* ---- Metric cards (signature: colored status rail) ---- */
    .metric-card {
        background-color: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: var(--radius);
        padding: 18px 20px;
        box-shadow: none;
        margin-bottom: 12px;
        transition: border-color 0.15s ease;
    }
    .metric-title {
        font-size: 11.5px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: var(--ink);
        line-height: 1.1;
    }

    /* ---- Alert / info cards ---- */
    .alert-card {
        background-color: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: var(--radius);
        padding: 22px 24px;
        box-shadow: none;
        margin-bottom: 20px;
    }

    .custom-icon {
        vertical-align: middle;
        margin-right: 8px;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background-color: #1d2030 !important;
        border-right: 1px solid var(--hairline) !important;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #ffffff;
    }
    [data-testid="stWidgetLabel"] p, label p {
        color: #f8f9fa !important;
        font-weight: 500 !important;
    }

    /* ---- Inputs ---- */
    .stTextInput input, .stNumberInput input, .stSelectbox > div > div,
    .stTextArea textarea {
        border-radius: 6px !important;
        border: 1px solid var(--hairline) !important;
        background-color: #222437 !important;
        color: #ffffff !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* ---- Slider ---- */
    .stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

    /* ---- Buttons ---- */
    .stButton button, .stDownloadButton button {
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.01em;
        border: none;
        transition: opacity 0.15s ease;
    }
    .stButton button[kind="primary"], .stDownloadButton button {
        background-color: var(--accent) !important;
        color: #fff !important;
    }
    .stButton button[kind="primary"]:hover, .stDownloadButton button:hover {
        opacity: 0.85;
    }

    /* ---- Tabs: underline style, not pill style ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 28px;
        border-bottom: 1px solid var(--hairline);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        font-weight: 600;
        font-size: 14px;
        color: var(--muted);
        padding: 8px 2px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        border-bottom: 2px solid var(--accent) !important;
        background: transparent !important;
    }

    /* ---- Dataframes ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--hairline);
        border-radius: var(--radius);
    }

    /* ---- Status chips (situation_chips) ---- */
    .status-chip {
        display:inline-flex; align-items:center;
        background: var(--surface);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        padding: 6px 14px;
        margin: 0 8px 8px 0;
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# GEOCODING — NEW: converts address text to lat/lng
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode_location(address_text):
    """Uses OpenStreetMap Nominatim — free, no key needed."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": f"{address_text}, Bengaluru, India",
                "format": "json",
                "limit": 1,
            },
            headers={"User-Agent": "GridlockSentinel/1.0"},
            timeout=5
        )
        if resp.status_code != 200:
            return None
        results = resp.json()
        if not results:
            return None
        result = results[0]
        lat = float(result.get("lat", 0))
        lng = float(result.get("lon", 0))
        formatted = result.get("display_name", address_text)
        if lat and lng:
            return lat, lng, formatted
        return None
    except Exception:
        return None

def lat_lng_to_zone(lat, lng):
    """Derives zone from coordinates for ML model."""
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
# NEARBY FACILITIES via OpenStreetMap Overpass API
# -------------------------------------------------------
OVERPASS_TAGS = {
    "police station": '"amenity"="police"',
    "hospital": '"amenity"="hospital"',
}

def get_nearby_facilities(lat, lng, keyword, radius=3000, limit=2):
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
            return []
        elements = resp.json().get("elements", [])
        facilities = []
        for el in elements[:limit]:
            tags = el.get("tags", {})
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
        return facilities
    except Exception:
        return []

def build_mappls_map(lat, lng, risk_level, risk_score, incident_node, target_hospital, routing_res, corridor_res, dispatch_res):
    radius_meters = {
        "Minor": 400, "Moderate": 750, "Severe": 1200
    }.get(risk_level, 500)

    risk_color = {"Minor": "#2e7d32", "Moderate": "#e65100", "Severe": "#c62828"}.get(risk_level, "#c62828")

    # Format routes as JavaScript coordinate arrays
    std_coords_js = "[]"
    has_routing_js = "false"
    congested_coords_js = "[]"
    if routing_res:
        has_routing_js = "true"
        std_coords_js = "[" + ", ".join(f"{{ lat: {tn.NODE_COORDS[node][0]}, lng: {tn.NODE_COORDS[node][1]} }}" for node in routing_res["std_path"]) + "]"
        congested_coords_js = "[" + ", ".join(f"{{ lat: {tn.NODE_COORDS[node][0]}, lng: {tn.NODE_COORDS[node][1]} }}" for node in routing_res["congested_path"]) + "]"

    corridor_coords_js = "[]"
    has_corridor_js = "false"
    hosp_lat, hosp_lng = 0.0, 0.0
    corridor_eta = 0.0
    if corridor_res:
        has_corridor_js = "true"
        corridor_coords_js = "[" + ", ".join(f"{{ lat: {coord[0]}, lng: {coord[1]} }}" for coord in corridor_res["coords_path"]) + "]"
        hosp_coords = tn.HOSPITAL_COORDS.get(target_hospital)
        if hosp_coords:
            hosp_lat, hosp_lng = hosp_coords
        corridor_eta = corridor_res["eta_mins"]

    # Format police markers
    police_markers_js = ""
    if dispatch_res:
        for d in dispatch_res:
            station_name = d["station"]
            if d["officers_dispatched"] > 0 or d["cars_dispatched"] > 0:
                stat_coords = tn.POLICE_STATIONS[station_name]["coords"]
                police_markers_js += f"""
                new mappls.Marker({{
                    map: map,
                    position: {{ lat: {stat_coords[0]}, lng: {stat_coords[1]} }},
                    popupHtml: '<b>\U0001f46e Dispatched Depot</b><br>{station_name}<br>Officers: {d["officers_dispatched"]}, Cars: {d["cars_dispatched"]}',
                    icon: 'https://img.icons8.com/color/48/police-badge.png',
                    width: 32,
                    height: 32
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
            background: #222437; padding: 12px 16px; border-radius: 8px;
            border: 1px solid #2f3149; font-size: 12px; font-family: Arial;
            z-index: 999; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            color: #ffffff;
        }}
        .legend-item {{ display:flex; align-items:center; margin: 5px 0; }}
        .dot {{ width:14px; height:14px; border-radius:50%; margin-right:8px; flex-shrink:0; }}
        #map-error {{
            display:none; padding: 14px; font-family: Arial, sans-serif; font-size: 13px;
            color:#c62828; background:#fff3f3; border:1px solid #ffcdd2; border-radius:8px; margin:10px;
        }}
    </style>
</head>
<body>
<div id="map-error"></div>
<div id="map"></div>
<div class="legend">
    <b>Map Legend</b><br>
    <div class="legend-item"><div class="dot" style="background:{risk_color}"></div> Affected Zone ({risk_level})</div>
    <div class="legend-item"><div class="dot" style="border: 2px dashed #e55353; background: transparent; width: 10px; height: 10px;"></div> Standard Route (Stuck)</div>
    <div class="legend-item"><div class="dot" style="background:#2ecc71"></div> Diversion Bypass Route</div>
    <div class="legend-item"><div class="dot" style="background:#3399ff"></div> Emergency Green Corridor</div>
</div>

<script>
var mapErrors = [];

function showMapError(msg) {{
    mapErrors.push(msg);
    var el = document.getElementById('map-error');
    el.style.display = 'block';
    el.innerHTML = '<b>Map issues:</b><br>' + mapErrors.map(function(m) {{ return '• ' + m; }}).join('<br>');
}}

function initMap() {{
    if (window.__gridlockMapInitialized) return;
    window.__gridlockMapInitialized = true;

    if (typeof mappls === 'undefined') {{
        showMapError('Mappls SDK script did not load (check MAPPLS_API_KEY / network / domain restrictions).');
        return;
    }}

    var map;
    try {{
        map = new mappls.Map('map', {{
            center: {{ lat: {lat}, lng: {lng} }},
            zoom: 13,
            search: false
        }});
    }} catch (err) {{
        showMapError('Map init failed: ' + err.message);
        return;
    }}

    function drawAllLayers() {{
        // Event Origin Marker
        try {{
            new mappls.Marker({{
                map: map,
                position: {{ lat: {lat}, lng: {lng} }},
                popupHtml: '<b>\U0001f4cd Event Origin ({incident_node})</b><br>Risk: {risk_level}<br>Score: {risk_score}/100',
                icon: 'https://img.icons8.com/color/48/traffic-light.png',
                width: 36,
                height: 36
            }});
        }} catch (err) {{
            showMapError('Event marker failed: ' + err.message);
        }}

        // Affected Buffer Zone Circle
        try {{
            new mappls.Circle({{
                map: map,
                center: {{ lat: {lat}, lng: {lng} }},
                radius: {radius_meters},
                strokeColor: '{risk_color}',
                strokeOpacity: 1,
                strokeWeight: 3,
                fillColor: '{risk_color}',
                fillOpacity: 0.25
            }});
        }} catch (err) {{
            showMapError('Risk circle failed: ' + err.message);
        }}

        // Standard Commuter Route (Stuck)
        if ({has_routing_js}) {{
            try {{
                new mappls.Polyline({{
                    map: map,
                    path: {std_coords_js},
                    strokeColor: '#e55353',
                    strokeOpacity: 0.8,
                    strokeWeight: 4,
                    strokeDasharray: '10, 10'
                }});
            }} catch (err) {{
                showMapError('Standard route failed: ' + err.message);
            }}

            // Diversion Route
            try {{
                new mappls.Polyline({{
                    map: map,
                    path: {congested_coords_js},
                    strokeColor: '#2ecc71',
                    strokeOpacity: 0.9,
                    strokeWeight: 5
                }});
            }} catch (err) {{
                showMapError('Diversion route failed: ' + err.message);
            }}
        }}

        // Emergency Corridor Route & Destination Hospital
        if ({has_corridor_js}) {{
            try {{
                new mappls.Polyline({{
                    map: map,
                    path: {corridor_coords_js},
                    strokeColor: '#3399ff',
                    strokeOpacity: 0.95,
                    strokeWeight: 5
                }});
                
                new mappls.Marker({{
                    map: map,
                    position: {{ lat: {hosp_lat}, lng: {hosp_lng} }},
                    popupHtml: '<b>\U0001f3e5 Hospital Destination</b><br>{target_hospital}<br>ETA: {corridor_eta} mins',
                    icon: 'https://img.icons8.com/color/48/hospital.png',
                    width: 32,
                    height: 32
                }});
            }} catch (err) {{
                showMapError('Emergency corridor failed: ' + err.message);
            }}
        }}

        // Police Station Markers
        try {{
            {police_markers_js}
        }} catch (err) {{
            showMapError('Police markers failed: ' + err.message);
        }}
    }}

    var layersDrawn = false;
    function drawOnce() {{
        if (layersDrawn) return;
        layersDrawn = true;
        drawAllLayers();
    }}

    try {{
        map.on('load', drawOnce);
    }} catch (err) {{
        showMapError('Could not attach load listener: ' + err.message);
    }}

    setTimeout(drawOnce, 1500);
}}

setTimeout(function() {{
    var mapDiv = document.getElementById('map');
    if (mapDiv && mapDiv.children.length === 0) {{
        if (typeof mappls !== 'undefined') {{
            initMap();
        }} else {{
            showMapError('Timed out waiting for the Mappls SDK to load. Verify MAPPLS_API_KEY is correct.');
        }}
    }}
}}, 6000);
</script>

<script
    src="https://apis.mappls.com/advancedmaps/api/{MAPPLS_KEY}/map_sdk?v=3.0&layer=vector&callback=initMap"
    onerror="showMapError('Could not load the Mappls SDK script tag — check API key validity.')">
</script>
</body>
</html>
"""
    return html

def build_folium_map(lat, lng, risk_level, risk_score, incident_node, target_hospital, routing_res, corridor_res, dispatch_res):
    radius_meters = {
        "Minor": 400, "Moderate": 750, "Severe": 1200
    }.get(risk_level, 500)

    risk_color = {"Minor": "#2e7d32", "Moderate": "#e65100", "Severe": "#c62828"}.get(risk_level, "#c62828")

    # Create base map with CartoDB dark_matter tiles
    m = folium.Map(location=[lat, lng], zoom_start=13, tiles="CartoDB dark_matter")

    # Add Affected Zone Circle
    folium.Circle(
        location=[lat, lng],
        radius=radius_meters,
        color=risk_color,
        fill=True,
        fill_color=risk_color,
        fill_opacity=0.25,
        tooltip=f"Incident Buffer Zone ({risk_level} Risk)",
        popup=f"Risk Score: {risk_score}/100"
    ).add_to(m)

    # Add Event Location Marker
    folium.Marker(
        location=[lat, lng],
        tooltip="📍 Event/Incident Origin",
        icon=folium.Icon(color="red" if risk_level == "Severe" else "orange" if risk_level == "Moderate" else "green", icon="exclamation-sign")
    ).add_to(m)

    # Draw all graph nodes
    for name, coords in tn.NODE_COORDS.items():
        if name == incident_node:
            continue
        folium.CircleMarker(
            location=coords,
            radius=5,
            color="#a5a6b4",
            fill=True,
            fill_color="#1d2030",
            fill_opacity=0.8,
            popup=name,
            tooltip=name
        ).add_to(m)

    # Draw all graph road edges
    for u, v in tn.ROAD_EDGES:
        u_coords = tn.NODE_COORDS[u]
        v_coords = tn.NODE_COORDS[v]
        folium.PolyLine(
            locations=[u_coords, v_coords],
            color="#2f3149",
            weight=1.5,
            opacity=0.6,
            tooltip=f"Road: {u} - {v}"
        ).add_to(m)

    # Draw Commuter Path Comparison (Standard vs diversion)
    if routing_res:
        std_coords = [tn.NODE_COORDS[node] for node in routing_res["std_path"]]
        folium.PolyLine(
            locations=std_coords,
            color="#e55353",
            weight=3.5,
            dash_array="6, 6",
            opacity=0.8,
            tooltip="Standard Path (Gridlocked)"
        ).add_to(m)

        congested_coords = [tn.NODE_COORDS[node] for node in routing_res["congested_path"]]
        folium.PolyLine(
            locations=congested_coords,
            color="#2ecc71",
            weight=4.5,
            opacity=0.9,
            tooltip="Diversion Route (Bypass)"
        ).add_to(m)

    # Draw Emergency Green Corridor Path
    if corridor_res:
        corridor_coords = corridor_res["coords_path"]
        folium.PolyLine(
            locations=corridor_coords,
            color="#3399ff",
            weight=5.0,
            opacity=0.95,
            tooltip="Emergency Green Corridor"
        ).add_to(m)

        hosp_coords = tn.HOSPITAL_COORDS.get(target_hospital)
        if hosp_coords:
            folium.Marker(
                location=hosp_coords,
                tooltip=f"🏥 Hospital Destination: {target_hospital}",
                popup=f"ETA: {corridor_res['eta_mins']} mins",
                icon=folium.Icon(color="blue", icon="plus-sign")
            ).add_to(m)

    # Draw Dispatched Police Stations
    if dispatch_res:
        for d in dispatch_res:
            station_name = d["station"]
            if d["officers_dispatched"] > 0 or d["cars_dispatched"] > 0:
                stat_coords = tn.POLICE_STATIONS[station_name]["coords"]
                folium.Marker(
                    location=stat_coords,
                    tooltip=f"👮 Dispatched Depot: {station_name}",
                    popup=f"Officers: {d['officers_dispatched']}, Cars: {d['cars_dispatched']}",
                    icon=folium.Icon(color="darkblue", icon="info-sign")
                ).add_to(m)

    return m

# -------------------------------------------------------
# CORE FUNCTIONS (unchanged from teammate's version)
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
    return {"risk_level": pred, "risk_score": risk_score, "confidence": proba_dict, "crowd_level": crowd_level}

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
        "event_cause": "Event Type", "crowd_proxy": "Crowd Size",
        "zone": "Zone/Location", "hour": "Time of Day",
        "day_of_week": "Day of Week", "is_weekend": "Weekend?",
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
        "vip_movement": 60, "accident": 45, "vehicle_breakdown": 30,
        "congestion": 90, "construction": 240, "tree_fall": 60,
        "water_logging": 120, "pot_holes": 30, "road_conditions": 60,
        "Debris": 45, "debris": 45, "others": 60,
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
        "Traffic Officers":      max(2, crowd_size // 500),
        "Barricades":            max(2, crowd_size // 1200),
        "Patrol Vehicles":       max(1, crowd_size // 3000),
        "Ambulances on Standby": max(1, crowd_size // 5000),
        "Fire Brigade Units":    1 if crowd_size > 5000 or risk_level == "Severe" else 0,
        "CCTV/Surveillance":     max(2, crowd_size // 4000),
        "First Aid Posts":       max(1, crowd_size // 8000),
        "Bus Route Diversions":  max(1, crowd_size // 6000),
        "PA System Units":       max(1, crowd_size // 10000),
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
    if score >= 75: return "Severe"
    if score >= 45: return "Moderate"
    return "Minor"

def adjust_operational_risk(base_score, event_cause, hour, mode, lead_time_min, lanes_blocked, rain_watch):
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
        return {"icon": "https://img.icons8.com/color/48/high-importance.png", "title": "High Alert", "caption": "Control room should treat this as a priority disruption.", "color": "#C62828"}
    if risk_level == "Moderate":
        return {"icon": "https://img.icons8.com/color/48/warning-shield.png", "title": "Watch Closely", "caption": "Extra deployment and active monitoring are recommended.", "color": "#EF6C00"}
    if mode == "Unplanned Incident":
        return {"icon": "https://img.icons8.com/color/48/info--v1.png", "title": "Verify Fast", "caption": "Risk is low, but live incidents still need quick confirmation.", "color": "#1565C0"}
    return {"icon": "https://img.icons8.com/color/48/checked.png", "title": "Manageable", "caption": "Standard deployment should be enough with routine monitoring.", "color": "#2E7D32"}

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
    if int(inputs["hour"]) in list(range(8, 11)) + list(range(17, 22)):
        chips.append(("https://img.icons8.com/color/48/alarm-clock.png", "Peak hour"))
    status_icon = {"Minor": "https://img.icons8.com/color/48/checked.png", "Moderate": "https://img.icons8.com/color/48/warning-shield.png", "Severe": "https://img.icons8.com/color/48/high-importance.png"}[risk_level]
    chips.append((status_icon, f"{risk_level} risk"))
    return chips

def make_command_brief(inputs, risk_score, risk_level, duration_min, busy_until, best_strategy, savings, resources, timeline, operational_reasons):
    resource_lines = "\n".join(f"- {name}: {count}" for name, count in resources.items())
    timeline_lines = "\n".join(f"- {item}" for item in timeline)
    reason_lines = "\n".join(f"- {item}" for item in operational_reasons) or "- No extra operational risk factors"
    busy_until_str = f"{int(busy_until):02d}"
    return f"""# Gridlock Command Brief

## Scenario
- Mode: {inputs['mode']}
- Event/Incident Type: {inputs['event_cause']}
- Location: {inputs.get('formatted_address', inputs.get('zone', 'N/A'))}
- Zone: {inputs['zone']}
- Crowd / impact estimate: {inputs['crowd_size']:,}
- Start time: {inputs['hour_display']}
- Day: {inputs['day']}
- Lanes blocked: {inputs['lanes_blocked']}
- Rain / waterlogging watch: {'Yes' if inputs['rain_watch'] else 'No'}

## Predicted Impact
- Operational risk score: {risk_score}/100
- Risk level: {risk_level}
- Typical disruption duration: {duration_min} minutes
- Area busy until: ~{busy_until_str}:00
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
        {"name": "Evening Rally Surge", "mode": "Planned Event", "event_cause": "public_event", "zone": "Central Zone 1", "crowd_size": 20000, "hour": 18, "minute": "00", "day": "Saturday", "lead_time": 120, "lanes_blocked": 0, "rain_watch": False, "location": "Kanteerava Stadium, Bengaluru"},
        {"name": "Rainy Accident Response", "mode": "Unplanned Incident", "event_cause": "accident", "zone": "East Zone 1", "crowd_size": 6000, "hour": 9, "minute": "00", "day": "Monday", "lead_time": 0, "lanes_blocked": 2, "rain_watch": True, "location": "Old Airport Road, Bengaluru"},
        {"name": "VIP Movement Prep", "mode": "Planned Event", "event_cause": "vip_movement", "zone": "Central Zone 2", "crowd_size": 5000, "hour": 17, "minute": "00", "day": "Friday", "lead_time": 240, "lanes_blocked": 0, "rain_watch": False, "location": "Raj Bhavan, Bengaluru"},
    ]

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
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

EVENT_CAUSES = sorted([
    'accident', 'congestion', 'construction', 'Debris',
    'Fog / Low Visibility', 'others', 'pot_holes', 'procession',
    'protest', 'public_event', 'road_conditions', 'tree_fall',
    'vehicle_breakdown', 'vip_movement', 'water_logging'
])
ZONES = sorted(list(tn.NODE_COORDS.keys()))
DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
HOURS = list(range(0, 24))
MINUTES = ["00", "15", "30", "45"]
PRESETS = {preset["name"]: preset for preset in get_demo_presets()}

# SIDEBAR
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0px 20px 0px; text-align: left;">
            <div style="font-size: 20px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 8px;">
                <img src="https://img.icons8.com/color/48/traffic-light.png" width="28" height="28"/>
                COREUI TRAFFIC
            </div>
        </div>
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 12px; background: #222437; border-radius: 6px; color: #ffffff; font-weight: 500; font-size: 14px; cursor: pointer;">
                <span>📂 Dashboard</span>
                <span style="background: #3b5bff; font-size: 10px; padding: 2px 6px; border-radius: 4px; font-weight: bold;">NEW</span>
            </div>
        </div>

        <div style="font-size: 11px; font-weight: 700; color: #a5a6b4; letter-spacing: 0.05em; margin-bottom: 8px; text-transform: uppercase;">Simulation Forms</div>
        """,
        unsafe_allow_html=True
    )

    preset_name = st.selectbox("Demo Scenario Preset", ["Custom"] + list(PRESETS.keys()), index=1, help="Use a ready-made situation for fast judging demos")
    preset = PRESETS.get(preset_name, {})

    mode = st.radio("Command Mode", ["Planned Event", "Unplanned Incident"],
                    index=["Planned Event", "Unplanned Incident"].index(preset.get("mode", "Planned Event")),
                    horizontal=True)

    event_cause = st.selectbox("Event Type", EVENT_CAUSES,
                               index=EVENT_CAUSES.index(preset.get("event_cause", "public_event")))

    # Location input defaults to preset value
    location_input = st.text_input(
        "Event Location",
        value=preset.get("location", "Kanteerava Stadium, Bengaluru"),
        placeholder="e.g. Lalbagh Botanical Garden, MG Road...",
        help="Type any Bengaluru landmark or address"
    )

    crowd_size = st.number_input(
        "Expected Crowd / Impact Size",
        min_value=10, max_value=500000,
        value=preset.get("crowd_size", 20000), step=500,
        help="For incidents, use estimated affected road users or queue impact"
    )

    st.markdown("**Event Start Time**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        hour = st.selectbox("Hour", HOURS, index=preset.get("hour", 18),
                            format_func=lambda x: f"{x:02d}")
    with t_col2:
        minute_str = st.selectbox("Min", MINUTES,
                                  index=MINUTES.index(preset.get("minute", "00")))

    day = st.selectbox("Day of Week", DAYS, index=DAYS.index(preset.get("day", "Saturday")))

    lead_time = preset.get("lead_time", 120)
    lanes_blocked = 0
    if mode == "Planned Event":
        lead_time = st.slider("Preparation Lead Time (minutes)", min_value=0, max_value=360, value=lead_time, step=15)
    else:
        lanes_blocked = st.slider("Lanes Blocked", 0, 4, preset.get("lanes_blocked", 1))

    rain_watch = st.checkbox("Rain / waterlogging watch", value=preset.get("rain_watch", False))

    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; margin-bottom: 5px;">
            <img src="https://img.icons8.com/color/48/hospital.png" width="20" height="20"/>
            <b style="font-size:14px;">Green Corridor Settings</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Geocode current location input (cached)
    geo_lat, geo_lng = 12.9716, 77.5946
    formatted_address = f"{location_input} (Fallback Coordinates)"
    current_zone = "Central Zone 1"
    geo_result = None

    if location_input.strip():
        geo_result = geocode_location(location_input.strip())
        if geo_result:
            geo_lat, geo_lng, formatted_address = geo_result
            current_zone = lat_lng_to_zone(geo_lat, geo_lng)
        else:
            st.sidebar.warning(f"Could not locate '{location_input}'. Using default coordinates.")

    # 2. Dynamic closest hospital recommendation
    closest_hospital = min(tn.HOSPITAL_COORDS.keys(), key=lambda h: tn.haversine_distance((geo_lat, geo_lng), tn.HOSPITAL_COORDS[h]))
    hosp_options = list(tn.HOSPITAL_COORDS.keys())
    target_hospital = st.selectbox("Emergency Hospital Destination", options=hosp_options,
                                   index=hosp_options.index(closest_hospital),
                                   help="System auto-recommends the closest facility, but you can override it.")

    st.markdown("---")

    # Keep a refresh button that clears cache
    if st.button("Force Geocode Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    hour_decimal = hour + int(minute_str) / 60
    inp = {
        "mode": mode,
        "event_cause": event_cause,
        "location_input": location_input,
        "formatted_address": formatted_address,
        "lat": geo_lat,
        "lng": geo_lng,
        "zone": current_zone,
        "crowd_size": crowd_size,
        "hour": hour_decimal,
        "hour_display": f"{hour:02d}:{minute_str}",
        "day": day,
        "lead_time": lead_time,
        "lanes_blocked": lanes_blocked,
        "rain_watch": rain_watch,
        "target_hospital": target_hospital,
    }

# MAIN CONTENT
if True:
    with st.spinner("Running system simulation models..."):
        risk_result = predict_event_risk(inp["event_cause"], inp["crowd_size"], inp["zone"], inp["hour"], inp["day"])
        model_risk = risk_result["risk_level"]
        model_risk_score = risk_result["risk_score"]

        risk_score, risk, operational_reasons = adjust_operational_risk(
            model_risk_score, inp["event_cause"], inp["hour"],
            inp["mode"], inp["lead_time"], inp["lanes_blocked"], inp["rain_watch"]
        )
        crowd_level = risk_result["crowd_level"]

        duration_median, duration_min, duration_max, data_points = predict_duration(inp["event_cause"])
        busy_until = int(inp["hour"] + duration_median // 60) % 24

        resources_base = recommend_resources(inp["crowd_size"], risk, inp["event_cause"])

        commuter_origin = "West Zone 1"
        commuter_destination = "East Zone 2"
        if inp["zone"] == "West Zone 1":
            commuter_origin = "Central Zone 2"
        elif inp["zone"] == "East Zone 2":
            commuter_destination = "East Zone 1"

        routing_res = tn.get_routing_scenarios(G, source=commuter_origin, target=commuter_destination, incident_node=inp["zone"], risk_score=risk_score)
        corridor_res = tn.get_emergency_corridor(G, incident_zone=inp["zone"], hospital_name=inp["target_hospital"], risk_score=risk_score)

        officers_needed = resources_base.get("Traffic Officers", 5)
        cars_needed = resources_base.get("Patrol Vehicles", 2)
        dispatch_res, unmet_off, unmet_cars = tn.optimize_police_dispatch(required_officers=officers_needed, required_cars=cars_needed, incident_zone=inp["zone"], G=G)
        transit_res = tn.check_bmtc_transit(G, incident_node=inp["zone"], risk_score=risk_score)

        best_strat = "Active Diversion Plan" if routing_res and routing_res["savings_mins"] > 0 else "Standard Monitoring"
        timeline = build_response_timeline(inp["mode"], risk, best_strat, inp["lead_time"])
        command_brief = make_command_brief(inp, risk_score, risk, duration_median, busy_until, best_strat, routing_res["savings_mins"] if routing_res else 0, resources_base, timeline, operational_reasons)
        command_mood = get_command_mood(risk, inp["mode"])
        situation_chips = get_situation_chips(inp, risk)

    # CoreUI Top Header Bar
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #222437; padding: 12px 24px; border-radius: 8px; border: 1px solid #2f3149; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 24px; color: #ffffff; font-size: 14px; font-weight: 500;">
                <span style="font-size: 18px; cursor: pointer; color: #a5a6b4;">☰</span>
                <span style="cursor: pointer; color: #a5a6b4;">Dashboard</span>
                <span style="cursor: pointer; color: #a5a6b4;">Users</span>
                <span style="cursor: pointer; color: #a5a6b4;">Settings</span>
            </div>
            <div style="display: flex; align-items: center; gap: 18px;">
                <img src="https://img.icons8.com/color/48/alarm.png" width="20" height="20" style="cursor: pointer;"/>
                <img src="https://img.icons8.com/color/48/task.png" width="20" height="20" style="cursor: pointer;"/>
                <img src="https://img.icons8.com/color/48/speech-bubble.png" width="20" height="20" style="cursor: pointer;"/>
                <img src="https://img.icons8.com/color/48/moon.png" width="20" height="20" style="cursor: pointer;"/>
                <img src="https://img.icons8.com/color/48/user-male-circle.png" width="28" height="28" style="cursor: pointer; border-radius: 50%; border: 1px solid #2f3149;"/>
            </div>
        </div>
        <div style="font-size: 13px; color: #a5a6b4; margin-bottom: 20px; font-weight: 500;">
            <span style="color: #3b5bff; cursor: pointer;">Home</span> / Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show resolved location info bar
    st.info(f"📍 **Location:** {inp['formatted_address']} | 🕐 **Time:** {inp['hour_display']} {inp['day']} | 🗺️ **Zone:** {inp['zone']}")

    # KPI HEADER
    st.markdown(
        f"""
        <div style="border: 1px solid #2f3149; border-left: 4px solid {command_mood['color']};
                    background: #222437; padding: 20px 22px; border-radius: 8px;
                    box-shadow: none; margin-bottom: 16px;">
            <div style="display:flex; align-items:center; gap:16px;">
                <img src="{command_mood['icon']}" width="40" height="40"/>
                <div>
                    <div style="font-family:'Archivo','Inter',sans-serif; font-size:20px; font-weight:700; color:{command_mood['color']}; margin:0;">{command_mood['title']}</div>
                    <div style="font-size:14px; color:#a5a6b4; margin-top:2px;">{command_mood['caption']}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chip_html = ""
    for icon_url, label in situation_chips:
        chip_html += (
            f"<span class='status-chip'>"
            f"<img src='{icon_url}' width='15' height='15' style='margin-right:6px;'/>{label}</span>"
        )
    st.markdown(chip_html, unsafe_allow_html=True)

    # 4 COREUI KPI METRIC CARDS GRID
    severe_count = int((df["risk_level"] == "Severe").sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(make_coreui_card("Total Traffic Volume", "26.4K", "-12.4%", "#5856d6", is_up=False), unsafe_allow_html=True)
        st.plotly_chart(make_line_sparkline([22, 24, 23, 25, 27, 26, 28, 26, 29, 26], "#5856d6"), use_container_width=True, config={'displayModeBar': False})
    with c2:
        st.markdown(make_coreui_card("Bypass Delay Saved", f"{routing_res['savings_mins']} min" if routing_res else "0 min", "40.9%", "#3399ff", is_up=True), unsafe_allow_html=True)
        st.plotly_chart(make_line_sparkline([12, 14, 19, 23, 24, 31, 35, 42, 48, 52], "#3399ff"), use_container_width=True, config={'displayModeBar': False})
    with c3:
        st.markdown(make_coreui_card("Operational Risk Score", f"{risk_score}%", "84.7%", "#f9b115", is_up=True), unsafe_allow_html=True)
        st.plotly_chart(make_area_sparkline([10, 15, 22, 20, 28, 35, 30, 42, 45, 40], "#f9b115"), use_container_width=True, config={'displayModeBar': False})
    with c4:
        st.markdown(make_coreui_card("Severe Gridlocks", f"{severe_count}", "-23.6%", "#e55353", is_up=False), unsafe_allow_html=True)
        st.plotly_chart(make_bar_sparkline([2, 3, 2, 4, 3, 5, 4, 3, 5, 4], "#e55353"), use_container_width=True, config={'displayModeBar': False})

    # CENTRAL TRAFFIC ANALYTICS GRAPH CARD
    st.markdown(
        """
        <div style="background-color: #222437; border-radius: 8px 8px 0px 0px; border: 1px solid #2f3149; border-bottom: none; padding: 20px 24px 5px 24px; display: flex; justify-content: space-between; align-items: center; margin-top: 15px; font-family: 'Inter', sans-serif;">
            <div>
                <h3 style="margin:0; color:#ffffff; font-size:16px; font-weight:700;">Traffic</h3>
                <span style="font-size:12px; color:#a5a6b4;">Expected Event Horizon (24 Hrs)</span>
            </div>
            <div style="display:flex; gap:5px; align-items: center;">
                <span style="background-color:#2f3149; color:#ffffff; font-size:12px; padding:6px 12px; border-radius:4px; cursor:pointer;">Day</span>
                <span style="background-color:#3B5BFF; color:#ffffff; font-size:12px; padding:6px 12px; border-radius:4px; font-weight:bold; cursor:pointer;">Month</span>
                <span style="background-color:#2f3149; color:#ffffff; font-size:12px; padding:6px 12px; border-radius:4px; cursor:pointer;">Year</span>
                <span style="background-color:#2f3149; color:#ffffff; font-size:12px; padding:6px 8px; border-radius:4px; cursor:pointer; margin-left: 8px; font-weight: bold;">📥</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(make_traffic_chart(df, inp["hour"], risk_score), use_container_width=True)


    # TABS
    t1, t2, t3, t4, t5 = st.tabs([
        "🎭 Event Digital Twin", "🔀 Dynamic Routing",
        "🚑 Emergency Corridor", "👮 Police Dispatch Planner", "🚌 BMTC Transit Advisor"
    ])

    with t1:
        st.subheader("Event Risk Assessment Profile")
        st.caption(f"ML Model Base Prediction: {model_risk} Risk ({model_risk_score}/100 score). Operational adjustments applied on top.")

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.markdown("**Core Action Recommendations:**")
            actions = {
                "Minor": ["Deploy standard traffic officers at key junctions surrounding the zone.", "Keep patrol vehicles on standard regional patrol rotation.", "Monitor intersection density feeds via local surveillance feeds."],
                "Moderate": ["Pre-position road barriers at secondary road entrances 1 hour before scheduled time.", "Activate alternate route signage at adjacent network junctions.", "Position an ambulance on regional standby.", "Broadcast traffic advisory details through regional warning channels."],
                "Severe": ["Execute complete road closure around target zone 2 hours prior to event.", "Position emergency fire brigade and ambulance services at nearby nodes.", "Initiate detour/diversion routing across the local network.", "Broadcast alert advisory updates to local travel applications.", "Activate Green Corridor protocols for medical response."]
            }
            for act in actions[risk]:
                st.markdown(f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;"><img src="https://img.icons8.com/flat-round/24/checkmark.png" width="16" height="16" style="margin-top:3px;"/><span style="font-size:15px;color:#f8f9fa;">{act}</span></div>', unsafe_allow_html=True)
            if operational_reasons:
                st.markdown("<br>**Operational adjustment factors:**", unsafe_allow_html=True)
                for reason in operational_reasons:
                    st.write(f"- {reason}")
        with col_right:
            st.markdown("**Expected Disruption Timeline:**")
            d1, d2 = st.columns(2)
            d1.metric("Typical Duration", f"{duration_median} min")
            d2.metric("Min/Max Horizon", f"{duration_min} - {duration_max} min")
            st.caption(f"Calculated from {data_points} historical events.")

        st.markdown("---")
        st.markdown("**SHAP Feature Importance (Decision Drivers)**")
        explain_df = explain_prediction(inp["event_cause"], inp["crowd_size"], inp["zone"], inp["hour"], inp["day"], crowd_level)
        fig_explain = go.Figure(go.Bar(
            x=explain_df["Impact"], y=explain_df["Factor"], orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in explain_df["Impact"]],
        ))
        fig_explain.update_layout(xaxis_title="Impact on Risk Probability", height=280, margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_explain, use_container_width=True)

        st.markdown("---")
        st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;"><img src="https://img.icons8.com/color/48/database.png" width="22" height="22"/><b style="font-size:16px;">Past Similar Events in Database</b></div>', unsafe_allow_html=True)
        past_events = get_past_similar_events(inp["event_cause"], inp["zone"])
        if len(past_events) > 0:
            cols = st.columns(2)
            for idx, row in past_events.iterrows():
                with cols[idx % 2]:
                    st.markdown(make_past_event_card(row), unsafe_allow_html=True)
                    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        else:
            st.info("No matching historical records found for this subset.")

    with t2:
        st.subheader("Dynamic Diversion Routing Engine")
        st.caption("Models a commuter journey traversing the city through the affected zone.")
        rc1, rc2 = st.columns(2)
        with rc1:
            route_origin = st.selectbox("Commuter Origin Point", ZONES, index=ZONES.index(commuter_origin))
        with rc2:
            route_destination = st.selectbox("Commuter Destination Point", ZONES, index=ZONES.index(commuter_destination))
        if route_origin != commuter_origin or route_destination != commuter_destination:
            routing_res = tn.get_routing_scenarios(G, source=route_origin, target=route_destination, incident_node=inp["zone"], risk_score=risk_score)
        if routing_res:
            col_times, col_paths = st.columns([1, 1])
            with col_times:
                st.markdown("**Travel Statistics Comparison:**")
                st.metric("Standard Path (Stuck Time)", f"{routing_res['stuck_time_mins']} mins", delta=f"+{routing_res['stuck_time_mins'] - routing_res['std_time_mins']} mins due to jam", delta_color="inverse")
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

    with t3:
        st.subheader("Active Emergency Green Corridor Planner")
        st.caption("Secures emergency ambulance transport paths with automated signal scheduling.")
        if corridor_res:
            c_left, c_right = st.columns([1, 1.5])
            with c_left:
                st.markdown(f'<div style="background-color:#222437;border:1px solid #2f3149;border-left:3px solid #3B5BFF;padding:18px 20px;border-radius:10px;margin-bottom:15px;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><img src="https://img.icons8.com/color/48/ambulance.png" width="22" height="22"/><b style="color:#ffffff;font-family:Archivo,Inter,sans-serif;font-size:15px;">Medical Transit Corridor Active</b></div><span style="color:#a5a6b4;font-size:13px;">Route:</span> <b style="color:#ffffff;">{inp["zone"]}</b> → <b style="color:#ffffff;">{inp["target_hospital"]}</b><br><span style="color:#a5a6b4;font-size:13px;">Total Corridor Distance:</span> <b style="color:#ffffff;">{corridor_res["distance_km"]:.2f} km</b><br><span style="color:#a5a6b4;font-size:13px;">ETA (Emergency Speed):</span> <b style="color:#ffffff;">{corridor_res["eta_mins"]} mins</b></div>', unsafe_allow_html=True)
                st.markdown("**Signal Override Instructions:**")
                st.write("1. Broadcast preemptive signals to all roadside police controllers.")
                st.write("2. Force green state on signal controllers at intersections matching the schedule.")
                st.write("3. Clear central lane approaches 2 minutes before the ETA window.")
            with c_right:
                st.markdown("**Green Corridor Signal Preemption Schedule:**")
                schedule_df = pd.DataFrame(corridor_res["schedule"])
                schedule_df.columns = ["Intersection Node", "Distance (km)", "ETA", "Override Active Window"]
                st.dataframe(schedule_df, hide_index=True, use_container_width=True)
        else:
            st.warning("Could not compute emergency corridor routing path.")

    with t4:
        st.subheader("Multi-Station Police Resource Allocation Optimizer")
        st.caption("Allocates officers and patrol units from nearest depots with available capacity.")
        ro1, ro2 = st.columns(2)
        with ro1:
            req_off_override = st.number_input("Adjust Required Officers", min_value=1, max_value=100, value=officers_needed)
        with ro2:
            req_cars_override = st.number_input("Adjust Required Patrol Cars", min_value=0, max_value=20, value=cars_needed)
        if req_off_override != officers_needed or req_cars_override != cars_needed:
            dispatch_res, unmet_off, unmet_cars = tn.optimize_police_dispatch(required_officers=req_off_override, required_cars=req_cars_override, incident_zone=inp["zone"], G=G)
        if dispatch_res:
            st.markdown("**Optimized Dispatch Schedule:**")
            disp_data = [{"Police Station Depot": d["station"], "Officers Sent": d["officers_dispatched"], "Vehicles Sent": d["cars_dispatched"], "Travel Time ETA": f"{d['travel_time_mins']} mins", "Dispatch Status": d["status"]} for d in dispatch_res]
            st.dataframe(pd.DataFrame(disp_data), hide_index=True, use_container_width=True)
            if unmet_off > 0 or unmet_cars > 0:
                st.error(f"Warning: Insufficient resources. Unmet: {unmet_off} Officers, {unmet_cars} Patrol Cars. Initiate regional mutual aid callbacks.")
            else:
                st.success("Target resource requirements fully covered by adjacent police stations.")
        else:
            st.warning("Police depots are outside network routing limits.")

    with t5:
        st.subheader("BMTC Public Transit Advisory Portal")
        st.caption("Detects delays and computes detours for public bus commuters intersecting the incident zone.")
        if transit_res:
            for route in transit_res:
                st.markdown(f'<div style="background-color:#222437;border:1px solid #2f3149;border-left:3px solid #C2740C;padding:18px 20px;border-radius:10px;margin-bottom:12px;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><img src="https://img.icons8.com/color/48/bus.png" width="20" height="20"/><b style="color:#ffffff;font-family:Archivo,Inter,sans-serif;font-size:14px;">{route["name"]}</b><span style="background-color:#e55353;color:#ffffff;font-size:10.5px;font-weight:600;padding:2px 8px;border-radius:999px;margin-left:10px;">{route["status"]}</span></div><span style="color:#a5a6b4;font-size:13px;">Standard Path:</span> <span style="font-size:13px;color:#f8f9fa;">{route["standard_stops"]}</span><br><span style="color:#a5a6b4;font-size:13px;">Diverted Path:</span> <span style="font-size:13px;color:#f8f9fa;font-weight:500;">{route["diverted_stops"]}</span><br><span style="color:#a5a6b4;font-size:13px;">Estimated Delay:</span> <b style="color:#ffffff;">+{route["estimated_delay_mins"]} minutes</b><br><b style="color:#f9b115;">Commuter advisory —</b> <span style="color:#f8f9fa;">{route["shifted_stop_advise"]}</span></div>', unsafe_allow_html=True)
        else:
            st.success("No active BMTC transit routes are disrupted by this incident.")

    st.markdown("---")

    # Download brief
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;"><img src="https://img.icons8.com/color/48/download.png" width="24" height="24"/><b style="font-size:16px;">Download Command Documentation</b></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.download_button("Download Brief Report", data=command_brief, file_name="gridlock_command_brief.md", mime="text/markdown", use_container_width=True)

    # NEW: Mappls Map at the bottom
    st.markdown("---")
    st.markdown('<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;"><img src="https://img.icons8.com/color/48/map.png" width="24" height="24"/><b style="font-size:16px;">Live Affected Network & Diversion Routes</b></div>', unsafe_allow_html=True)
    st.caption(f"📍 {inp['formatted_address']} | Affected radius, real road diversion routes, nearby police & hospitals")

    if MAPPLS_KEY:
        map_html = build_mappls_map(inp["lat"], inp["lng"], risk, risk_score, inp["zone"], inp["target_hospital"], routing_res, corridor_res, dispatch_res)
        components.html(map_html, height=520)
    else:
        folium_map = build_folium_map(inp["lat"], inp["lng"], risk, risk_score, inp["zone"], inp["target_hospital"], routing_res, corridor_res, dispatch_res)
        st_folium(folium_map, height=520, width="stretch", returned_objects=[])