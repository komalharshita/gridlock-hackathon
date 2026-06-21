import math
import networkx as nx

# Coordinates for major nodes (Zones in dataset and key intersections)
NODE_COORDS = {
    # Competition Zones
    "Central Zone 1": (12.9716, 77.5946),  # MG Road
    "Central Zone 2": (12.9650, 77.5850),  # Majestic
    "North Zone 1":   (13.0500, 77.5900),  # Hebbal
    "North Zone 2":   (13.0800, 77.5700),  # Yelahanka
    "South Zone 1":   (12.9100, 77.5800),  # Jayanagar
    "South Zone 2":   (12.8800, 77.6000),  # JP Nagar
    "East Zone 1":    (12.9900, 77.6500),  # Indiranagar
    "East Zone 2":    (12.9700, 77.7000),  # Whitefield
    "West Zone 1":    (12.9600, 77.5200),  # Vijayanagar
    "West Zone 2":    (12.9400, 77.4900),  # Kengeri
    
    # Key Arterial Intersections
    "Town Hall":        (12.9640, 77.5800),
    "Mekhri Circle":    (13.0140, 77.5830),
    "Silk Board":       (12.9176, 77.6244),
    "Marathahalli":     (12.9592, 77.6974),
    "KR Puram":         (13.0040, 77.6980),
    "Domlur":           (12.9610, 77.6380),
    "Koramangala":      (12.9350, 77.6250),
    "Yeshwanthpur":     (13.0230, 77.5500),
    "Dairy Circle":     (12.9430, 77.5970),
    "Richmond Town":    (12.9600, 77.6010),
}

# Hospital Coordinates (Destination nodes for Green Corridors)
HOSPITAL_COORDS = {
    "NIMHANS Hospital": (12.9429, 77.5975),              # Central/South area
    "Manipal Hospital (Old Airport Rd)": (12.9592, 77.6441),  # East area
    "Fortis Hospital (Bannerghatta Rd)": (12.8943, 77.5976),  # South area
    "Aster CMI Hospital (Hebbal)": (13.0530, 77.5920),         # North area
}

# Police Station Coordinates & Available Capacities
POLICE_STATIONS = {
    "MG Road Police Station": {
        "coords": (12.9730, 77.6010),
        "officers": 15,
        "cars": 3
    },
    "Majestic Traffic Station": {
        "coords": (12.9690, 77.5810),
        "officers": 20,
        "cars": 4
    },
    "Hebbal Traffic Station": {
        "coords": (13.0350, 77.5970),
        "officers": 12,
        "cars": 2
    },
    "Jayanagar Police Station": {
        "coords": (12.9250, 77.5880),
        "officers": 15,
        "cars": 3
    },
    "Indiranagar Police Station": {
        "coords": (12.9780, 77.6400),
        "officers": 18,
        "cars": 4
    },
    "Vijayanagar Police Station": {
        "coords": (12.9620, 77.5300),
        "officers": 15,
        "cars": 3
    }
}

# Bangalore major connecting edges (Road segments)
ROAD_EDGES = [
    ("Central Zone 2", "Town Hall"),
    ("Town Hall", "Central Zone 1"),
    ("Central Zone 1", "Richmond Town"),
    ("Richmond Town", "Domlur"),
    ("Domlur", "East Zone 1"),
    ("East Zone 1", "KR Puram"),
    ("KR Puram", "East Zone 2"),
    ("East Zone 2", "Marathahalli"),
    ("Marathahalli", "Domlur"),
    ("Marathahalli", "Silk Board"),
    ("Silk Board", "Koramangala"),
    ("Koramangala", "Dairy Circle"),
    ("Dairy Circle", "South Zone 1"),
    ("South Zone 1", "South Zone 2"),
    ("Dairy Circle", "Town Hall"),
    ("Town Hall", "West Zone 1"),
    ("West Zone 1", "West Zone 2"),
    ("West Zone 1", "Yeshwanthpur"),
    ("Yeshwanthpur", "Mekhri Circle"),
    ("Mekhri Circle", "North Zone 1"),
    ("North Zone 1", "North Zone 2"),
    ("Mekhri Circle", "Central Zone 2"),
    ("Mekhri Circle", "Central Zone 1"),
    ("North Zone 1", "KR Puram"),
    ("North Zone 1", "Yeshwanthpur"),
]

# Simulated BMTC Bus Routes
BMTC_ROUTES = {
    "Route 335E (Majestic to Whitefield)": [
        "Central Zone 2", "Town Hall", "Richmond Town", "Domlur", "Marathahalli", "East Zone 2"
    ],
    "Route 500C (Hebbal to Silk Board)": [
        "North Zone 1", "KR Puram", "Marathahalli", "Silk Board"
    ],
    "Route 201G (Vijayanagar to Domlur)": [
        "West Zone 1", "Town Hall", "Dairy Circle", "Koramangala", "Domlur"
    ],
    "Route 365 (Majestic to JP Nagar)": [
        "Central Zone 2", "Town Hall", "Dairy Circle", "South Zone 1", "South Zone 2"
    ]
}

def haversine_distance(coord1, coord2):
    """Calculate distance in km between two coordinate pairs."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0 # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def create_bengaluru_graph():
    """Build Bengaluru street network graph using NetworkX."""
    G = nx.Graph()
    
    # Add nodes with positions and coordinates
    for name, coords in NODE_COORDS.items():
        G.add_node(name, coords=coords, lat=coords[0], lon=coords[1])
        
    # Add edges with physical distances as base weights
    for u, v in ROAD_EDGES:
        dist = haversine_distance(NODE_COORDS[u], NODE_COORDS[v])
        G.add_edge(u, v, distance=dist, base_weight=dist, weight=dist)
        
    return G

def get_routing_scenarios(G, source, target, incident_node, risk_score):
    """
    Calculate travel paths and times:
    1. Standard path (ignoring the incident congestion).
    2. Congested/Diversion path (taking congestion into account).
    """
    if source not in G.nodes or target not in G.nodes:
        # Fallback to standard coordinates
        return None
        
    # 1. Standard Shortest Path
    try:
        std_path = nx.shortest_path(G, source=source, target=target, weight='base_weight')
        std_distance = sum(G[std_path[i]][std_path[i+1]]['distance'] for i in range(len(std_path)-1))
    except (nx.NetworkXNoPath, KeyError):
        std_path = [source, target]
        std_distance = haversine_distance(NODE_COORDS[source], NODE_COORDS[target])
        
    # Average speed in normal conditions = 30 km/h (2 mins per km)
    std_time_mins = std_distance * 2.0
    
    # 2. Congested Path
    # Clone graph to avoid polluting the global state
    G_temp = G.copy()
    
    # Apply congestion penalty based on the risk score (0-100)
    # Severe risk multiplier = up to 6x delay on edges connected to incident node
    penalty_multiplier = 1.0 + (risk_score / 20.0) 
    
    # Apply penalty to edges connected to incident node
    affected_edges = []
    if incident_node in G_temp.nodes:
        for neighbor in G_temp.neighbors(incident_node):
            dist = G_temp[incident_node][neighbor]['distance']
            # Congested weight increases distance multiplier
            G_temp[incident_node][neighbor]['weight'] = dist * penalty_multiplier
            affected_edges.append((incident_node, neighbor))
            
    # Solve shortest path under congested weights
    try:
        congested_path = nx.shortest_path(G_temp, source=source, target=target, weight='weight')
        # Compute real travel time on this path
        congested_time_mins = 0.0
        congested_distance = 0.0
        for i in range(len(congested_path)-1):
            u, v = congested_path[i], congested_path[i+1]
            dist = G_temp[u][v]['distance']
            congested_distance += dist
            
            # Check if this edge is congested
            is_edge_congested = (u == incident_node or v == incident_node)
            speed = (30.0 / penalty_multiplier) if is_edge_congested else 30.0
            congested_time_mins += (dist / speed) * 60.0
            
    except (nx.NetworkXNoPath, KeyError):
        congested_path = std_path
        congested_distance = std_distance
        congested_time_mins = std_time_mins * penalty_multiplier
        
    # Delay savings
    delay_without_diversion = std_distance * (30.0 / penalty_multiplier) * 60.0 # if we went standard path but got stuck
    # Let's keep it simple: delay savings is the difference between standard path stuck in traffic vs taking diversion
    stuck_time_mins = 0.0
    for i in range(len(std_path)-1):
        u, v = std_path[i], std_path[i+1]
        dist = G[u][v]['distance']
        is_edge_congested = (u == incident_node or v == incident_node)
        speed = (30.0 / penalty_multiplier) if is_edge_congested else 30.0
        stuck_time_mins += (dist / speed) * 60.0
        
    savings_mins = max(0.0, stuck_time_mins - congested_time_mins)
    
    return {
        "std_path": std_path,
        "std_distance": std_distance,
        "std_time_mins": round(std_time_mins),
        "congested_path": congested_path,
        "congested_distance": congested_distance,
        "congested_time_mins": round(congested_time_mins),
        "stuck_time_mins": round(stuck_time_mins),
        "savings_mins": round(savings_mins),
        "affected_edges": affected_edges
    }

def get_emergency_corridor(G, incident_zone, hospital_name, risk_score):
    """
    Plan a green corridor from the incident node to a specific hospital.
    Automatically snaps the hospital coordinate to the nearest graph node,
    and returns routing coordinates and signal preemption timeline.
    """
    if incident_zone not in G.nodes:
        return None
        
    hosp_coords = HOSPITAL_COORDS.get(hospital_name)
    if not hosp_coords:
        return None
        
    # Snap hospital to the closest graph node
    closest_node = min(G.nodes, key=lambda n: haversine_distance(NODE_COORDS[n], hosp_coords))
    
    # Find shortest path (Green corridor gets signal override, so we route under normal weights)
    try:
        path = nx.shortest_path(G, source=incident_zone, target=closest_node, weight='base_weight')
        distance = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
    except (nx.NetworkXNoPath, KeyError):
        path = [incident_zone, closest_node]
        distance = haversine_distance(NODE_COORDS[incident_zone], hosp_coords)
        
    # Add final leg from closest node to actual hospital building
    final_leg = haversine_distance(NODE_COORDS[closest_node], hosp_coords)
    distance += final_leg
    
    # Emergency vehicle average speed under green corridor preemption = 50 km/h
    # (1.2 mins per km)
    eta_mins = distance * 1.2
    
    # Build signal preemption schedule
    schedule = []
    cumulative_dist = 0.0
    for i, node in enumerate(path):
        if i > 0:
            cumulative_dist += G[path[i-1]][path[i]]['distance']
            
        node_eta_mins = cumulative_dist * 1.2
        eta_sec = int((node_eta_mins * 60) % 60)
        eta_min = int(node_eta_mins)
        
        schedule.append({
            "node": node,
            "distance_km": round(cumulative_dist, 2),
            "eta_str": f"{eta_min:02d}m {eta_sec:02d}s",
            "preempt_window": f"{max(0, eta_min-1):02d}:{eta_sec:02d} - {eta_min+1:02d}:{eta_sec:02d}"
        })
        
    # Append final hospital node
    hospital_eta = distance * 1.2
    h_sec = int((hospital_eta * 60) % 60)
    h_min = int(hospital_eta)
    schedule.append({
        "node": hospital_name,
        "distance_km": round(distance, 2),
        "eta_str": f"{h_min:02d}m {h_sec:02d}s",
        "preempt_window": "Destination Arrived"
    })
    
    # Format coordinate path for map rendering
    coords_path = [NODE_COORDS[n] for n in path]
    coords_path.append(hosp_coords)
    
    return {
        "path": path,
        "hospital_coords": hosp_coords,
        "distance_km": round(distance, 2),
        "eta_mins": round(eta_mins, 1),
        "schedule": schedule,
        "coords_path": coords_path
    }

def optimize_police_dispatch(required_officers, required_cars, incident_zone, G):
    """
    Find nearby police stations, sort them by travel time/distance,
    and allocate resources up to the required levels.
    """
    if incident_zone not in G.nodes:
        return [], required_officers, required_cars
        
    dispatch_results = []
    remaining_officers = required_officers
    remaining_cars = required_cars
    
    # Calculate travel times from all stations
    station_distances = []
    for station_name, station_info in POLICE_STATIONS.items():
        station_coords = station_info["coords"]
        
        # Snap station to nearest node in road network
        closest_node = min(G.nodes, key=lambda n: haversine_distance(NODE_COORDS[n], station_coords))
        
        # Compute path distance
        try:
            path = nx.shortest_path(G, source=closest_node, target=incident_zone, weight='base_weight')
            dist = sum(G[path[i]][path[i+1]]['distance'] for i in range(len(path)-1))
        except (nx.NetworkXNoPath, KeyError):
            dist = haversine_distance(station_coords, NODE_COORDS[incident_zone])
            
        # Add buffer distance from station to nearest node
        dist += haversine_distance(station_coords, NODE_COORDS[closest_node])
        
        # Dispatch vehicle speed = 40 km/h
        travel_time_mins = (dist / 40.0) * 60.0
        
        station_distances.append({
            "name": station_name,
            "info": station_info,
            "dist": dist,
            "time_mins": round(travel_time_mins)
        })
        
    # Sort stations by travel time (closest first)
    station_distances.sort(key=lambda x: x["time_mins"])
    
    for station in station_distances:
        if remaining_officers <= 0 and remaining_cars <= 0:
            break
            
        avail_officers = station["info"]["officers"]
        avail_cars = station["info"]["cars"]
        
        dispatch_officers = min(remaining_officers, avail_officers)
        dispatch_cars = min(remaining_cars, avail_cars)
        
        if dispatch_officers > 0 or dispatch_cars > 0:
            remaining_officers -= dispatch_officers
            remaining_cars -= dispatch_cars
            
            dispatch_results.append({
                "station": station["name"],
                "coords": station["info"]["coords"],
                "officers_dispatched": dispatch_officers,
                "cars_dispatched": dispatch_cars,
                "travel_time_mins": station["time_mins"],
                "status": "Dispatched" if dispatch_officers > 0 or dispatch_cars > 0 else "Standby"
            })
            
    # If requirements are still not fully met, flag it
    unmet_officers = max(0, remaining_officers)
    unmet_cars = max(0, remaining_cars)
    
    return dispatch_results, unmet_officers, unmet_cars

def check_bmtc_transit(G, incident_node, risk_score):
    """
    Assess which BMTC bus routes are affected by the incident zone,
    calculate diversion delay, and provide shifted bus stop advice.
    """
    affected_routes = []
    
    # Multiplier for bus delays is higher since buses are bulkier
    # Severe risk multiplier = up to 8x delay
    bus_delay_multiplier = 1.0 + (risk_score / 15.0)
    
    for route_name, path in BMTC_ROUTES.items():
        # Check if the incident node is part of this route's stops
        is_affected = (incident_node in path)
        
        # Check if any edges on the route are directly connected to the incident node
        # (meaning the bus has to drive through the incident zone)
        route_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        is_edge_affected = False
        for u, v in route_edges:
            if u == incident_node or v == incident_node:
                is_edge_affected = True
                break
                
        if is_affected or is_edge_affected:
            # Route is disrupted! Let's find a diversion path for the bus
            # Find the path segments before and after the incident
            try:
                incident_idx = path.index(incident_node) if incident_node in path else -1
                
                # We want to route from the node before the incident to the node after
                if incident_idx > 0 and incident_idx < len(path) - 1:
                    u_before = path[incident_idx - 1]
                    v_after = path[incident_idx + 1]
                else:
                    # Incident is at terminal or edge-connected, find first/last route nodes
                    u_before = path[0]
                    v_after = path[-1]
                    
                # Calculate alternate route for this subsegment
                routing = get_routing_scenarios(G, source=u_before, target=v_after, incident_node=incident_node, risk_score=risk_score)
                
                if routing:
                    delay_mins = max(2, routing["congested_time_mins"] - routing["std_time_mins"])
                    diverted_path = path[:incident_idx] + routing["congested_path"][1:-1] + path[incident_idx+1:] if incident_idx != -1 else routing["congested_path"]
                else:
                    delay_mins = int(risk_score * 0.4)
                    diverted_path = path
            except Exception:
                delay_mins = int(risk_score * 0.3)
                diverted_path = path
                
            # Suggest shifted bus stops
            # Commuters should walk to adjacent nodes of the incident to catch the bus
            adjacent_nodes = []
            if incident_node in G.nodes:
                adjacent_nodes = list(G.neighbors(incident_node))[:2]
                
            shifted_stops = ", ".join(adjacent_nodes) if adjacent_nodes else "Richmond Town"
            
            affected_routes.append({
                "name": route_name,
                "standard_stops": " -> ".join(path),
                "diverted_stops": " -> ".join(diverted_path),
                "estimated_delay_mins": round(delay_mins),
                "shifted_stop_advise": f"Commuters at {incident_node} should walk to: {shifted_stops} stops",
                "status": "Delayed & Rerouted" if delay_mins > 5 else "Minor Delays"
            })
            
    return affected_routes
