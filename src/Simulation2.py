import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import branca.colormap as cm
import time
import math

# Graph-based simulation classes and functions
class Person:
    def __init__(self, id, sector, lat, lon):
        self.id = id
        self.sector = sector
        self.lat = lat
        self.lon = lon
        self.is_infected = False
        self.day_infected = None
        self.recovery_day = None
        self.connections = []  # List of connected person IDs
    
    def is_recovered(self, current_day):
        if self.recovery_day is None:
            return False
        return current_day >= self.recovery_day

class GraphSimulation:
    def __init__(self, sectors, r_value, intervention_day, intervention_strength):
        self.sectors = sectors
        self.r_value = r_value
        self.intervention_day = intervention_day
        self.intervention_strength = intervention_strength
        self.people = {}  # Dictionary of Person objects
        self.next_person_id = 0
        self.sector_connections = {}  # Tracks connections between sectors
        
    def create_person(self, sector, lat, lon):
        person = Person(self.next_person_id, sector, lat, lon)
        self.people[self.next_person_id] = person
        self.next_person_id += 1
        return person.id
    
    def add_connection(self, person1_id, person2_id):
        if person1_id in self.people and person2_id in self.people:
            self.people[person1_id].connections.append(person2_id)
            self.people[person2_id].connections.append(person1_id)
            
            # Track sector connections
            sector1 = self.people[person1_id].sector
            sector2 = self.people[person2_id].sector
            if sector1 != sector2:
                key = tuple(sorted([sector1, sector2]))
                self.sector_connections[key] = self.sector_connections.get(key, 0) + 1
    
    def initialize_population(self, initial_infections):
        # Create people in each sector
        for sector in self.sectors:
            sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector), None)
            if not sector_info:
                continue
                
            # Create people based on sector population (scaled down for performance)
            num_people = int(sector_info["population"] / 100)  # Scale down population
            
            # Create people
            for _ in range(num_people):
                # Add randomness to position
                lat_offset = (random.random() - 0.5) * 0.01
                lon_offset = (random.random() - 0.5) * 0.01
                self.create_person(
                    sector,
                    sector_info["lat"] + lat_offset,
                    sector_info["lon"] + lon_offset
                )
        
        # Create connections between people
        # First, create connections within sectors
        for sector in self.sectors:
            sector_people = [p_id for p_id, p in self.people.items() if p.sector == sector]
            
            # Each person connects with ~5-10 other people in their sector
            for person_id in sector_people:
                num_connections = random.randint(5, 10)
                possible_connections = [p for p in sector_people if p != person_id]
                if possible_connections:
                    connections = random.sample(possible_connections, min(num_connections, len(possible_connections)))
                    for conn_id in connections:
                        self.add_connection(person_id, conn_id)
        
        # Then, create some connections between sectors
        # People have a 10% chance to connect with someone from another sector
        for person_id, person in self.people.items():
            if random.random() < 0.1:  # 10% chance
                other_sectors = [s for s in self.sectors if s != person.sector]
                if other_sectors:
                    target_sector = random.choice(other_sectors)
                    target_people = [p_id for p_id, p in self.people.items() if p.sector == target_sector]
                    if target_people:
                        target_id = random.choice(target_people)
                        self.add_connection(person_id, target_id)
        
        # Initialize infections
        initial_sectors = random.sample(self.sectors, min(initial_infections, len(self.sectors)))
        for sector in initial_sectors:
            sector_people = [p_id for p_id, p in self.people.items() if p.sector == sector]
            if sector_people:
                # Infect 5-10 people in each initial sector
                num_infections = random.randint(5, 10)
                infected_people = random.sample(sector_people, min(num_infections, len(sector_people)))
                for person_id in infected_people:
                    person = self.people[person_id]
                    person.is_infected = True
                    person.day_infected = 0
                    person.recovery_day = random.randint(7, 14)
    
    def run_simulation(self, days):
        # Initialize data structure for tracking infections
        sectors_data = {sector: {"timeline": []} for sector in self.sectors}
        
        # Run simulation for each day
        for day in range(days + 1):
            # Apply intervention effect if past intervention day
            current_r = self.r_value
            if day >= self.intervention_day:
                current_r = self.r_value * (1 - self.intervention_strength)
            
            # Track infections per sector
            active_infections = {sector: 0 for sector in self.sectors}
            cumulative_infections = {sector: 0 for sector in self.sectors}
            
            # Process recoveries and count active infections
            for person in self.people.values():
                if person.is_infected:
                    if person.recovery_day <= day:
                        person.is_infected = False
                        person.day_infected = None  # Reset day_infected when recovered
                        person.recovery_day = None  # Reset recovery_day when recovered
                    else:
                        active_infections[person.sector] += 1
                        cumulative_infections[person.sector] += 1
            
            # Process new infections
            for person in self.people.values():
                if not person.is_infected:
                    # Calculate probability of infection based on infected connections
                    infected_connections = sum(1 for conn_id in person.connections 
                                            if self.people[conn_id].is_infected)
                    
                    if infected_connections > 0:
                        # Base infection probability from R value
                        base_prob = current_r / (7 * len(person.connections))
                        # Adjust probability based on number of infected connections
                        infection_prob = 1 - (1 - base_prob) ** infected_connections
                        
                        if random.random() < infection_prob:
                            person.is_infected = True
                            person.day_infected = day  # Set the day when person gets infected
                            person.recovery_day = day + random.randint(7, 14)
            
            # Update timeline data for each sector
            for sector in self.sectors:
                prev_day_data = next((d for d in sectors_data[sector]["timeline"] 
                                    if d["day"] == day - 1), None)
                prev_day_total = prev_day_data["cumulative_infections"] if prev_day_data else 0
                daily_new = cumulative_infections[sector] - prev_day_total
                
                sectors_data[sector]["timeline"].append({
                    "day": day,
                    "active_infections": active_infections[sector],
                    "cumulative_infections": cumulative_infections[sector],
                    "daily_new_cases": daily_new if daily_new >= 0 else 0
                })
        
        return sectors_data, list(self.people.values())

# Set page config
st.set_page_config(layout="wide", page_title="Islamabad Virus Spread Simulator")

# App title
st.title("Interactive Virus Spread Simulator - Islamabad")

# Initialize session state variables if they don't exist
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

if 'antidote_applied' not in st.session_state:
    st.session_state.antidote_applied = False

if 'antidote_day' not in st.session_state:
    st.session_state.antidote_day = 0

if 'antidote_effectiveness' not in st.session_state:
    st.session_state.antidote_effectiveness = 0.5

if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "Dots Only"

if 'dot_density' not in st.session_state:
    st.session_state.dot_density = 0.5

if 'selected_day' not in st.session_state:
    st.session_state.selected_day = 0

if 'dot_seed' not in st.session_state:
    st.session_state.dot_seed = random.randint(1, 10000)  # Create a fixed seed for random dot generation

# Add animation state variables
if 'is_animating' not in st.session_state:
    st.session_state.is_animating = False

if 'animation_speed' not in st.session_state:
    st.session_state.animation_speed = 2.0  # seconds between frames (slower default)

if 'animation_last_update' not in st.session_state:
    st.session_state.animation_last_update = time.time()

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Add simulation type selector
simulation_type = st.sidebar.radio(
    "Simulation Type",
    ["Original Simulation", "Graph-based Simulation"],
    help="Original: Uses spatial proximity for infection spread. Graph-based: Uses social connections between people."
)

# Parameter inputs
initial_infections = st.sidebar.slider("Initial infection centers", 1, 5, 2)
r_value = st.sidebar.slider("R value (infection rate)", 0.5, 5.0, 2.5)
simulation_days = st.sidebar.slider("Simulation duration (days)", 10, 90, 45)
intervention_day = st.sidebar.slider("Day to start interventions", 0, 45, 15)
intervention_strength = st.sidebar.slider("Intervention strength", 0.0, 1.0, 0.5)

# Add a parameter to control dot density
st.session_state.dot_density = st.sidebar.slider("Dot density", 0.1, 1.0, st.session_state.dot_density, 
                               help="Controls how many dots are shown (higher values show more dots but may affect performance)")

# IMPORTANT: Add a display mode selector - DOTS ONLY is the default
st.session_state.display_mode = st.sidebar.radio(
    "Display Mode",
    ["Dots Only", "Sectors Only", "Dots + Sectors (Light)", "Dots + Sectors (Labels Only)"],
    index=["Dots Only", "Sectors Only", "Dots + Sectors (Light)", "Dots + Sectors (Labels Only)"].index(st.session_state.display_mode)
)

# Islamabad data
islamabad = {
    "lat": 33.6844,
    "lon": 73.0479,
    "population": 1095000,
    # Define districts/sectors within Islamabad for better visualization
    "sectors": [
        {"name": "F-6", "lat": 33.7294, "lon": 73.0815, "population": 43800},
        {"name": "F-7", "lat": 33.7196, "lon": 73.0570, "population": 58400},
        {"name": "F-8", "lat": 33.7030, "lon": 73.0540, "population": 65000},
        {"name": "F-10", "lat": 33.6950, "lon": 73.0210, "population": 72000},
        {"name": "F-11", "lat": 33.6845, "lon": 73.0018, "population": 66000},
        {"name": "G-6", "lat": 33.7173, "lon": 73.0876, "population": 53000},
        {"name": "G-7", "lat": 33.7049, "lon": 73.0900, "population": 69000},
        {"name": "G-8", "lat": 33.6932, "lon": 73.0710, "population": 76000},
        {"name": "G-9", "lat": 33.6870, "lon": 73.0490, "population": 85000},
        {"name": "G-10", "lat": 33.6768, "lon": 73.0323, "population": 79000},
        {"name": "G-11", "lat": 33.6676, "lon": 73.0131, "population": 75000},
        {"name": "I-8", "lat": 33.6595, "lon": 73.0700, "population": 59000},
        {"name": "I-9", "lat": 33.6535, "lon": 73.0484, "population": 55000},
        {"name": "I-10", "lat": 33.6478, "lon": 73.0300, "population": 68000},
        {"name": "E-7", "lat": 33.7310, "lon": 73.0400, "population": 36000},
        {"name": "E-11", "lat": 33.7100, "lon": 72.9850, "population": 45000},
    ]
}

# Class to efficiently manage infected individuals
class InfectedPerson:
    def __init__(self, lat, lon, day_infected):
        self.lat = lat
        self.lon = lon
        self.day_infected = day_infected
        self.recovery_day = day_infected + random.randint(7, 14)  # Recovery after 7-14 days
       
    def is_recovered(self, current_day):
        return current_day >= self.recovery_day

# Optimized simulation function focused on Islamabad
def run_virus_simulation(r_value, simulation_days, intervention_day, intervention_strength, antidote_applied=False, antidote_day=None, antidote_effectiveness=0.0):
    # Initialize data structures
    sectors_data = {}
    infected_people = []
    
    # Setup initial data for each sector
    for sector in islamabad["sectors"]:
        sectors_data[sector["name"]] = {
            "lat": sector["lat"],
            "lon": sector["lon"],
            "population": sector["population"],
            "timeline": []
        }
       
    # Choose initial infection locations
    initial_sectors = random.sample(islamabad["sectors"], min(initial_infections, len(islamabad["sectors"])))
   
    # Create initial infections - more dots at the beginning for visibility
    for sector in initial_sectors:
        sector_name = sector["name"]
        initial_count = random.randint(20, 50)  # Increased initial count for better visibility
       
        # Create infected people objects
        for _ in range(initial_count):
            # Add some randomness to positions
            lat_offset = (random.random() - 0.5) * 0.01
            lon_offset = (random.random() - 0.5) * 0.01
           
            infected_people.append(InfectedPerson(
                sector["lat"] + lat_offset,
                sector["lon"] + lon_offset,
                0  # Day 0
            ))
   
    # Pre-calculate distances between sectors for mobility modeling
    sector_distances = {}
    for s1 in islamabad["sectors"]:
        sector_distances[s1["name"]] = {}
        for s2 in islamabad["sectors"]:
            dist = ((s1["lat"] - s2["lat"])**2 + (s1["lon"] - s2["lon"])**2)**0.5
            sector_distances[s1["name"]][s2["name"]] = dist
   
    # Run simulation for each day
    for day in range(simulation_days + 1):
        # Apply intervention effect if past intervention day
        current_r = r_value
        if day >= intervention_day:
            current_r = r_value * (1 - intervention_strength)
       
        # Count active infections and recoveries per sector
        active_infections = {sector["name"]: 0 for sector in islamabad["sectors"]}
        cumulative_infections = {sector["name"]: 0 for sector in islamabad["sectors"]}
       
        # Process recoveries and count active infections
        to_remove = []
        for i, person in enumerate(infected_people):
            # Check if recovered
            if person.is_recovered(day) or (antidote_applied and day >= antidote_day and random.random() < antidote_effectiveness):
                to_remove.append(i)
                continue
               
            # Find which sector this person is in
            closest_sector = min(islamabad["sectors"],
                                key=lambda s: ((s["lat"] - person.lat)**2 + (s["lon"] - person.lon)**2)**0.5)
            active_infections[closest_sector["name"]] += 1
            cumulative_infections[closest_sector["name"]] += 1
       
        # Remove recovered people (in reverse order to avoid index issues)
        for i in sorted(to_remove, reverse=True):
            if i < len(infected_people):
                infected_people.pop(i)
       
        # Process new infections
        new_infections = []
       
        # Each infected person can infect others
        for sector in islamabad["sectors"]:
            sector_name = sector["name"]
            sector_infected = active_infections[sector_name]
           
            if sector_infected == 0:
                continue
               
            # Calculate expected new infections based on R value and active infections
            expected_new_infections = sector_infected * current_r / 7  # R value is per week
           
            # Add randomness
            actual_new_infections = np.random.poisson(expected_new_infections)
           
            # Cap maximum new infections for performance
            MAX_NEW_INFECTIONS_PER_SECTOR = 100  # Reduced for better performance
            if actual_new_infections > MAX_NEW_INFECTIONS_PER_SECTOR:
                actual_new_infections = MAX_NEW_INFECTIONS_PER_SECTOR
           
            # Create new infected people
            for _ in range(actual_new_infections):
                # Determine where new infection occurs - either in same sector or nearby
                if random.random() < 0.7:  # 70% chance to infect in same sector
                    target_sector = sector
                else:
                    # Choose another sector with probability based on distance
                    distances = sector_distances[sector_name]
                    target_sector_name = random.choices(
                        list(distances.keys()),
                        weights=[1 / (1 + dist * 10) for dist in distances.values()],
                        k=1
                    )[0]
                    target_sector = next(s for s in islamabad["sectors"] if s["name"] == target_sector_name)
               
                # Add randomness to position
                lat_offset = (random.random() - 0.5) * 0.01
                lon_offset = (random.random() - 0.5) * 0.01
               
                new_infections.append(InfectedPerson(
                    target_sector["lat"] + lat_offset,
                    target_sector["lon"] + lon_offset,
                    day
                ))
       
        # Add new infections to the list but cap for performance
        MAX_TOTAL_INFECTIONS = 5000  # Reduced for better performance
        if len(infected_people) + len(new_infections) <= MAX_TOTAL_INFECTIONS:
            infected_people.extend(new_infections)
        else:
            # If we would exceed the max, add as many as we can
            space_left = max(0, MAX_TOTAL_INFECTIONS - len(infected_people))
            if space_left > 0:
                infected_people.extend(new_infections[:space_left])
       
        # Update timeline data for each sector
        for sector in islamabad["sectors"]:
            sector_name = sector["name"]
           
            # Calculate daily new cases
            prev_day_data = next((d for d in sectors_data[sector_name]["timeline"] if d["day"] == day - 1), None)
            prev_day_total = prev_day_data["cumulative_infections"] if prev_day_data else 0
            daily_new = cumulative_infections[sector_name] - prev_day_total
           
            # Store data
            sectors_data[sector_name]["timeline"].append({
                "day": day,
                "active_infections": active_infections[sector_name],
                "cumulative_infections": cumulative_infections[sector_name],
                "daily_new_cases": daily_new if daily_new >= 0 else 0
            })
   
    # Store both daily infected person data and totals
    return sectors_data, infected_people

# First fix the rerun issue by replacing experimental_rerun
if 'map_key' not in st.session_state:
    st.session_state.map_key = 0

def create_islamabad_map(sectors_data, infected_people, selected_day, dot_density=0.5, display_mode="Dots Only"):
    # Create map centered on Islamabad with zoom restrictions
    m = folium.Map(
        location=[33.6844, 73.0479],
        zoom_start=12,
        tiles="CartoDB positron",
        min_zoom=11,
        max_zoom=15
    )
   
    # Filter infected people for current day - FIXED THIS FUNCTION
    current_infected = [p for p in infected_people if p.day_infected is not None and p.day_infected <= selected_day and not p.is_recovered(selected_day)]
    
    # Find recovered people for current day
    current_recovered = [p for p in infected_people if p.day_infected is not None and p.day_infected <= selected_day and p.is_recovered(selected_day)]
    
    # Use fixed random seed for consistent dot generation
    # Combine the seed with the day to ensure different days show different patterns
    # But the same day always shows the same pattern
    dot_random = random.Random(st.session_state.dot_seed + selected_day)
    
    # Handle Dots Only mode - COMPLETELY REWRITTEN
    if display_mode == "Dots Only":
        # Get total infected and recovered counts across all sectors
        total_active = sum(
            next((d for d in data["timeline"] if d["day"] == selected_day), {"active_infections": 0})["active_infections"] 
            for data in sectors_data.values()
        )
        
        total_recovered = sum(
            (next((d for d in data["timeline"] if d["day"] == selected_day), {"cumulative_infections": 0})["cumulative_infections"] -
             next((d for d in data["timeline"] if d["day"] == selected_day), {"active_infections": 0})["active_infections"])
            for data in sectors_data.values()
        )
        
        # Only proceed if we have cases to show
        if total_active > 0 or total_recovered > 0:
            # Create feature groups for the dots
            active_dot_group = folium.FeatureGroup(name="Active Infection Dots")
            recovered_dot_group = folium.FeatureGroup(name="Recovered Dots")
            
            # For each sector, create dots
            for sector_name, data in sectors_data.items():
                sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector_name), None)
                if not sector_info:
                    continue
                   
                # Get data for current day
                day_data = next((d for d in data["timeline"] if d["day"] == selected_day), None)
                if not day_data:
                    continue
                
                # Find active cases and recovered cases in this sector
                active_cases = day_data["active_infections"]
                total_cases = day_data["cumulative_infections"]
                recovered_cases = total_cases - active_cases
                
                # Skip if no cases
                if active_cases == 0 and recovered_cases == 0:
                    continue
                
                # Calculate radius for dot distribution (bigger than sector radius for better spread)
                sector_radius = 500
                
                # Calculate number of dots to show
                MAX_DOTS_PER_SECTOR = 200 * dot_density
                active_dots_to_show = min(active_cases, int(MAX_DOTS_PER_SECTOR))
                recovered_dots_to_show = min(recovered_cases, int(MAX_DOTS_PER_SECTOR))
                
                # Scale down if too many dots
                if active_cases + recovered_cases > MAX_DOTS_PER_SECTOR * 2:
                    scale_factor = (MAX_DOTS_PER_SECTOR * 2) / (active_cases + recovered_cases)
                    active_dots_to_show = int(active_cases * scale_factor)
                    recovered_dots_to_show = int(recovered_cases * scale_factor)
                
                # Generate positions for infected dots within the sector circle - using fixed seed
                for i in range(active_dots_to_show):
                    # Generate random point within sector circle - using our seeded random generator
                    angle = dot_random.uniform(0, 2 * 3.14159)
                    radius_factor = dot_random.uniform(0, 0.9)  # Stay within 90% of circle radius
                    offset_lat = radius_factor * sector_radius * 0.000009 * math.cos(angle)  # Convert meters to approx degrees
                    offset_lon = radius_factor * sector_radius * 0.000015 * math.sin(angle)  # Adjust for latitude
                    
                    folium.CircleMarker(
                        location=[sector_info["lat"] + offset_lat, sector_info["lon"] + offset_lon],
                        radius=2,  # Slightly larger for better visibility
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.8,
                        weight=1,
                        popup="Active Infection"
                    ).add_to(active_dot_group)
                
                # Generate positions for recovered dots - using fixed seed
                for i in range(recovered_dots_to_show):
                    # Generate random point within sector circle - using our seeded random generator
                    angle = dot_random.uniform(0, 2 * 3.14159)
                    radius_factor = dot_random.uniform(0, 0.9)
                    offset_lat = radius_factor * sector_radius * 0.000009 * math.cos(angle)
                    offset_lon = radius_factor * sector_radius * 0.000015 * math.sin(angle)
                    
                    folium.CircleMarker(
                        location=[sector_info["lat"] + offset_lat, sector_info["lon"] + offset_lon],
                        radius=2,
                        color='green',
                        fill=True,
                        fillColor='green',
                        fillOpacity=0.8,
                        weight=1,
                        popup="Recovered Case"
                    ).add_to(recovered_dot_group)
            
            # Add the dot groups to the map
            active_dot_group.add_to(m)
            recovered_dot_group.add_to(m)
    
    # Handle combined modes with dots - also use the seeded random
    elif "Dots" in display_mode and "Labels" not in display_mode and len(current_infected) > 0:
        # Calculate max dots based on density
        MAX_DOTS = int(2000 * dot_density)
        
        # If we have a reasonable number of dots, show them all
        if len(current_infected) <= MAX_DOTS:
            dots_to_show = current_infected
        else:
            # Efficient sampling for larger numbers - use seeded random
            dots_to_show = dot_random.sample(current_infected, MAX_DOTS)
        
        # Create a feature group for the dots
        dot_group = folium.FeatureGroup(name="Infection Dots")
        
        # Add dots to the feature group
        for dot in dots_to_show:
            folium.CircleMarker(
                location=[dot.lat, dot.lon],
                radius=2,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.8,
                weight=1,
                popup=f"Infected on day {dot.day_infected}"
            ).add_to(dot_group)
        
        # Add the feature group to the map
        dot_group.add_to(m)

    # Add sector circles if display mode includes sectors
    if "Sectors" in display_mode:
        # Calculate sector radius in meters (adjusted for map aesthetics)
        sector_radius = 300
        
        for sector_name, data in sectors_data.items():
            sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector_name), None)
            if not sector_info:
                continue
               
            # Get data for current day
            day_data = next((d for d in data["timeline"] if d["day"] == selected_day), None)
            if not day_data:
                continue
            
            # Find active cases and recovered cases in this sector
            active_cases = day_data["active_infections"]
            total_cases = day_data["cumulative_infections"]
            recovered_cases = total_cases - active_cases
           
            # Calculate infection percentage
            infection_percent = (active_cases / sector_info["population"]) * 100
           
            # Create popup with sector information
            popup_text = f"""
            <strong>{sector_name}</strong><br>
            Population: {sector_info['population']:,}<br>
            Active Cases: {active_cases:,} ({infection_percent:.2f}%)<br>
            Recovered Cases: {recovered_cases:,}<br>
            Total Infections: {total_cases:,}<br>
            New Cases Today: {day_data['daily_new_cases']:,}
            """
            
            # In "Light" mode, add a light circle to indicate sector boundary
            if display_mode != "Dots + Sectors (Labels Only)":
                # Add a light boundary for the sector
                folium.Circle(
                    location=[sector_info["lat"], sector_info["lon"]],
                    radius=sector_radius,
                    color='gray',
                    fill=True,
                    fill_opacity=0.05,
                    weight=1,
                    popup=folium.Popup(popup_text, max_width=250)
                ).add_to(m)
            
            # Add sector label - modified for Labels Only mode to show infection counts
            if display_mode == "Dots + Sectors (Labels Only)":
                # For Labels Only, show the sector name and case count
                html_content = f'''
                <div style="font-size: 10pt; color: black; text-align: center;">
                    <strong>{sector_name}</strong><br>
                    {active_cases:,}
                </div>
                '''
            else:
                # For other modes, just show the sector name
                html_content = f'''
                <div style="font-size: 10pt; color: black; text-align: center;">
                    <strong>{sector_name}</strong>
                </div>
                '''
            
            folium.Marker(
                location=[sector_info["lat"], sector_info["lon"]],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 0),
                    html=html_content
                )
            ).add_to(m)
            
            # NEW: Add representative dots within each sector circle
            # Only for Sectors Only or Dots + Sectors (Light) modes, NOT for Labels Only
            if display_mode != "Dots + Sectors (Labels Only)" and display_mode != "Dots Only":
                # Calculate number of dots to show based on density
                MAX_DOTS_PER_SECTOR = 100
                active_dots_to_show = min(active_cases, MAX_DOTS_PER_SECTOR)
                recovered_dots_to_show = min(recovered_cases, MAX_DOTS_PER_SECTOR)
                
                # Scale number of dots if there are too many cases
                if active_cases + recovered_cases > MAX_DOTS_PER_SECTOR * 2:
                    scale_factor = (MAX_DOTS_PER_SECTOR * 2) / (active_cases + recovered_cases)
                    active_dots_to_show = int(active_cases * scale_factor)
                    recovered_dots_to_show = int(recovered_cases * scale_factor)
                
                # Create a feature group for this sector's dots
                sector_dots = folium.FeatureGroup(name=f"Sector {sector_name} Dots")
                
                # Generate positions for infected dots within the sector circle
                for _ in range(active_dots_to_show):
                    # Generate random point within sector circle - use seeded random
                    angle = dot_random.uniform(0, 2 * 3.14159)
                    radius_factor = dot_random.uniform(0, 0.9)  # Stay within 90% of circle radius
                    offset_lat = radius_factor * sector_radius * 0.000009 * math.cos(angle)  # Convert meters to approx degrees
                    offset_lon = radius_factor * sector_radius * 0.000015 * math.sin(angle)  # Adjust for latitude
                    
                    folium.CircleMarker(
                        location=[sector_info["lat"] + offset_lat, sector_info["lon"] + offset_lon],
                        radius=2,
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.8,
                        weight=1,
                        popup="Active Infection"
                    ).add_to(sector_dots)
                
                # Generate positions for recovered dots
                for _ in range(recovered_dots_to_show):
                    # Generate random point within sector circle - use seeded random
                    angle = dot_random.uniform(0, 2 * 3.14159)
                    radius_factor = dot_random.uniform(0, 0.9)
                    offset_lat = radius_factor * sector_radius * 0.000009 * math.cos(angle)
                    offset_lon = radius_factor * sector_radius * 0.000015 * math.sin(angle)
                    
                    folium.CircleMarker(
                        location=[sector_info["lat"] + offset_lat, sector_info["lon"] + offset_lon],
                        radius=2,
                        color='green',
                        fill=True,
                        fillColor='green',
                        fillOpacity=0.8,
                        weight=1,
                        popup="Recovered Case"
                    ).add_to(sector_dots)
                
                # Add the sector dots to the map
                sector_dots.add_to(m)
   
    # Add title
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background-color: rgba(0, 0, 0, 0.8);
                padding: 10px; border-radius: 5px; border: 2px solid #666;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
        <h4 style="color: white; margin: 0; text-align: center;">Islamabad Virus Simulation - Day {}</h4>
    </div>
    '''.format(selected_day)
    m.get_root().html.add_child(folium.Element(title_html))
   
    return m

# Create a callback function to handle day changes
def update_selected_day():
    # Update the map key to force a refresh when day changes
    st.session_state.map_key += 1

# Animation toggle function
def toggle_animation():
    st.session_state.is_animating = not st.session_state.is_animating
    if st.session_state.is_animating:
        # If starting animation, make sure we're not at the end
        if st.session_state.selected_day >= simulation_days:
            st.session_state.selected_day = 0
        st.session_state.map_key += 1
    # Always rerun when toggling animation state
    st.rerun()

# Function to handle animation progression
def handle_animation():
    if st.session_state.is_animating:
        # Increment the day
        st.session_state.selected_day += 1
        # Check if we reached the end
        if st.session_state.selected_day >= simulation_days:
            # Stop animation at the end
            st.session_state.is_animating = False
        else:
            # Force map refresh
            st.session_state.map_key += 1
            # Schedule next update by rerunning
            st.rerun()

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Reset antidote status
        st.session_state.antidote_applied = False
        st.session_state.antidote_day = 0
        
        if simulation_type == "Original Simulation":
            # Run original simulation
            sectors_data, infected_people = run_virus_simulation(
                r_value,
                simulation_days,
                intervention_day,
                intervention_strength
            )
        else:
            # Run graph-based simulation
            graph_sim = GraphSimulation(
                [s["name"] for s in islamabad["sectors"]],
                r_value,
                intervention_day,
                intervention_strength
            )
            graph_sim.initialize_population(initial_infections)
            sectors_data, infected_people = graph_sim.run_simulation(simulation_days)
        
        # Cache simulation data in session state
        st.session_state.sectors_data = sectors_data
        st.session_state.infected_people = infected_people
        st.session_state.simulation_run = True
        st.session_state.selected_day = 0
        st.session_state.simulation_type = simulation_type
        
        # Force a complete page refresh to ensure the map renders correctly
        st.rerun()

# Update the map display section
if st.session_state.simulation_run:
    # Create two columns for map and statistics
    col1, col2 = st.columns([3, 1])
   
    with col1:
        # Day control row with slider and animation buttons
        day_col1, day_col2 = st.columns([3, 1])
        
        with day_col1:
            # Use Streamlit's on_change parameter to handle day changes properly
            st.slider(
                "Simulation Day",
                0,
                simulation_days,
                value=st.session_state.selected_day,
                key="day_slider_value",
                on_change=update_selected_day,
                disabled=st.session_state.is_animating  # Disable slider during animation
            )
            
            # Make sure selected_day is updated from the slider value
            if not st.session_state.is_animating:
                st.session_state.selected_day = st.session_state.day_slider_value
        
        with day_col2:
            # Animation controls
            animation_button_text = "⏹️ Stop" if st.session_state.is_animating else "▶️ Play"
            animation_button_color = "secondary" if st.session_state.is_animating else "primary"
            
            # Prominent animation button
            button_pressed = st.button(
                animation_button_text,
                key="animation_toggle",
                type=animation_button_color,
                use_container_width=True
            )
            
            # Handle button press
            if button_pressed:
                toggle_animation()
            
            # Only show speed control when not animating
            if not st.session_state.is_animating:
                # Animation speed control - slower options
                st.select_slider(
                    "Speed",
                    options=[1.0, 1.5, 2.0, 3.0, 4.0],
                    value=st.session_state.animation_speed,
                    key="animation_speed_slider",
                    format_func=lambda x: f"{x}s"
                )
                
                # Update animation speed from slider
                st.session_state.animation_speed = st.session_state.animation_speed_slider
            else:
                # Show day counter during animation
                st.markdown(f"**Day {st.session_state.selected_day}**")
        
        # Create and display the map
        current_map = create_islamabad_map(
            st.session_state.sectors_data,
            st.session_state.infected_people,
            st.session_state.selected_day,
            st.session_state.dot_density,
            st.session_state.display_mode
        )
        
        # Display the map with a stable key
        st_folium(
            current_map,
            width=800,
            height=600,
            key=f"map_{st.session_state.map_key}"
        )
    
    # IMPORTANT: Move the stats column here, OUTSIDE the col1 context
    # This ensures the statistics are updated with each rerun    
    with col2:
        # Display statistics - use current selected_day
        current_day = st.session_state.selected_day
        st.subheader(f"Day {current_day} Statistics")
        
        # Show simulation type
        st.info(f"Using {st.session_state.simulation_type}")
       
        # Calculate totals
        total_active = 0
        total_cumulative = 0
        total_new_cases = 0
       
        for sector_name, data in st.session_state.sectors_data.items():
            day_data = next((d for d in data["timeline"] if d["day"] == current_day), None)
            if day_data:
                total_active += day_data["active_infections"]
                total_cumulative += day_data["cumulative_infections"]
                total_new_cases += day_data["daily_new_cases"]
       
        # Display summary statistics
        st.metric("Active Cases", f"{total_active:,}")
        st.metric("New Cases Today", f"{total_new_cases:,}")
        st.metric("Total Infections", f"{total_cumulative:,}")
       
        # Show intervention status
        if current_day >= intervention_day:
            st.info(f"Interventions active (reducing R by {intervention_strength*100:.0f}%)")
        else:
            st.warning(f"No interventions yet (starting day {intervention_day})")
       
        # Antidote controls
        st.subheader("Antidote Control Panel")
       
        if not st.session_state.antidote_applied:
            antidote_effectiveness = st.slider("Antidote Effectiveness", 0.1, 1.0, 0.5)
           
            if st.button("Deploy Antidote"):
                with st.spinner("Applying antidote..."):
                    st.session_state.antidote_applied = True
                    st.session_state.antidote_day = st.session_state.selected_day
                    st.session_state.antidote_effectiveness = antidote_effectiveness
                   
                    # Re-run simulation with antidote
                    if st.session_state.simulation_type == "Original Simulation":
                        sectors_data, infected_people = run_virus_simulation(
                            r_value,
                            simulation_days,
                            intervention_day,
                            intervention_strength,
                            True,
                            st.session_state.selected_day,
                            antidote_effectiveness
                        )
                    else:
                        graph_sim = GraphSimulation(
                            [s["name"] for s in islamabad["sectors"]],
                            r_value,
                            intervention_day,
                            intervention_strength
                        )
                        graph_sim.initialize_population(initial_infections)
                        sectors_data, infected_people = graph_sim.run_simulation(simulation_days)
                    
                    st.session_state.sectors_data = sectors_data
                    st.session_state.infected_people = infected_people
                    st.session_state.map_key += 1  # Force map refresh
                    st.success(f"Antidote deployed on day {st.session_state.selected_day} with {antidote_effectiveness*100:.0f}% effectiveness!")
                    st.rerun()
        else:
            st.success(f"Antidote deployed on day {st.session_state.antidote_day} with {st.session_state.antidote_effectiveness*100:.0f}% effectiveness!")
       
        # Show top 3 most infected sectors
        st.subheader("Most affected sectors:")
        sector_infections = []
       
        for sector_name, data in st.session_state.sectors_data.items():
            sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector_name), None)
            day_data = next((d for d in data["timeline"] if d["day"] == current_day), None)
           
            if sector_info and day_data:
                infection_percent = (day_data["active_infections"] / sector_info["population"]) * 100
                sector_infections.append((sector_name, day_data["active_infections"], infection_percent))
       
        # Sort by number of infections
        sector_infections.sort(key=lambda x: x[1], reverse=True)
       
        # Display top 3
        for i, (sector, infected, percentage) in enumerate(sector_infections[:3]):
            st.write(f"{i+1}. {sector}: {infected:,} cases ({percentage:.2f}%)")
    
    # Move the animation handling AFTER both map and stats are displayed    
    # Introduce artificial delay to control animation speed
    if st.session_state.is_animating:
        time.sleep(st.session_state.animation_speed)
        # Proceed to next day after delay
        handle_animation()
else:
    # Instructions if simulation hasn't been run
    st.write("### Instructions")
    st.write("1. Choose your simulation type (Original or Graph-based)")
    st.write("2. Adjust the simulation parameters in the sidebar")
    st.write("3. Click 'Run Simulation' to start")
    st.write("4. Use the day slider to see how the virus spreads over time")
    st.write("5. Deploy the antidote when you're ready to start treating infected people")
    st.write("6. Adjust dot density if you experience performance issues")
    st.write("7. Choose 'Dots Only' display mode to see infected individuals clearly")
   
    st.info("This simulation shows infected individuals as red dots. Run the simulation to begin.")