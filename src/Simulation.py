import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import geopandas as gpd
import json
import branca.colormap as cm
import random

# Set page config
st.set_page_config(layout="wide", page_title="Pakistan Virus Spread Simulator")

# App title
st.title("Interactive Virus Spread Simulator - Pakistan")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Parameter inputs
initial_infections = st.sidebar.slider("Initial infection centers", 1, 10, 3)
r_value = st.sidebar.slider("R value (infection rate)", 0.5, 5.0, 2.5)
recovery_days = st.sidebar.slider("Days to recovery", 7, 21, 14)
simulation_days = st.sidebar.slider("Simulation duration (days)", 10, 120, 60)
mobility_factor = st.sidebar.slider("Population mobility", 0.1, 1.0, 0.5)
intervention_day = st.sidebar.slider("Day to start interventions", 0, 60, 30)
intervention_strength = st.sidebar.slider("Intervention strength", 0.0, 1.0, 0.5)

# Pakistan major cities with coordinates and population estimates
pakistan_cities = {
    "Karachi": {"lat": 24.8607, "lon": 67.0011, "population": 14910000},
    "Lahore": {"lat": 31.5204, "lon": 74.3587, "population": 11126000},
    "Faisalabad": {"lat": 31.4504, "lon": 73.1350, "population": 3204000},
    "Rawalpindi": {"lat": 33.6007, "lon": 73.0679, "population": 2098000},
    "Multan": {"lat": 30.1798, "lon": 71.4214, "population": 1871000},
    "Hyderabad": {"lat": 25.3960, "lon": 68.3578, "population": 1734000},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "population": 1095000},
    "Peshawar": {"lat": 34.0150, "lon": 71.5805, "population": 1970000},
    "Quetta": {"lat": 30.1798, "lon": 66.9750, "population": 1001000},
    "Sialkot": {"lat": 32.4945, "lon": 74.5229, "population": 655000}
}

# Simulation function
def run_virus_simulation(initial_cities, r_value, recovery_days, simulation_days, mobility, intervention_day, intervention_strength):
    # Initialize data structures
    cities_data = {}
    for city, data in pakistan_cities.items():
        # Initially everyone is susceptible
        susceptible = data["population"]
        infected = 0
        recovered = 0
        
        # Set initial infections
        if city in initial_cities:
            initial_infected = random.randint(10, 100)  # Random number of initial cases
            infected = initial_infected
            susceptible -= initial_infected
            
        cities_data[city] = {
            "lat": data["lat"],
            "lon": data["lon"],
            "population": data["population"],
            "timeline": [{
                "day": 0,
                "susceptible": susceptible,
                "infected": infected,
                "recovered": recovered,
                "daily_new_cases": infected if infected > 0 else 0
            }]
        }
    
    # Run simulation for each day
    for day in range(1, simulation_days + 1):
        # Apply intervention effect if past intervention day
        current_r = r_value
        if day >= intervention_day:
            current_r = r_value * (1 - intervention_strength)
        
        # Calculate new infections and recoveries for each city
        for city, data in cities_data.items():
            yesterday = data["timeline"][-1]
            
            # Natural recoveries
            new_recovered = int(yesterday["infected"] / recovery_days)
            
            # New infections from within the city
            new_infected_local = int(yesterday["infected"] * current_r * (yesterday["susceptible"] / data["population"]))
            
            # Infections from other cities (mobility factor)
            new_infected_travelers = 0
            if mobility > 0:
                for other_city, other_data in cities_data.items():
                    if other_city != city:
                        other_yesterday = other_data["timeline"][-1]
                        # Calculate based on distance and infected population
                        infected_ratio = other_yesterday["infected"] / other_data["population"]
                        # Simple model for travel-based infections
                        new_infected_travelers += int(infected_ratio * mobility * 10)  # Simplified model
            
            total_new_infected = min(yesterday["susceptible"], new_infected_local + new_infected_travelers)
            
            # Update counts
            infected = yesterday["infected"] + total_new_infected - new_recovered
            susceptible = yesterday["susceptible"] - total_new_infected
            recovered = yesterday["recovered"] + new_recovered
            
            # Ensure non-negative values
            infected = max(0, infected)
            susceptible = max(0, susceptible)
            recovered = max(0, recovered)
            
            # Store daily data
            data["timeline"].append({
                "day": day,
                "susceptible": susceptible,
                "infected": infected,
                "recovered": recovered,
                "daily_new_cases": total_new_infected
            })
    
    return cities_data

# Create map function
def create_pakistan_map(cities_data, selected_day):
    # Create map centered on Pakistan
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=6)
    
    # Create color scale for infection levels
    max_infections = 0
    for city, data in cities_data.items():
        city_infections = data["timeline"][selected_day]["infected"]
        if city_infections > max_infections:
            max_infections = city_infections
    
    # Create color map based on infection rates
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'], 
        vmin=0, 
        vmax=max_infections
    )
    
    # Add markers for cities
    for city, data in cities_data.items():
        day_data = data["timeline"][selected_day]
        infected_ratio = day_data["infected"] / data["population"]
        infected_percent = infected_ratio * 100
        
        # Circle size based on population
        radius = (data["population"]) ** 0.5 / 50
        
        # Color based on infection rate
        color = colormap(day_data["infected"])
        
        # Create popup text
        popup_text = f"""
        <strong>{city}</strong><br>
        Population: {data['population']:,}<br>
        Infected: {day_data['infected']:,} ({infected_percent:.2f}%)<br>
        Recovered: {day_data['recovered']:,}<br>
        New cases today: {day_data['daily_new_cases']:,}
        """
        
        # Add circle
        folium.Circle(
            location=[data["lat"], data["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
        
        # Add city label
        folium.Marker(
            location=[data["lat"], data["lon"]],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 0),
                html=f'<div style="font-size: 10pt; color: black; text-align: center;">{city}<br>{day_data["infected"]:,}</div>'
            )
        ).add_to(m)
    
    # Add a legend
    colormap.caption = "Number of Active Infections"
    colormap.add_to(m)
    
    return m

# Run simulation button
if st.sidebar.button("Run Simulation"):
    # Select initial infection cities randomly
    initial_cities = random.sample(list(pakistan_cities.keys()), initial_infections)
    st.sidebar.write("Initial infection cities:", ", ".join(initial_cities))
    
    # Run simulation
    simulation_data = run_virus_simulation(
        initial_cities,
        r_value,
        recovery_days,
        simulation_days,
        mobility_factor,
        intervention_day,
        intervention_strength
    )
    
    # Cache simulation data in session state
    st.session_state.simulation_data = simulation_data
    st.session_state.simulation_run = True

# Display map if simulation has been run
if 'simulation_run' in st.session_state and st.session_state.simulation_run:
    # Create two columns for map and statistics
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Day selector
        selected_day = st.slider("Simulation Day", 0, simulation_days, 0)
        
        # Create map
        map_data = create_pakistan_map(st.session_state.simulation_data, selected_day)
        
        # Render the map
        st_folium(map_data, width=800, height=600)
    
    with col2:
        # Display statistics
        st.subheader(f"Day {selected_day} Statistics")
        
        # Calculate totals
        total_infected = 0
        total_recovered = 0
        total_susceptible = 0
        total_new_cases = 0
        
        for city, data in st.session_state.simulation_data.items():
            day_data = data["timeline"][selected_day]
            total_infected += day_data["infected"]
            total_recovered += day_data["recovered"]
            total_susceptible += day_data["susceptible"]
            total_new_cases += day_data["daily_new_cases"]
        
        # Display summary statistics
        st.metric("Total Infected", f"{total_infected:,}")
        st.metric("New Cases Today", f"{total_new_cases:,}")
        st.metric("Total Recovered", f"{total_recovered:,}")
        
        # Show intervention status
        if selected_day >= intervention_day:
            st.info(f"Interventions active (reducing R by {intervention_strength*100:.0f}%)")
        else:
            st.warning(f"No interventions yet (starting day {intervention_day})")
        
        # Show top 3 most infected cities
        st.subheader("Most affected cities:")
        city_infections = []
        for city, data in st.session_state.simulation_data.items():
            day_data = data["timeline"][selected_day]
            city_infections.append((city, day_data["infected"], day_data["infected"]/data["population"]*100))
        
        # Sort by number of infections
        city_infections.sort(key=lambda x: x[1], reverse=True)
        
        # Display top 3
        for i, (city, infected, percentage) in enumerate(city_infections[:3]):
            st.write(f"{i+1}. {city}: {infected:,} cases ({percentage:.2f}%)")
else:
    # Instructions if simulation hasn't been run
    st.write("### Instructions")
    st.write("1. Adjust the simulation parameters in the sidebar")
    st.write("2. Click 'Run Simulation' to start")
    st.write("3. Use the day slider to see how the virus spreads over time")
    st.write("4. Click on city circles for detailed information")