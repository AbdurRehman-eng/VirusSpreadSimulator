import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import branca.colormap as cm
import random
from folium.plugins import MarkerCluster

# Set page config
st.set_page_config(layout="wide", page_title="Pakistan Virus Spread Simulator")

# App title
st.title("Interactive Virus Spread Simulator - Pakistan")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Parameter inputs
initial_infections = st.sidebar.slider("Initial infection centers", 1, 10, 3)
r_value = st.sidebar.slider("R value (infection rate)", 0.5, 5.0, 2.5)
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

# Simulation function - now without automatic recovery
def run_virus_simulation(initial_cities, r_value, simulation_days, mobility, intervention_day, intervention_strength, antidote_applied=False, antidote_day=None, antidote_effectiveness=0.0):
    # Initialize data structures
    cities_data = {}
    infected_people = {}  # To track individual dots for infected people
    
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
            
            # Create individual infected person dots
            city_infected_people = []
            for _ in range(infected):
                # Create random positions around the city center
                rand_lat = data["lat"] + (random.random() - 0.5) * 0.1
                rand_lon = data["lon"] + (random.random() - 0.5) * 0.1
                city_infected_people.append({
                    "lat": rand_lat,
                    "lon": rand_lon,
                    "day_infected": 0,
                    "recovered": False
                })
            infected_people[city] = city_infected_people
        else:
            infected_people[city] = []
            
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
        
        # Apply antidote if specified
        if antidote_applied and day >= antidote_day:
            for city in infected_people:
                # Mark some infected people as recovered based on effectiveness
                for person in infected_people[city]:
                    if not person["recovered"] and random.random() < antidote_effectiveness:
                        person["recovered"] = True
        
        # Calculate new infections for each city
        for city, data in cities_data.items():
            yesterday = data["timeline"][-1]
            
            # Manual recoveries (only if antidote was applied)
            new_recovered = 0
            if antidote_applied and day >= antidote_day:
                # Count newly recovered people
                for person in infected_people[city]:
                    if person["recovered"] and person["day_infected"] < day:
                        new_recovered += 1
            
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
            
            # Create dots for new infections
            for _ in range(total_new_infected):
                # Create random positions around the city center
                rand_lat = data["lat"] + (random.random() - 0.5) * 0.1
                rand_lon = data["lon"] + (random.random() - 0.5) * 0.1
                infected_people[city].append({
                    "lat": rand_lat,
                    "lon": rand_lon,
                    "day_infected": day,
                    "recovered": False
                })
            
            # Store daily data
            data["timeline"].append({
                "day": day,
                "susceptible": susceptible,
                "infected": infected,
                "recovered": recovered,
                "daily_new_cases": total_new_infected
            })
    
    return cities_data, infected_people

# Create map function with individual infected person dots
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm

def create_pakistan_map(cities_data, infected_people, selected_day):
    # Create map centered on Pakistan
    m = folium.Map(location=[30.3753, 69.3451], zoom_start=6)

    # Determine maximum infections for color scale
    max_infections = max(data["timeline"][selected_day]["infected"] for data in cities_data.values())

    # Define colormap
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'], 
        vmin=0, 
        vmax=max_infections
    )

    # Add city circles and labels
    for city, data in cities_data.items():
        day_data = data["timeline"][selected_day]
        infected_ratio = day_data["infected"] / data["population"]
        infected_percent = infected_ratio * 100
        radius = (data["population"]) ** 0.5 / 50
        color = colormap(day_data["infected"])

        popup_text = f"""
        <strong>{city}</strong><br>
        Population: {data['population']:,}<br>
        Infected: {day_data['infected']:,} ({infected_percent:.2f}%)<br>
        Recovered: {day_data['recovered']:,}<br>
        New cases today: {day_data['daily_new_cases']:,}
        """

        folium.Circle(
            location=[data["lat"], data["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.4,
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)

        # Add city name and infection count
        folium.Marker(
            location=[data["lat"], data["lon"]],
            icon=folium.DivIcon(
                icon_size=(150, 36),
                icon_anchor=(75, 0),
                html=f'<div style="font-size: 10pt; color: black; text-align: center;">{city}<br>{day_data["infected"]:,}</div>'
            )
        ).add_to(m)

    # Add infected dots (limited for performance)
    MAX_DOTS_PER_CITY = 300

    if selected_day <= 50:  # Avoid dots for very large datasets
        marker_group = folium.FeatureGroup(name="Infected Individuals")
        for city, people in infected_people.items():
            count = 0
            for person in people:
                if person["day_infected"] <= selected_day and not person["recovered"]:
                    if count >= MAX_DOTS_PER_CITY:
                        break
                    folium.CircleMarker(
                        location=[person["lat"], person["lon"]],
                        radius=2,
                        color='red',
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"Infected on day {person['day_infected']}"
                    ).add_to(marker_group)
                    count += 1
        marker_group.add_to(m)

    # Add legend
    colormap.caption = "Number of Active Infections"
    colormap.add_to(m)

    return m

# Initialize session state variables if they don't exist
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

if 'antidote_applied' not in st.session_state:
    st.session_state.antidote_applied = False

if 'antidote_day' not in st.session_state:
    st.session_state.antidote_day = 0

if 'antidote_effectiveness' not in st.session_state:
    st.session_state.antidote_effectiveness = 0.5

# Run simulation button
if st.sidebar.button("Run Simulation"):
    # Reset antidote status
    st.session_state.antidote_applied = False
    st.session_state.antidote_day = 0
    
    # Select initial infection cities randomly
    initial_cities = random.sample(list(pakistan_cities.keys()), initial_infections)
    st.sidebar.write("Initial infection cities:", ", ".join(initial_cities))
    
    # Run simulation
    simulation_data, infected_people = run_virus_simulation(
        initial_cities,
        r_value,
        simulation_days,
        mobility_factor,
        intervention_day,
        intervention_strength
    )
    
    # Cache simulation data in session state
    st.session_state.simulation_data = simulation_data
    st.session_state.infected_people = infected_people
    st.session_state.simulation_run = True
    st.session_state.initial_cities = initial_cities

# Display map if simulation has been run
if st.session_state.simulation_run:
    # Create two columns for map and statistics
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Day selector
        selected_day = st.slider("Simulation Day", 0, simulation_days, 0)
        
        # Create map
        map_data = create_pakistan_map(
            st.session_state.simulation_data, 
            st.session_state.infected_people, 
            selected_day
        )
        
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
        
        # Antidote controls
        st.subheader("Antidote Control Panel")
        
        if not st.session_state.antidote_applied:
            antidote_effectiveness = st.slider("Antidote Effectiveness", 0.1, 1.0, 0.5)
            
            if st.button("Deploy Antidote"):
                st.session_state.antidote_applied = True
                st.session_state.antidote_day = selected_day
                st.session_state.antidote_effectiveness = antidote_effectiveness
                
                # Re-run simulation with antidote
                simulation_data, infected_people = run_virus_simulation(
                    st.session_state.initial_cities,
                    r_value,
                    simulation_days,
                    mobility_factor,
                    intervention_day,
                    intervention_strength,
                    True,
                    selected_day,
                    antidote_effectiveness
                )
                
                st.session_state.simulation_data = simulation_data
                st.session_state.infected_people = infected_people
                st.success(f"Antidote deployed on day {selected_day} with {antidote_effectiveness*100:.0f}% effectiveness!")
                st.experimental_rerun()
        else:
            st.success(f"Antidote deployed on day {st.session_state.antidote_day} with {st.session_state.antidote_effectiveness*100:.0f}% effectiveness!")
        
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
    st.write("5. Deploy the antidote when you're ready to start treating infected people")