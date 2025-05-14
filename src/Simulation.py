import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import branca.colormap as cm

# Set page config
st.set_page_config(layout="wide", page_title="Islamabad Virus Spread Simulator")

# App title
st.title("Interactive Virus Spread Simulator - Islamabad")

# Sidebar for simulation parameters
st.sidebar.header("Simulation Parameters")

# Parameter inputs
initial_infections = st.sidebar.slider("Initial infection centers", 1, 5, 2)
r_value = st.sidebar.slider("R value (infection rate)", 0.5, 5.0, 2.5)
simulation_days = st.sidebar.slider("Simulation duration (days)", 10, 90, 45)
intervention_day = st.sidebar.slider("Day to start interventions", 0, 45, 15)
intervention_strength = st.sidebar.slider("Intervention strength", 0.0, 1.0, 0.5)

# Add a parameter to control dot density
dot_density = st.sidebar.slider("Dot density", 0.1, 1.0, 0.5, 
                               help="Controls how many dots are shown (higher values show more dots but may affect performance)")

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
   
    return sectors_data, infected_people

# Create map with focus on Islamabad - OPTIMIZED VERSION
def create_islamabad_map(sectors_data, infected_people, selected_day, dot_density=0.5, show_sectors=False):
    # Create map centered on Islamabad
    m = folium.Map(location=[33.6844, 73.0479], zoom_start=12, tiles="CartoDB positron")
   
    # Filter infected people for current day
    current_infected = [p for p in infected_people if p.day_infected <= selected_day and not p.is_recovered(selected_day)]
   
    # Add sector circles if requested
    if show_sectors:
        for sector_name, data in sectors_data.items():
            sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector_name), None)
            if not sector_info:
                continue
               
            # Get data for current day
            day_data = next((d for d in data["timeline"] if d["day"] == selected_day), None)
            if not day_data:
                continue
           
            # Calculate infection percentage
            infection_percent = (day_data["active_infections"] / sector_info["population"]) * 100
           
            # Determine color based on infection rate
            if infection_percent > 5:
                color = "red"
            elif infection_percent > 1:
                color = "orange"
            elif infection_percent > 0.1:
                color = "yellow"
            else:
                color = "green"
           
            # Create popup with sector information
            popup_text = f"""
            <strong>{sector_name}</strong><br>
            Population: {sector_info['population']:,}<br>
            Active Cases: {day_data['active_infections']:,} ({infection_percent:.2f}%)<br>
            Total Infections: {day_data['cumulative_infections']:,}<br>
            New Cases Today: {day_data['daily_new_cases']:,}
            """
           
            # Add sector circle with reduced opacity to better see dots
            folium.Circle(
                location=[sector_info["lat"], sector_info["lon"]],
                radius=300,  # Fixed radius for sectors
                color=color,
                fill=True,
                fill_opacity=0.1,  # Reduced opacity to see dots better
                popup=folium.Popup(popup_text, max_width=250)
            ).add_to(m)
           
            # Add sector label
            folium.Marker(
                location=[sector_info["lat"], sector_info["lon"]],
                icon=folium.DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(75, 0),
                    html=f'<div style="font-size: 10pt; color: black; text-align: center;">{sector_name}<br>{day_data["active_infections"]}</div>'
                )
            ).add_to(m)
   
    # Add infected people as dots - ALWAYS show dots from day 0
    # Limit number of dots for performance
    MAX_DOTS = int(2000 * dot_density)  # Adjust based on dot density parameter
   
    # Create feature group just once
    marker_group = folium.FeatureGroup(name="Infected Individuals")
   
    # If we have a reasonable number of dots, show them all
    if len(current_infected) <= MAX_DOTS:
        # Use a list comprehension to create all the circle marker options first
        for person in current_infected:
            folium.CircleMarker(
                location=[person.lat, person.lon],
                radius=2,
                color='red',
                fill=True,
                fill_opacity=0.7,
                weight=1
            ).add_to(marker_group)
    else:
        # Efficient sampling for larger numbers
        # Use stride sampling instead of random sampling for better performance
        sample_stride = max(1, len(current_infected) // MAX_DOTS)
        sampled_infected = current_infected[::sample_stride]
       
        # Use the same efficient approach for the sampled dots
        for person in sampled_infected:
            folium.CircleMarker(
                location=[person.lat, person.lon],
                radius=2,
                color='red',
                fill=True,
                fill_opacity=0.7,
                weight=1
            ).add_to(marker_group)
   
    # Add the marker group to the map
    marker_group.add_to(m)
   
    # Add title
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50%; transform: translateX(-50%);
                z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 1px solid grey;">
        <h4>Islamabad COVID-19 Simulation - Day {}</h4>
        <p style="margin: 0; text-align: center;">{} Active Cases</p>
    </div>
    '''.format(selected_day, len(current_infected))
    m.get_root().html.add_child(folium.Element(title_html))
   
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

if 'show_sectors' not in st.session_state:
    st.session_state.show_sectors = False

# Add toggle for showing sector circles
st.sidebar.subheader("Display Options")
show_sectors = st.sidebar.checkbox("Show sector circles", value=False)
st.session_state.show_sectors = show_sectors

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Reset antidote status
        st.session_state.antidote_applied = False
        st.session_state.antidote_day = 0
       
        # Run simulation
        sectors_data, infected_people = run_virus_simulation(
            r_value,
            simulation_days,
            intervention_day,
            intervention_strength
        )
       
        # Cache simulation data in session state
        st.session_state.sectors_data = sectors_data
        st.session_state.infected_people = infected_people
        st.session_state.simulation_run = True
       
        # Initialize the map immediately to avoid the AttributeError
        st.session_state.selected_day = 0
        st.session_state.current_map = create_islamabad_map(
            sectors_data,
            infected_people,
            0,  # Start with day 0
            dot_density,
            st.session_state.show_sectors
        )

# Create a single map that's less likely to cause rerendering issues
if st.session_state.simulation_run:
    # Create two columns for map and statistics
    col1, col2 = st.columns([3, 1])
   
    with col1:
        # Make the slider a session state variable to prevent map rerendering
        if 'selected_day' not in st.session_state:
            st.session_state.selected_day = 0
           
        # Initialize current_map if it doesn't exist yet
        if 'current_map' not in st.session_state:
            st.session_state.current_map = create_islamabad_map(
                st.session_state.sectors_data,
                st.session_state.infected_people,
                0,  # Start with day 0
                dot_density,
                st.session_state.show_sectors
            )
           
        # Day selector
        selected_day = st.slider(
            "Simulation Day",
            0,
            simulation_days,
            st.session_state.selected_day,
            key="day_slider"
        )
       
        # Only recreate the map if the day has changed or display options changed
        if (selected_day != st.session_state.selected_day or 
            show_sectors != st.session_state.show_sectors):
            
            st.session_state.selected_day = selected_day
            st.session_state.show_sectors = show_sectors
           
            # Create and cache the map
            with st.spinner("Updating map..."):
                st.session_state.current_map = create_islamabad_map(
                    st.session_state.sectors_data,
                    st.session_state.infected_people,
                    selected_day,
                    dot_density,
                    show_sectors
                )
       
        # Always use the cached map
        st_folium(
            st.session_state.current_map,
            width=800,
            height=600,
            # Add a key to prevent rerendering
            key="map_display"
        )
   
    with col2:
        # Display statistics
        st.subheader(f"Day {selected_day} Statistics")
       
        # Calculate totals
        total_active = 0
        total_cumulative = 0
        total_new_cases = 0
       
        for sector_name, data in st.session_state.sectors_data.items():
            day_data = next((d for d in data["timeline"] if d["day"] == selected_day), None)
            if day_data:
                total_active += day_data["active_infections"]
                total_cumulative += day_data["cumulative_infections"]
                total_new_cases += day_data["daily_new_cases"]
       
        # Display summary statistics
        st.metric("Active Cases", f"{total_active:,}")
        st.metric("New Cases Today", f"{total_new_cases:,}")
        st.metric("Total Infections", f"{total_cumulative:,}")
       
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
                with st.spinner("Applying antidote..."):
                    st.session_state.antidote_applied = True
                    st.session_state.antidote_day = selected_day
                    st.session_state.antidote_effectiveness = antidote_effectiveness
                   
                    # Re-run simulation with antidote
                    sectors_data, infected_people = run_virus_simulation(
                        r_value,
                        simulation_days,
                        intervention_day,
                        intervention_strength,
                        True,
                        selected_day,
                        antidote_effectiveness
                    )
                   
                    st.session_state.sectors_data = sectors_data
                    st.session_state.infected_people = infected_people
                    st.success(f"Antidote deployed on day {selected_day} with {antidote_effectiveness*100:.0f}% effectiveness!")
                   
                    # Recreate the map for the current day
                    st.session_state.current_map = create_islamabad_map(
                        st.session_state.sectors_data,
                        st.session_state.infected_people,
                        selected_day,
                        dot_density,
                        st.session_state.show_sectors
                    )
        else:
            st.success(f"Antidote deployed on day {st.session_state.antidote_day} with {st.session_state.antidote_effectiveness*100:.0f}% effectiveness!")
       
        # Show top 3 most infected sectors
        st.subheader("Most affected sectors:")
        sector_infections = []
       
        for sector_name, data in st.session_state.sectors_data.items():
            sector_info = next((s for s in islamabad["sectors"] if s["name"] == sector_name), None)
            day_data = next((d for d in data["timeline"] if d["day"] == selected_day), None)
           
            if sector_info and day_data:
                infection_percent = (day_data["active_infections"] / sector_info["population"]) * 100
                sector_infections.append((sector_name, day_data["active_infections"], infection_percent))
       
        # Sort by number of infections
        sector_infections.sort(key=lambda x: x[1], reverse=True)
       
        # Display top 3
        for i, (sector, infected, percentage) in enumerate(sector_infections[:3]):
            st.write(f"{i+1}. {sector}: {infected:,} cases ({percentage:.2f}%)")
else:
    # Instructions if simulation hasn't been run
    st.write("### Instructions")
    st.write("1. Adjust the simulation parameters in the sidebar")
    st.write("2. Click 'Run Simulation' to start")
    st.write("3. Use the day slider to see how the virus spreads over time")
    st.write("4. Deploy the antidote when you're ready to start treating infected people")
    st.write("5. Adjust dot density if you experience performance issues")
   
    st.info("This simulation focuses on Islamabad with infected individuals shown as red dots.")