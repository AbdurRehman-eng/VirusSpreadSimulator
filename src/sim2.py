import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import networkx as nx
import random
import time
import warnings
import pandas as pd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import json
import os

warnings.filterwarnings("ignore")

class SIRModel:
    """
    SIR (Susceptible, Infected, Recovered) model for epidemic simulation on a network
    """
    def __init__(self, network, infection_prob=0.3, recovery_days=14):
        """
        Initialize the SIR model with parameters
        
        Parameters:
        -----------
        network : networkx.Graph
            The social network on which the disease spreads
        infection_prob : float
            Probability of infection between connected individuals
        recovery_days : int
            Number of days after which an infected person recovers
        """
        self.network = network
        self.infection_prob = infection_prob
        self.recovery_days = recovery_days
        self.day = 0
        
        # Initialize all nodes with health status
        for node in self.network.nodes():
            self.network.nodes[node]['health'] = 'susceptible'
            self.network.nodes[node]['days_infected'] = 0
        
        # Statistics tracking
        self.stats = {
            'susceptible': [sum(1 for n in network.nodes() if self.network.nodes[n]['health'] == 'susceptible')],
            'infected': [0],
            'recovered': [0],
            'days': [0]
        }
    
    def seed_infection(self, num_patients=5):
        """
        Randomly select initial infected nodes
        
        Parameters:
        -----------
        num_patients : int
            Number of initially infected individuals
        """
        patients_zero = random.sample(list(self.network.nodes()), num_patients)
        for node in patients_zero:
            self.network.nodes[node]['health'] = 'infected'
        
        # Update statistics
        self.stats['infected'][0] = num_patients
        self.stats['susceptible'][0] -= num_patients
    
    def spread_disease(self):
        """
        Execute one day of disease spread
        
        Returns:
        --------
        int : Number of new infections
        """
        self.day += 1
        new_infections = 0
        new_recoveries = 0
        
        # Process infections and recoveries
        for node in self.network.nodes():
            if self.network.nodes[node]['health'] == 'infected':
                # Check for recovery
                self.network.nodes[node]['days_infected'] += 1
                if self.network.nodes[node]['days_infected'] >= self.recovery_days:
                    self.network.nodes[node]['health'] = 'recovered'
                    new_recoveries += 1
                else:
                    # Try to infect neighbors
                    for neighbor in self.network.neighbors(node):
                        if (self.network.nodes[neighbor]['health'] == 'susceptible' and 
                            random.random() < self.infection_prob):
                            self.network.nodes[neighbor]['health'] = 'infected'
                            new_infections += 1
        
        # Update statistics
        self.stats['infected'].append(self.stats['infected'][-1] + new_infections - new_recoveries)
        self.stats['recovered'].append(self.stats['recovered'][-1] + new_recoveries)
        self.stats['susceptible'].append(self.stats['susceptible'][-1] - new_infections)
        self.stats['days'].append(self.day)
        
        return new_infections
    
    def run_simulation(self, days=100):
        """
        Run the complete simulation for a specified number of days
        
        Parameters:
        -----------
        days : int
            Maximum number of days to simulate
            
        Returns:
        --------
        dict : Simulation statistics
        """
        print(f"Running simulation for up to {days} days...")
        for i in range(days):
            infections = self.spread_disease()
            if i % 10 == 0:
                print(f"Day {i}: {self.stats['infected'][-1]} infected, {self.stats['recovered'][-1]} recovered")
            if infections == 0 and self.stats['infected'][-1] == 0:
                print(f"Disease extinct after {self.day} days")
                break
        
        # Calculate additional statistics
        results = {
            "days": self.day,
            "stats": self.stats,
            "peak_infected": max(self.stats['infected']),
            "final_recovered": self.stats['recovered'][-1],
            "total_affected": self.stats['recovered'][-1] + self.stats['infected'][-1],
            "percent_affected": (self.stats['recovered'][-1] + self.stats['infected'][-1]) / 
                               (self.stats['susceptible'][0] + self.stats['infected'][0]) * 100
        }
        
        return results
    
    def save_results(self, filepath="data/sir_results.json"):
        """
        Save simulation results to a JSON file
        
        Parameters:
        -----------
        filepath : str
            Path to save the results
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results = {
            "days": self.day,
            "stats": self.stats,
            "peak_infected": max(self.stats['infected']),
            "final_recovered": self.stats['recovered'][-1],
            "total_affected": self.stats['recovered'][-1] + self.stats['infected'][-1],
            "percent_affected": (self.stats['recovered'][-1] + self.stats['infected'][-1]) / 
                               (self.stats['susceptible'][0] + self.stats['infected'][0]) * 100
        }
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return results
    
    def plot_results(self):
        """
        Plot simulation results
        
        Returns:
        --------
        matplotlib.figure.Figure : Plot of SIR model results
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        days = np.array(self.stats['days'])
        ax.plot(days, self.stats['susceptible'], 'b-', label='Susceptible')
        ax.plot(days, self.stats['infected'], 'r-', label='Infected')
        ax.plot(days, self.stats['recovered'], 'g-', label='Recovered')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Population')
        ax.set_title(f'SIR Model Results (Network with {len(self.network.nodes())} individuals)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


class NetworkGenerator:
    """
    Generate various types of networks for epidemic simulation
    """
    
    @staticmethod
    def generate_barabasi_albert(num_nodes=5000, avg_connections=3):
        """
        Generate scale-free network using Barabási-Albert model
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the network
        avg_connections : int
            Number of edges to attach from a new node to existing nodes
            
        Returns:
        --------
        networkx.Graph : Generated network
        """
        print(f"Generating Barabási-Albert network with {num_nodes} nodes...")
        G = nx.barabasi_albert_graph(n=num_nodes, m=avg_connections)
        print(f"Network generated with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    @staticmethod
    def generate_watts_strogatz(num_nodes=5000, k=4, p=0.1):
        """
        Generate small-world network using Watts-Strogatz model
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the network
        k : int
            Each node is connected to k nearest neighbors in ring topology
        p : float
            Probability of rewiring each edge
            
        Returns:
        --------
        networkx.Graph : Generated network
        """
        print(f"Generating Watts-Strogatz small-world network with {num_nodes} nodes...")
        G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p)
        print(f"Network generated with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    @staticmethod
    def generate_erdos_renyi(num_nodes=5000, p=0.001):
        """
        Generate random network using Erdős-Rényi model
        
        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the network
        p : float
            Probability of edge creation between any two nodes
            
        Returns:
        --------
        networkx.Graph : Generated network
        """
        print(f"Generating Erdős-Rényi random network with {num_nodes} nodes...")
        G = nx.erdos_renyi_graph(n=num_nodes, p=p)
        print(f"Network generated with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    @staticmethod
    def save_network(G, gexf_path="data/social_network.gexf", stats_path="data/network_stats.json"):
        """
        Save network to files
        
        Parameters:
        -----------
        G : networkx.Graph
            Network to save
        gexf_path : str
            Path to save the network in GEXF format
        stats_path : str
            Path to save network statistics
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(gexf_path), exist_ok=True)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        # Save network
        nx.write_gexf(G, gexf_path)
        
        # Calculate and save statistics
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        
        with open(stats_path, "w") as f:
            json.dump({
                "nodes": len(G.nodes()),
                "edges": len(G.edges()),
                "average_degree": sum(degree_sequence) / len(degree_sequence),
                "max_degree": max(degree_sequence),
                "degree_histogram": nx.degree_histogram(G)[:20],
                "clustering_coefficient": nx.average_clustering(G)
            }, f, indent=2)
        
        print(f"Network saved to {gexf_path}")
        print(f"Network statistics saved to {stats_path}")


class GeoSIRModel:
    """
    Geographic SIR (Susceptible, Infected, Recovered) model for epidemic simulation
    """
    def __init__(self, population_size=5000, initial_infections=5, 
                 infection_radius=0.01, recovery_rate=0.02, 
                 city_center=(73.084488, 33.684422), city_radius=0.18):
        """
        Initialize the model with parameters
        
        Parameters:
        -----------
        population_size : int
            Number of individuals in the simulation
        initial_infections : int
            Number of initially infected individuals
        infection_radius : float
            Distance within which infection can spread (in degrees)
        recovery_rate : float
            Probability of recovery each day
        city_center : tuple
            (longitude, latitude) of city center
        city_radius : float
            Radius of the city area in degrees
        """
        self.population_size = population_size
        self.initial_infections = initial_infections
        self.infection_radius = infection_radius
        self.recovery_rate = recovery_rate
        self.city_center = city_center
        self.city_radius = city_radius
        
        # City boundaries (approximate)
        self.lon_min, self.lon_max = city_center[0] - city_radius, city_center[0] + city_radius
        self.lat_min, self.lat_max = city_center[1] - city_radius, city_center[1] + city_radius
        
        # Initialize population
        self.initialize_population()
        
        # Statistics tracking
        self.stats = {
            'susceptible': [self.population_size - self.initial_infections],
            'infected': [self.initial_infections],
            'recovered': [0],
            'days': [0],
            'clusters': [self.initial_infections]
        }
        
        # Current simulation day
        self.current_day = 0
        
        # Create graph for network analysis
        self.G = nx.Graph()
        
    def initialize_population(self):
        """Initialize the population with realistic distribution"""
        # Generate random positions with natural clustering
        self.positions = np.zeros((self.population_size, 2))
        
        # Create a realistic population distribution
        points_placed = 0
        
        # Generate uniformly within boundary first
        while points_placed < self.population_size:
            remaining = self.population_size - points_placed
            
            # Generate uniform random points within the bounding box
            lons = np.random.uniform(self.lon_min, self.lon_max, remaining)
            lats = np.random.uniform(self.lat_min, self.lat_max, remaining)
            
            # Check if they're within the city radius
            distances = np.sqrt((lons - self.city_center[0])**2 + 
                               (lats - self.city_center[1])**2)
            
            valid_points = distances <= self.city_radius
            valid_count = np.sum(valid_points)
            
            if valid_count > 0:
                self.positions[points_placed:points_placed+valid_count, 0] = lons[valid_points]
                self.positions[points_placed:points_placed+valid_count, 1] = lats[valid_points]
                points_placed += valid_count
        
        # Apply density adjustment - people tend to cluster in urban areas
        for _ in range(3):  # Apply multiple passes
            # Pick some random points as centers
            cluster_centers = self.positions[np.random.choice(self.population_size, size=50, replace=False)]
            
            # For each point, with some probability, move it closer to a random center
            for i in range(self.population_size):
                if np.random.random() < 0.7:  # 70% of points are adjusted
                    # Select a random center
                    center = cluster_centers[np.random.randint(0, len(cluster_centers))]
                    
                    # Move point 30-70% of the way toward the center
                    weight = np.random.uniform(0.3, 0.7)
                    self.positions[i] = (1 - weight) * self.positions[i] + weight * center
        
        # Initialize states (0=susceptible, 1=infected, 2=recovered)
        self.states = np.zeros(self.population_size, dtype=int)
        
        # Set initial infections
        initial_infected = np.random.choice(self.population_size, size=self.initial_infections, replace=False)
        self.states[initial_infected] = 1
        
        # Days since infection (0 for susceptible and recovered)
        self.days_infected = np.zeros(self.population_size, dtype=int)
        
    def update(self):
        """Update the simulation by one day"""
        self.current_day += 1
        
        # Update infection status
        self.update_infections()
        
        # Update recovery status
        self.update_recoveries()
        
        # Update statistics
        self.update_statistics()
        
    def update_infections(self):
        """Update infection spread based on proximity"""
        # Find all susceptible individuals
        susceptible_indices = np.where(self.states == 0)[0]
        infected_indices = np.where(self.states == 1)[0]
        
        # If there are no infections or no susceptible individuals, return
        if len(infected_indices) == 0 or len(susceptible_indices) == 0:
            return
        
        # Get positions
        susceptible_positions = self.positions[susceptible_indices]
        infected_positions = self.positions[infected_indices]
        
        # Process in batches to avoid memory issues
        new_infections = []
        batch_size = 1000  # Adjust based on memory constraints
        
        for i in range(0, len(susceptible_indices), batch_size):
            batch_end = min(i + batch_size, len(susceptible_indices))
            batch_indices = susceptible_indices[i:batch_end]
            batch_positions = susceptible_positions[i:batch_end]
            
            # For each susceptible person, check if they're close to any infected person
            for j, (sus_idx, sus_pos) in enumerate(zip(batch_indices, batch_positions)):
                # Calculate distances to all infected
                dists = np.sqrt(np.sum((infected_positions - sus_pos)**2, axis=1))
                
                # Check if any infected person is within infection radius
                if np.any(dists <= self.infection_radius):
                    new_infections.append(sus_idx)
        
        # Update states for newly infected individuals
        self.states[new_infections] = 1
        
    def update_recoveries(self):
        """Update recovery status based on recovery rate"""
        # Find all infected individuals
        infected_indices = np.where(self.states == 1)[0]
        
        # Increment days infected
        self.days_infected[infected_indices] += 1
        
        # Determine recoveries (chance increases with days infected)
        recovery_chances = self.recovery_rate * self.days_infected[infected_indices]
        recovery_chances = np.clip(recovery_chances, 0, 0.7)  # Cap at 70% chance per day
        
        # Generate random numbers for each infected individual
        random_vals = np.random.random(len(infected_indices))
        
        # Identify those who recover
        recoveries = infected_indices[random_vals < recovery_chances]
        
        # Update states and reset days infected
        self.states[recoveries] = 2
        self.days_infected[recoveries] = 0
        
    def update_statistics(self):
        """Update tracking statistics"""
        # Count states
        susceptible_count = np.sum(self.states == 0)
        infected_count = np.sum(self.states == 1)
        recovered_count = np.sum(self.states == 2)
        
        # Count clusters using network analysis
        cluster_count = self.count_clusters()
        
        # Record statistics
        self.stats['susceptible'].append(susceptible_count)
        self.stats['infected'].append(infected_count)
        self.stats['recovered'].append(recovered_count)
        self.stats['days'].append(self.current_day)
        self.stats['clusters'].append(cluster_count)
        
    def count_clusters(self):
        """Count infection clusters using network analysis"""
        # Reset graph
        self.G.clear()
        
        # Add all infected individuals as nodes
        infected_indices = np.where(self.states == 1)[0]
        for idx in infected_indices:
            self.G.add_node(idx)
        
        # Connect nodes that are close to each other
        for i in range(len(infected_indices)):
            for j in range(i+1, len(infected_indices)):
                idx1, idx2 = infected_indices[i], infected_indices[j]
                pos1, pos2 = self.positions[idx1], self.positions[idx2]
                
                # Calculate distance
                dist = np.sqrt(np.sum((pos1 - pos2)**2))
                
                # If within infection radius, add edge
                if dist <= self.infection_radius * 2:  # Use double radius to identify clusters
                    self.G.add_edge(idx1, idx2)
        
        # Count connected components (clusters)
        if len(self.G) > 0:
            return nx.number_connected_components(self.G)
        else:
            return 0
    
    def run_simulation(self, days):
        """Run the simulation for a specified number of days"""
        print(f"Running geographic simulation for {days} days...")
        for day in range(days):
            self.update()
            if day % 5 == 0:
                print(f"Day {day}: {self.stats['infected'][-1]} infected, "
                      f"{self.stats['recovered'][-1]} recovered, "
                      f"{self.stats['clusters'][-1]} clusters")
                
            # Stop if disease is extinct
            if self.stats['infected'][-1] == 0:
                print(f"Disease extinct after {self.current_day} days")
                break
                
        return self.stats
    
    def get_state_positions(self):
        """Return positions grouped by state"""
        susceptible = self.positions[self.states == 0]
        infected = self.positions[self.states == 1]
        recovered = self.positions[self.states == 2]
        
        return susceptible, infected, recovered
        
    def get_cluster_data(self):
        """Get detailed information about infection clusters"""
        # Reset graph
        self.G.clear()
        
        # Add all infected individuals as nodes
        infected_indices = np.where(self.states == 1)[0]
        for idx in infected_indices:
            self.G.add_node(idx)
        
        # Connect nodes that are close to each other
        for i in range(len(infected_indices)):
            for j in range(i+1, len(infected_indices)):
                idx1, idx2 = infected_indices[i], infected_indices[j]
                pos1, pos2 = self.positions[idx1], self.positions[idx2]
                
                # Calculate distance
                dist = np.sqrt(np.sum((pos1 - pos2)**2))
                
                # If within infection radius, add edge
                if dist <= self.infection_radius * 2:  # Use double radius to identify clusters
                    self.G.add_edge(idx1, idx2)
        
        # Get connected components (clusters)
        clusters = list(nx.connected_components(self.G))
        
        # Prepare cluster data
        cluster_data = []
        for i, cluster in enumerate(clusters):
            cluster_list = list(cluster)
            # Get positions of all infected individuals in this cluster
            cluster_positions = np.array([self.positions[idx] for idx in cluster_list])
            
            # Calculate center and size
            center = np.mean(cluster_positions, axis=0) if len(cluster_positions) > 0 else (0, 0)
            size = len(cluster)
            
            cluster_data.append({
                'id': i,
                'center': center,
                'size': size,
                'positions': cluster_positions
            })
        
        return cluster_data
    
    def plot_results(self):
        """
        Plot simulation results
        
        Returns:
        --------
        matplotlib.figure.Figure : Plot of SIR model results
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot the population map
        susceptible, infected, recovered = self.get_state_positions()
        
        if len(susceptible) > 0:
            ax1.scatter(susceptible[:, 0], susceptible[:, 1], s=5, color='blue', alpha=0.5, label='Susceptible')
        if len(infected) > 0:
            ax1.scatter(infected[:, 0], infected[:, 1], s=20, color='red', alpha=0.7, label='Infected')
        if len(recovered) > 0:
            ax1.scatter(recovered[:, 0], recovered[:, 1], s=5, color='green', alpha=0.5, label='Recovered')
        
        # Draw city boundary
        theta = np.linspace(0, 2*np.pi, 100)
        x = self.city_center[0] + self.city_radius * np.cos(theta)
        y = self.city_center[1] + self.city_radius * np.sin(theta)
        ax1.plot(x, y, 'k--', alpha=0.3)
        
        ax1.set_title(f'Geographic SIR Model (Day {self.current_day})')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot SIR curves
        days = np.array(self.stats['days'])
        ax2.plot(days, self.stats['susceptible'], 'b-', label='Susceptible')
        ax2.plot(days, self.stats['infected'], 'r-', label='Infected')
        ax2.plot(days, self.stats['recovered'], 'g-', label='Recovered')
        ax2.plot(days, self.stats['clusters'], 'k--', label='Clusters')
        
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Population')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig


def create_interactive_map(model, days=30):
    """
    Create an interactive visualization of the disease spread
    
    Parameters:
    -----------
    model : GeoSIRModel
        The epidemic model
    days : int
        Number of days to simulate
    
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.animation.FuncAnimation
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Set initial extent
    ax.set_xlim(model.lon_min, model.lon_max)
    ax.set_ylim(model.lat_min, model.lat_max)
    
    # Create scatter plots for each group
    susceptible_plot = ax.scatter([], [], s=10, color='blue', alpha=0.5, label='Susceptible')
    infected_plot = ax.scatter([], [], s=30, color='red', alpha=0.7, label='Infected')
    recovered_plot = ax.scatter([], [], s=10, color='green', alpha=0.5, label='Recovered')
    
    # Draw city boundary
    theta = np.linspace(0, 2*np.pi, 100)
    x = model.city_center[0] + model.city_radius * np.cos(theta)
    y = model.city_center[1] + model.city_radius * np.sin(theta)
    ax.plot(x, y, 'k--', alpha=0.3)
    
    # Setup legend and title
    ax.legend(loc='upper left')
    title = ax.set_title(f'Day 0: {model.stats["infected"][0]} infected, '
                       f'{model.stats["susceptible"][0]} susceptible, '
                       f'{model.stats["recovered"][0]} recovered, '
                       f'{model.stats["clusters"][0]} clusters')
    
    # Add statistics subplot
    divider = make_axes_locatable(ax)
    stats_ax = divider.append_axes("bottom", size="20%", pad=0.5)
    
    # Initialize statistics lines
    days = np.array(model.stats['days'])
    infected_line, = stats_ax.plot(days, model.stats['infected'], 'r-', label='Infected')
    susceptible_line, = stats_ax.plot(days, model.stats['susceptible'], 'b-', label='Susceptible')
    recovered_line, = stats_ax.plot(days, model.stats['recovered'], 'g-', label='Recovered')
    clusters_line, = stats_ax.plot(days, model.stats['clusters'], 'k--', label='Clusters')
    
    stats_ax.set_xlabel('Days')
    stats_ax.set_ylabel('Population')
    stats_ax.legend(loc='upper right')
    stats_ax.grid(True, linestyle='--', alpha=0.7)
    
    # Animation update function
    def update(frame):
        # Skip day 0 as we already initialized it
        if frame > 0:
            model.update()
        
        # Get updated positions
        susceptible, infected, recovered = model.get_state_positions()
        
        # Update scatter plots
        if len(susceptible) > 0:
            susceptible_plot.set_offsets(susceptible)
        else:
            susceptible_plot.set_offsets(np.empty((0, 2)))
            
        if len(infected) > 0:
            infected_plot.set_offsets(infected)
            
            # Adjust point sizes based on cluster membership
            sizes = np.ones(len(infected)) * 30
            
            # Get cluster information
            cluster_data = model.get_cluster_data()
            
            # Enhance the size of points in larger clusters
            for cluster in cluster_data:
                if cluster['size'] > 5:
                    # Find the points in this cluster
                    for pos in cluster['positions']:
                        # Find matching positions in infected
                        matches = np.all(infected == pos, axis=1)
                        # Update sizes
                        sizes[matches] = 30 + min(cluster['size'] * 2, 100)
            
            infected_plot.set_sizes(sizes)
        else:
            infected_plot.set_offsets(np.empty((0, 2)))
        
        if len(recovered) > 0:
            recovered_plot.set_offsets(recovered)
        else:
            recovered_plot.set_offsets(np.empty((0, 2)))
        
        # Update title
        title.set_text(f'Day {model.current_day}: {model.stats["infected"][-1]} infected, '
                      f'{model.stats["susceptible"][-1]} susceptible, '
                      f'{model.stats["recovered"][-1]} recovered, '
                      f'{model.stats["clusters"][-1]} clusters')
        
        # Update statistics plot
        days = np.array(model.stats['days'])
        infected_line.set_data(days, model.stats['infected'])
        susceptible_line.set_data(days, model.stats['susceptible'])
        recovered_line.set_data(days, model.stats['recovered'])
        clusters_line.set_data(days, model.stats['clusters'])
        
        stats_ax.relim()
        stats_ax.autoscale_view()
        
        return susceptible_plot, infected_plot, recovered_plot, title
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=days+1, interval=200, blit=False)
    
    plt.tight_layout()
    return fig, ax, ani

def run_network_simulation():
    """Run a simulation using the network-based SIR model"""
    # Create a network
    print("Creating social network...")
    network = NetworkGenerator.generate_barabasi_albert(num_nodes=5000, avg_connections=3)

    # Initialize the SIR model
    sir_model = SIRModel(network, infection_prob=0.3, recovery_days=14)

    # Seed initial infections
    sir_model.seed_infection(num_patients=5)

    # Run the simulation
    results = sir_model.run_simulation(days=100)

    # Save the results
    sir_model.save_results(filepath="data/sir_results.json")

    # Plot the results
    fig = sir_model.plot_results()
    plt.show()

# Call the function to run the simulation
run_network_simulation()
