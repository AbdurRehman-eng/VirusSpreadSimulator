import pygame
import numpy as np
import random
import math
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 30
BACKGROUND_COLOR = (10, 10, 30)
PARTICLE_SIZES = {"susceptible": 3, "infected": 5, "recovered": 4}
PARTICLE_COLORS = {
    "susceptible": (100, 200, 255, 180),  # Blue with alpha
    "infected": (255, 50, 50, 220),       # Red with alpha
    "recovered": (100, 255, 100, 180)     # Green with alpha
}

# Simulation parameters
NUM_NODES = 200
INITIAL_INFECTED = 5
INFECTION_RADIUS = 1.5
INFECTION_PROB = 0.02
RECOVERY_DAYS = 14
MOVEMENT_SPEED = 0.5
CLUSTER_FACTOR = 0.5  # How strongly nodes are attracted to population centers

class Node:
    def __init__(self, x, y, health="susceptible"):
        self.x = x
        self.y = y
        self.health = health
        self.days_infected = 0
        self.vel_x = random.uniform(-0.5, 0.5)
        self.vel_y = random.uniform(-0.5, 0.5)
        self.home_x = x
        self.home_y = y
        self.cluster_id = random.randint(0, 4)  # Assign to a random cluster
        
    def move(self, centers):
        # Random movement
        self.vel_x += random.uniform(-0.1, 0.1)
        self.vel_y += random.uniform(-0.1, 0.1)
        
        # Limit velocity
        max_vel = 1.0
        vel_magnitude = math.sqrt(self.vel_x**2 + self.vel_y**2)
        if vel_magnitude > max_vel:
            self.vel_x = (self.vel_x / vel_magnitude) * max_vel
            self.vel_y = (self.vel_y / vel_magnitude) * max_vel
        
        # Attraction to home/cluster
        center_x, center_y = centers[self.cluster_id]
        dx = center_x - self.x
        dy = center_y - self.y
        dist = max(1, math.sqrt(dx**2 + dy**2))
        
        # Stronger attraction when far from center
        attraction = min(1.0, dist / 200) * CLUSTER_FACTOR
        self.vel_x += (dx / dist) * attraction
        self.vel_y += (dy / dist) * attraction
        
        # Update position
        self.x += self.vel_x * MOVEMENT_SPEED
        self.y += self.vel_y * MOVEMENT_SPEED
        
        # Boundary check
        padding = 20
        if self.x < padding: 
            self.x = padding
            self.vel_x *= -0.5
        elif self.x > WIDTH - padding: 
            self.x = WIDTH - padding
            self.vel_x *= -0.5
        if self.y < padding: 
            self.y = padding
            self.vel_y *= -0.5
        elif self.y > HEIGHT - padding: 
            self.y = HEIGHT - padding
            self.vel_y *= -0.5

class Simulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("COVID-19 Spread Simulation - Islamabad")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.small_font = pygame.font.SysFont("Arial", 16)
        
        # Load background image
        self.bg_image = self.load_background(r"C:\Users\Ali\Desktop\DAA\VirusSpreadSimulator\src\islamabad.png")        
        
        # Create population centers (clusters)
        self.population_centers = [
            (400, 200),   # North area
            (700, 350),   # East area
            (300, 600),   # South-West area
            (800, 550),   # South-East area
            (550, 400),   # Central area
        ]
        
        # Initialize nodes
        self.nodes = []
        self.initialize_nodes()
        
        # Statistics
        self.day = 0
        self.stats = {
            "susceptible": [NUM_NODES - INITIAL_INFECTED],
            "infected": [INITIAL_INFECTED],
            "recovered": [0],
            "days": []
        }
        
        # Heatmap surface
        self.heatmap = None
        self.update_heatmap()
        
    def load_background(self, image_path):
        # Load the image and scale it to the screen size
        img = pygame.image.load(image_path)
        img = pygame.transform.scale(img, (WIDTH, HEIGHT))
        return img

    def initialize_nodes(self):
        for i in range(NUM_NODES):
            # Distribute nodes around population centers
            cluster = random.randint(0, len(self.population_centers) - 1)
            center_x, center_y = self.population_centers[cluster]
            
            # Random position with Gaussian distribution around center
            x = min(max(20, int(np.random.normal(center_x, 150))), WIDTH-20)
            y = min(max(20, int(np.random.normal(center_y, 150))), HEIGHT-20)
            
            node = Node(x, y)
            node.cluster_id = cluster
            self.nodes.append(node)
        
        # Initialize infected nodes
        for i in random.sample(range(NUM_NODES), INITIAL_INFECTED):
            self.nodes[i].health = "infected"
    
    def update_heatmap(self):
        # Create a heatmap of infections
        heatmap = np.zeros((HEIGHT, WIDTH))
        
        for node in self.nodes:
            if node.health == "infected":
                # Add heat around each infected node with Gaussian distribution
                x, y = int(node.x), int(node.y)
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    # Intensity based on days infected (more visible in early days)
                    intensity = max(0.5, 1 - (node.days_infected / RECOVERY_DAYS))
                    
                    # Use a smaller radius for the heatmap effect
                    radius = int(INFECTION_RADIUS * 2)
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                                dist = math.sqrt(dx**2 + dy**2)
                                if dist <= radius:
                                    # Gaussian falloff
                                    value = intensity * math.exp(-0.1 * dist**2)
                                    heatmap[ny, nx] += value
        
        # Normalize and create surface
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert heatmap to surface
        heatmap_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        # Custom colormap for infections (white/red gradient)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                val = heatmap[y, x]
                if val > 0.05:  # Threshold to only show significant areas
                    alpha = int(min(180, val * 220))
                    color = (255, int(255 * (1 - val)), int(255 * (1 - val)), alpha)
                    heatmap_surface.set_at((x, y), color)
        
        self.heatmap = heatmap_surface
    
    def spread_disease(self):
        self.day += 1
        new_infections = 0
        new_recoveries = 0
        
        # Update nodes and check for infections
        infected_nodes = [n for n in self.nodes if n.health == "infected"]
        susceptible_nodes = [n for n in self.nodes if n.health == "susceptible"]
        
        # Recovery phase
        for node in infected_nodes:
            node.days_infected += 1
            if node.days_infected >= RECOVERY_DAYS:
                node.health = "recovered"
                new_recoveries += 1
        
        # Infection phase - optimized to avoid checking all pairs
        for s_node in susceptible_nodes:
            for i_node in infected_nodes:
                dist = math.sqrt((s_node.x - i_node.x)**2 + (s_node.y - i_node.y)**2)
                if dist < INFECTION_RADIUS and random.random() < INFECTION_PROB:
                    s_node.health = "infected"
                    new_infections += 1
                    break  # Node already infected, no need to check other infected nodes
        
        # Update statistics
        self.stats["days"].append(self.day)
        self.stats["susceptible"].append(self.stats["susceptible"][-1] - new_infections)
        self.stats["infected"].append(self.stats["infected"][-1] + new_infections - new_recoveries)
        self.stats["recovered"].append(self.stats["recovered"][-1] + new_recoveries)
        
        # Update heatmap every few days to improve performance
        if self.day % 2 == 0:
            self.update_heatmap()
        
        return new_infections

    def run(self):
        running = True
        paused = False
        show_heatmap = True
        show_particles = True
        
        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_h:
                        show_heatmap = not show_heatmap
                    elif event.key == pygame.K_p:
                        show_particles = not show_particles
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Update simulation if not paused
            if not paused:
                # Move nodes
                for node in self.nodes:
                    node.move(self.population_centers)
                
                # Run disease spread
                new_infections = self.spread_disease()
                
                # Check if simulation is complete (no more infections)
                if self.stats["infected"][-1] == 0:
                    paused = True
            
            # Draw everything
            self.screen.blit(self.bg_image, (0, 0))
            
            # Draw heatmap
            if show_heatmap and self.heatmap:
                self.screen.blit(self.heatmap, (0, 0))
            
            # Draw population centers
            for i, (cx, cy) in enumerate(self.population_centers):
                pygame.draw.circle(self.screen, (60, 60, 90, 120), (cx, cy), 20, 2)
                name = f"District {i+1}"
                text = self.small_font.render(name, True, (180, 180, 200))
                self.screen.blit(text, (cx - text.get_width()//2, cy - 30))
            
            # Draw nodes
            if show_particles:
                for node in self.nodes:
                    size = PARTICLE_SIZES[node.health]
                    color = PARTICLE_COLORS[node.health]
                    pygame.draw.circle(self.screen, color, (int(node.x), int(node.y)), size)
            
            # Draw statistics
            stats_bg = pygame.Surface((300, 120), pygame.SRCALPHA)
            stats_bg.fill((10, 10, 30, 180))
            self.screen.blit(stats_bg, (10, 10))
            
            texts = [
                f"Day: {self.day}",
                f"Susceptible: {self.stats['susceptible'][-1]}",
                f"Infected: {self.stats['infected'][-1]}",
                f"Recovered: {self.stats['recovered'][-1]}",
            ]
            
            y_offset = 15
            for text in texts:
                text_surface = self.font.render(text, True, (220, 220, 220))
                self.screen.blit(text_surface, (20, y_offset))
                y_offset += 25
            
            # Draw controls
            controls_text = self.small_font.render(
                "Controls: SPACE - Pause | H - Toggle Heatmap | P - Toggle Particles | ESC - Quit", 
                True, (180, 180, 180)
            )
            self.screen.blit(controls_text, (10, HEIGHT - 30))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        
        # When simulation ends, generate a plot
        self.generate_stats_plot()
        
    def generate_stats_plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.stats["days"], self.stats["susceptible"], label="Susceptible", color="blue")
        plt.plot(self.stats["days"], self.stats["infected"], label="Infected", color="red")
        plt.plot(self.stats["days"], self.stats["recovered"], label="Recovered", color="green")
        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title(f"Disease Spread in Islamabad (N={NUM_NODES})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("disease_simulation_results.png")
        print("Statistics plot saved as disease_simulation_results.png")

if __name__ == "__main__":
    sim = Simulation()
    sim.run()