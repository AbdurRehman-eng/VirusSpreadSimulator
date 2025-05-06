import json
import random
class SIRModel:
    def __init__(self, network):
        self.network = network
        self.initialization()
        self.day = 0
        self.infection_prob = 0.3
        self.recovery_days = 14
        self.stats = {
            'susceptible': [sum(1 for n in network.nodes() if network.nodes[n]['health'] == 'susceptible')],
            'infected': [0],
            'recovered': [0]
        }
    
    def initialization(self):
        # Initializing all nodes with health status
        for node in self.network.nodes():
            if 'health' not in self.network.nodes[node]:
                self.network.nodes[node]['health'] = 'susceptible'
            if 'days_infected' not in self.network.nodes[node]:
                self.network.nodes[node]['days_infected'] = 0

    def seed_infection(self, num_patients=5):
        """Randomly select initial infected nodes"""
        patients_zero = random.sample(list(self.network.nodes()), num_patients)
        for node in patients_zero:
            self.network.nodes[node]['health'] = 'infected'
        self.stats['infected'][0] = num_patients
        self.stats['susceptible'][0] -= num_patients
    
    def spread_disease(self):
        """Execute one day of disease spread"""
        self.day += 1
        new_infections = 0
        new_recoveries = 0
        
        # Infection Processing
        for node in self.network.nodes():
            if self.network.nodes[node]['health'] == 'infected':
                # Check recovery
                self.network.nodes[node]['days_infected'] += 1
                if self.network.nodes[node]['days_infected'] >= self.recovery_days:
                    self.network.nodes[node]['health'] = 'recovered'
                    new_recoveries += 1
                else:
                    # Infect neighbors
                    for neighbor in self.network.neighbors(node):
                        if (self.network.nodes[neighbor]['health'] == 'susceptible' and 
                            random.random() < self.infection_prob):
                            self.network.nodes[neighbor]['health'] = 'infected'
                            new_infections += 1
        
        # Update statistics
        self.stats['infected'].append(self.stats['infected'][-1] + new_infections - new_recoveries)
        self.stats['recovered'].append(self.stats['recovered'][-1] + new_recoveries)
        self.stats['susceptible'].append(self.stats['susceptible'][-1] - new_infections)
        
        return new_infections
    
    def run_simulation(self, days=100):
        """Run complete simulation"""
        for _ in range(days):
            infections = self.spread_disease()
            if infections == 0:  # Stop if no more spread
                break
        self.save_results()
    
    def save_results(self):
        """Save statistics for visualization team"""
        with open("data/sir_results.json", "w") as f:
            json.dump({
                "days": self.day,
                "stats": self.stats,
                "peak_infected": max(self.stats['infected']),
                "final_recovered": self.stats['recovered'][-1]
            }, f, indent=2)
    