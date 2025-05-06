from dataset.barbasi import generate_network
from models.SIR_model import SIRModel
import json

def main():
    #dataset generation
    social_network = generate_network(num_nodes=5000)
    
    #model initialization
    epidemic = SIRModel(social_network)
    epidemic.seed_infection(num_patients=25)  # 0.5% initial infection
    
    #simulating
    print("Starting disease spread simulation...")
    epidemic.run_simulation(days=100)
    
    #data storage
    with open("data/sir_results.json") as f:
        results = json.load(f)
        print(f"\nSimulation completed in {results['days']} days")
        print(f"Peak infections: {results['peak_infected']}")
        print(f"Final recovered: {results['final_recovered']}")

if __name__ == "__main__":
    main()