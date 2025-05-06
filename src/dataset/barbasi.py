import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import os
from pathlib import Path

def generate_network(num_nodes=5000, avg_connections=3):
    """
    Generate scale-free network using Barabási-Albert model
    """
    # Getting the root directory of project
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Generating BA network with {num_nodes} nodes...")
    G = nx.barabasi_albert_graph(n=num_nodes, m=avg_connections)
    
    # Saving files
    gexf_path = data_dir / "social_network.gexf"
    stats_path = data_dir / "network_stats.json"
    
    nx.write_gexf(G, str(gexf_path))
    with open(stats_path, "w") as f:
        json.dump({
            "nodes": len(G.nodes()),
            "edges": len(G.edges()),
            "degree_histogram": nx.degree_histogram(G)[:20]
        }, f)
    
    print(f"Network generated. Saved to {gexf_path}")
    return G

if __name__ == "__main__":
    network = generate_network()
    print("Sample node attributes:", list(network.nodes(data=True))[:3])