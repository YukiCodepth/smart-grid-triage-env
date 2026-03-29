import networkx as nx
from typing import Dict, Any

def build_grid_graph(scenario_config: Dict[str, Any]) -> nx.Graph:
    """
    Constructs a NetworkX graph representing the physical power grid.
    Nodes contain power injection/consumption data.
    Edges contain thermal limits, reactance, and breaker states.
    """
    G = nx.Graph()
    
    # 1. Add Nodes (Substations, Junctions, Consumers)
    for node in scenario_config.get("nodes", []):
        G.add_node(
            node["id"],
            type=node["type"],
            power_injection_mw=node.get("power_injection_mw", 0.0), # Positive for generation, negative for load
            voltage_kv=node.get("voltage_kv", 11.0),
            priority_weight=node.get("priority_weight", 1),
            powered=False # Default to false until power flow runs
        )
        
    # 2. Add Edges (Transmission Lines & Breakers)
    for edge in scenario_config.get("edges", []):
        G.add_edge(
            edge["source"],
            edge["target"],
            id=edge["id"],
            reactance=edge.get("reactance", 0.1), # Important for DC power flow math
            thermal_limit_amps=edge["thermal_limit_amps"],
            current_amps=0.0,
            breaker_state=edge.get("breaker_state", "CLOSED")
        )
        
    return G

def get_connected_components(G: nx.Graph, source_node: str = "SO") -> set:
    """
    Returns a set of all nodes physically connected to the main power source 
    via CLOSED breakers.
    """
    # Create a subgraph of only closed lines
    closed_edges = [
        (u, v) for u, v, d in G.edges(data=True) 
        if d.get("breaker_state") == "CLOSED"
    ]
    live_grid = G.edge_subgraph(closed_edges)
    
    if source_node in live_grid:
        return set(nx.node_connected_component(live_grid, source_node))
    return set()
