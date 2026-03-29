import numpy as np
import networkx as nx

def solve_dc_power_flow(G: nx.Graph, slack_bus: str = "SO") -> nx.Graph:
    """
    Solves the linearized DC power flow equations for the grid.
    Updates the 'current_amps' on all edges and 'powered' status on nodes.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return G

    node_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize Susceptance matrix B (n x n)
    B = np.zeros((n, n))
    
    # 1. Build the B matrix based on active topology
    for u, v, data in G.edges(data=True):
        if data.get("breaker_state") == "OPEN":
            continue # Open lines have 0 susceptance
            
        i, j = node_idx[u], node_idx[v]
        b = 1.0 / data.get("reactance", 0.1) # Susceptance = 1 / Reactance
        
        B[i][i] += b
        B[j][j] += b
        B[i][j] -= b
        B[j][i] -= b
        
    # 2. Build Power Injection Vector P
    P = np.array([G.nodes[node].get("power_injection_mw", 0.0) for node in nodes])
    
    # 3. Handle the Slack Bus (Reference Node)
    try:
        slack_i = node_idx[slack_bus]
    except KeyError:
        # If substation is missing entirely, grid is dead
        for u, v in G.edges(): G[u][v]["current_amps"] = 0.0
        for n in G.nodes(): G.nodes[n]["powered"] = False
        return G

    # Remove slack bus row and column to make matrix invertible
    B_reduced = np.delete(np.delete(B, slack_i, 0), slack_i, 1)
    P_reduced = np.delete(P, slack_i)
    
    # 4. Solve the linear system: theta = B^-1 * P
    try:
        theta_reduced = np.linalg.solve(B_reduced, P_reduced)
    except np.linalg.LinAlgError:
        # Singular matrix means part of the grid is disconnected (islanded).
        # We use the Moore-Penrose pseudo-inverse to solve the healthy parts 
        # of the grid while gracefully handling the isolated nodes.
        theta_reduced = np.linalg.pinv(B_reduced).dot(P_reduced)   
     
    # Reconstruct full angle array (slack bus angle is 0)
    theta = np.insert(theta_reduced, slack_i, 0.0)
    
    # 5. Compute Line Currents from phase angles
    for u, v, data in G.edges(data=True):
        if data.get("breaker_state") == "OPEN":
            G[u][v]["current_amps"] = 0.0
            continue
            
        i, j = node_idx[u], node_idx[v]
        b = 1.0 / data.get("reactance", 0.1)
        
        # Power flow (MW) = B_ij * (Theta_i - Theta_j)
        power_flow_mw = b * (theta[i] - theta[j])
        
        # Convert MW to Amperes
        voltage_kv = G.nodes[u].get("voltage_kv", 11.0)
        current_amps = abs((power_flow_mw * 1000.0) / voltage_kv) 
        
        G[u][v]["current_amps"] = current_amps

    return G
