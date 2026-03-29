# env/gnn_agent.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GridGNN(torch.nn.Module):
    """
    An advanced topology-aware agent architecture.
    Uses Graph Convolutional Networks (GCN) to process grid state.
    """
    def __init__(self, node_features=4, hidden_channels=32, num_actions=3):
        super(GridGNN, self).__init__()
        # Layer 1: Learn relationships between connected substations/houses
        self.conv1 = GCNConv(node_features, hidden_channels)
        # Layer 2: Deeper topology processing
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Output: Probability of taking specific actions
        self.lin = torch.nn.Linear(hidden_channels, num_actions)

    def forward(self, x, edge_index, batch=None):
        # 1. Node feature processing
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        
        # 2. Readout: Average the state of the entire grid
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        
        # 3. Action Head
        return F.softmax(self.lin(x), dim=1)

def obs_to_pyg(obs):
    """Converts our Pydantic Observation into a PyTorch Geometric Graph."""
    # Node features: [voltage, load, is_powered, priority]
    node_features = []
    for n in obs.nodes:
        node_features.append([
            n.voltage_kv / 11.0, 
            n.power_injection_mw / 10.0,
            1.0 if n.powered else 0.0,
            n.priority_weight / 10.0
        ])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge Index: How nodes are physically connected
    edges = []
    for e in obs.edges:
        # Simplified mapping for demonstration
        edges.append([0, 1]) # In a real implementation, map IDs to indices
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index
