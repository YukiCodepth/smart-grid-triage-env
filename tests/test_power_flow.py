# tests/test_power_flow.py
import pytest
import networkx as nx
from env.graph_builder import build_grid_graph
from env.power_flow import solve_dc_power_flow

@pytest.fixture
def sample_scenario():
    return {
        "nodes": [
            {"id": "SO", "type": "substation", "power_injection_mw": 5.0, "voltage_kv": 11.0},
            {"id": "N1", "type": "residential", "power_injection_mw": -2.0, "voltage_kv": 11.0},
            {"id": "N2", "type": "residential", "power_injection_mw": -3.0, "voltage_kv": 11.0}
        ],
        "edges": [
            {"id": "E1", "source": "SO", "target": "N1", "thermal_limit_amps": 500, "reactance": 0.05, "breaker_state": "CLOSED"},
            {"id": "E2", "source": "SO", "target": "N2", "thermal_limit_amps": 500, "reactance": 0.05, "breaker_state": "CLOSED"}
        ]
    }

def test_graph_construction(sample_scenario):
    """Test that the topology builder creates the correct nodes and edges."""
    G = build_grid_graph(sample_scenario)
    assert len(G.nodes) == 3
    assert len(G.edges) == 2
    assert G.nodes["SO"]["power_injection_mw"] == 5.0

def test_power_flow_conservation(sample_scenario):
    """Test that the DC power flow solver calculates currents correctly without crashing."""
    G = build_grid_graph(sample_scenario)
    G = solve_dc_power_flow(G)
    
    # Check if currents were calculated and populated
    assert G["SO"]["N1"]["current_amps"] > 0.0
    assert G["SO"]["N2"]["current_amps"] > 0.0

def test_open_breaker_logic(sample_scenario):
    """Test that opening a breaker drops the current on that line to strictly 0.0."""
    sample_scenario["edges"][0]["breaker_state"] = "OPEN"
    G = build_grid_graph(sample_scenario)
    G = solve_dc_power_flow(G)
    
    # E1 should have 0 current, E2 should still have current
    assert G["SO"]["N1"]["current_amps"] == 0.0
    assert G["SO"]["N2"]["current_amps"] > 0.0
