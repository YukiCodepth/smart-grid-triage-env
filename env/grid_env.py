import yaml
import random
from typing import Tuple, Any, Dict
import networkx as nx
from .telemetry import GridTelemetry

# Import our custom physics and topology engines
from .graph_builder import build_grid_graph, get_connected_components
from .power_flow import solve_dc_power_flow

# Import the strict OpenEnv Pydantic models
from .models import (
    GridObservation, GridAction, NodeState, 
    EdgeState, Alarm
)

class SmartGridTriageEnv:
    """
    OpenEnv-compliant class for the Smart Grid Triage simulation.
    """
    def __init__(self, scenario_config_path: str):
        # Load the YAML scenario definition
        with open(scenario_config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
            
        self.max_timesteps = self.cfg.get("max_timesteps", 15)
        self.graph = None
        self.timestep = 0
        self.alarm_log = []
        self.last_toggle_count = 0
        self.telemetry = GridTelemetry()

    def reset(self) -> GridObservation:
        """
        OpenEnv API: Returns the environment to a clean starting state.
        """
        self.timestep = 0
        self.alarm_log = []
        self.last_toggle_count = 0
        
        # Rebuild fresh graph from config
        self.graph = build_grid_graph(self.cfg)
        
        # Run initial power flow to populate baseline currents
        self.graph = solve_dc_power_flow(self.graph)
        self._update_node_power_status()
        
        # Inject initial faults if defined in the scenario
        self._inject_scenario_faults()
        
        return self.state()

    def step(self, action: GridAction) -> Tuple[GridObservation, float, bool, Dict[str, Any]]:
        """
        OpenEnv API: Processes the LLM's action and advances physics by 1 tick.
        """
        self.last_toggle_count = 0
        self.timestep += 1

        # 1. Execute Action
        if action.action_type == "toggle_breaker" and action.target_id:
            self._toggle_breaker(action.target_id)
        elif action.action_type == "shed_load" and action.target_id:
            self._shed_load(action.target_id)

        # 2. Run Physics Engine
        self.graph = solve_dc_power_flow(self.graph)
        self._update_node_power_status()

        # 3. Stochastic Cascade Model (The "Real World" chaos)
        self._propagate_faults()

        # 4. Calculate Dense Reward
        reward = self._calculate_reward()

        # 5. Check Termination
        is_done = self.timestep >= self.max_timesteps
        
        # --- NEW TELEMETRY LOGGING ---
        self.telemetry.log_step(self.graph, self.last_toggle_count)
        info = {
            "status": "running" if not is_done else "finished",
            "telemetry": self.telemetry.get_report()
        }
        # -----------------------------
              
        return self.state(), reward, is_done, info

    def state(self) -> GridObservation:
        """
        OpenEnv API: Serializes the NetworkX graph into the Pydantic Observation model.
        """
        nodes = []
        for n_id, data in self.graph.nodes(data=True):
            nodes.append(NodeState(
                id=n_id,
                type=data.get("type", "junction"),
                load_amps=abs(data.get("power_injection_mw", 0.0) * 100), # simplified
                voltage_kv=data.get("voltage_kv", 11.0),
                priority_weight=data.get("priority_weight", 1),
                powered=data.get("powered", False)
            ))

        edges = []
        grid_status = "STABLE"
        for u, v, data in self.graph.edges(data=True):
            edges.append(EdgeState(
                id=data["id"],
                source=u,
                target=v,
                current_amps=data.get("current_amps", 0.0),
                thermal_limit_amps=data["thermal_limit_amps"],
                breaker_state=data["breaker_state"]
            ))
            # Determine overall grid status
            if data.get("current_amps", 0.0) > data["thermal_limit_amps"]:
                grid_status = "FAULT"

        return GridObservation(
            timestep=self.timestep,
            nodes=nodes,
            edges=edges,
            active_alarms=self.alarm_log[-5:], # Keep only the latest 5 alarms
            grid_status=grid_status
        )

    # --- INTERNAL ENGINEERING HELPERS ---

    def _toggle_breaker(self, edge_id: str):
        for u, v, data in self.graph.edges(data=True):
            if data["id"] == edge_id:
                # Flip the state
                new_state = "OPEN" if data["breaker_state"] == "CLOSED" else "CLOSED"
                self.graph[u][v]["breaker_state"] = new_state
                self.last_toggle_count += 1
                break

    def _shed_load(self, node_id: str):
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]["power_injection_mw"] = 0.0

    def _update_node_power_status(self):
        connected_to_source = get_connected_components(self.graph, "SO")
        for n_id in self.graph.nodes():
            self.graph.nodes[n_id]["powered"] = (n_id in connected_to_source)

    def _propagate_faults(self):
        """
        Research-grade feature: Lines running over capacity have a 30% chance of tripping.
        """
        for u, v, data in self.graph.edges(data=True):
            if data["breaker_state"] == "CLOSED" and data.get("current_amps", 0.0) > data["thermal_limit_amps"]:
                if random.random() < 0.30: # 30% cascade probability per tick [cite: 59]
                    self.graph[u][v]["breaker_state"] = "OPEN"
                    self.alarm_log.append(Alarm(
                        timestep=self.timestep,
                        severity="TRIPPED",
                        message=f"Line {data['id']} tripped due to thermal overload!"
                    ))

    def _calculate_reward(self) -> float:
        """
        Implements the math: R_t = sum(w_i * s_i) - (lambda * C) - (mu * V)
        """
        # 1. Weighted Uptime (Sum of priority weights of powered nodes) [cite: 108]
        uptime_score = sum(
            data["priority_weight"] for _, data in self.graph.nodes(data=True) 
            if data.get("powered", False)
        )
        
        # 2. Switching Penalty (Penalize erratic LLM behavior) [cite: 112]
        switch_penalty = self.last_toggle_count * 0.5 
        
        # 3. Thermal Violation Penalty (Hard safety signal) [cite: 115]
        thermal_violations = sum(
            1 for u, v, data in self.graph.edges(data=True)
            if data.get("current_amps", 0.0) > data.get("thermal_limit_amps", 9999)
        )
        violation_penalty = thermal_violations * 5.0 

        # Normalize logic can be added here, but returning raw float for dense signal
        return float(uptime_score - switch_penalty - violation_penalty)

    def _inject_scenario_faults(self):
        """Applies initial conditions based on the loaded YAML scenario."""
        fault_type = self.cfg.get("initial_fault")
        if fault_type == "overload_E1":
            # Artificially spike a load to force the agent to balance
            for n in self.graph.nodes():
                if self.graph.nodes[n].get("type") == "residential":
                    self.graph.nodes[n]["power_injection_mw"] *= 1.8 
        elif fault_type == "short_circuit_E3":
            # Force a line open
            self._toggle_breaker("E3")
            self.alarm_log.append(Alarm(
                timestep=0, severity="CRITICAL", message="Short circuit detected on E3."
            ))
