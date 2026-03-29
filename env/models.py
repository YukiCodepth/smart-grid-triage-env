from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Optional

# --- SUB-COMPONENTS FOR OBSERVATION ---

class NodeState(BaseModel):
    id: str = Field(..., description="Unique identifier for the node (e.g., 'N1', 'SO')")
    type: Literal["substation", "hospital", "water_treatment", "residential", "commercial", "junction"]
    load_amps: float = Field(..., description="Current power draw in Amperes")
    voltage_kv: float = Field(..., description="Current voltage in kilovolts")
    priority_weight: int = Field(..., description="Importance of this node (1 to 10)")
    powered: bool = Field(..., description="True if currently receiving power")

class EdgeState(BaseModel):
    id: str = Field(..., description="Unique identifier for the transmission line (e.g., 'E1')")
    source: str = Field(..., description="ID of the starting node")
    target: str = Field(..., description="ID of the ending node")
    current_amps: float = Field(..., description="Actual current flowing through the line")
    thermal_limit_amps: float = Field(..., description="Maximum safe current before a trip occurs")
    breaker_state: Literal["OPEN", "CLOSED"] = Field(..., description="OPEN means no power flows. CLOSED means power flows.")

class Alarm(BaseModel):
    timestep: int
    severity: Literal["WARNING", "CRITICAL", "TRIPPED"]
    message: str

# --- THE REQUIRED OPENENV MODELS ---

class GridObservation(BaseModel):
    """The current state of the power grid."""
    timestep: int = Field(..., description="Current simulation step")
    nodes: List[NodeState] = Field(..., description="State of all consumers and generators")
    edges: List[EdgeState] = Field(..., description="State of all transmission lines and breakers")
    active_alarms: List[Alarm] = Field(..., description="Recent grid events or warnings")
    grid_status: Literal["STABLE", "WARNING", "FAULT", "BLACKOUT"]

class GridAction(BaseModel):
    """The command the agent wishes to execute."""
    action_type: Literal["toggle_breaker", "shed_load", "noop"] = Field(..., description="The operation to perform.")
    target_id: Optional[str] = Field(None, description="The ID of the breaker (Edge ID) or node to act upon. Null if noop.")

class GridReward(BaseModel):
    """The outcome of the agent's action."""
    score: float = Field(..., description="The current step reward (-1.0 to 1.0)")
    is_done: bool = Field(..., description="True if the task is complete or failed")
    info: Dict[str, str] = Field(default_factory=dict, description="Metadata for debugging")
