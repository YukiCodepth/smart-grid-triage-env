# inference.py
import os
import json
from openai import OpenAI
from env.grid_env import SmartGridTriageEnv
from env.models import GridAction

# ==========================================================
# 1. HACKATHON CONFIGURATION (Mandatory Variables)
# ==========================================================
# The judge's automated system will inject these variables.
# We default to standard OpenAI if they are missing.
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")

# The rules state participants must use HF_TOKEN or OpenAI Key
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are an automated Smart Grid Operator managing a distribution network.
Your goal is to maintain power to priority nodes while preventing thermal overloads.
Respond with EXACTLY ONE valid JSON object representing your action.
Valid action_types: "toggle_breaker", "shed_load", "noop".
Example: {"action_type": "toggle_breaker", "target_id": "E7"}
"""

def build_prompt(obs) -> str:
    """Formats the Pydantic observation into a prompt for the LLM."""
    return f"""
    Current Step: {obs.timestep}
    Grid Status: {obs.grid_status}
    
    Nodes (Consumers/Sources):
    {json.dumps([n.model_dump() for n in obs.nodes], indent=2)}
    
    Lines (Edges/Topology):
    {json.dumps([e.model_dump() for e in obs.edges], indent=2)}
    
    Active Alarms:
    {json.dumps([a.model_dump() for a in obs.active_alarms], indent=2)}
    
    What is your next action to optimize the grid? Output only raw JSON.
    """

def evaluate_task(task_file: str, client: OpenAI) -> float:
    """Runs a single scenario and returns the normalized score (0.0 - 1.0)."""
    print(f"\n--- 🚀 Evaluating Scenario: {task_file} ---")
    
    # Initialize the environment with the YAML scenario
    env = SmartGridTriageEnv(f"scenarios/{task_file}")
    obs = env.reset()
    
    total_reward = 0.0
    
    while True:
        try:
            # Call the LLM using the OpenAI client as mandated by rules
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(obs)}
                ],
                temperature=0.1,
                response_format={ "type": "json_object" }
            )
            
            # Parse the JSON response
            action_data = json.loads(response.choices[0].message.content)
            action = GridAction(**action_data)
            
        except Exception as e:
            # If the LLM fails or the API times out, we default to 'noop' 
            # to ensure the simulation continues without crashing.
            print(f"⚠️ Agent Logic Error: {e}. Defaulting to 'noop'.")
            action = GridAction(action_type="noop")

        print(f"Step {obs.timestep}: Agent Action -> {action.action_type} on {action.target_id}")
        
        # Advance the environment physics by one tick
        obs, reward, is_done, info = env.step(action)
        total_reward += reward
        
        if is_done:
            # Log the final telemetry provided by our custom logger
            print(f"📊 Final Telemetry: {json.dumps(info.get('telemetry', {}), indent=2)}")
            break
            
    # Calculate a normalized score (0.0 to 1.0) based on perfect performance
    # This is a benchmark requirement for the 3 tasks.
    max_possible = sum(n.priority_weight for n in obs.nodes) * env.max_timesteps
    normalized_score = max(0.0, min(1.0, total_reward / max_possible))
    
    print(f"✅ Task Complete. Final Normalized Score: {normalized_score:.2f}/1.0")
    return normalized_score

if __name__ == "__main__":
    if not API_KEY:
        print("❌ ERROR: Missing API Key! Set HF_TOKEN or OPENAI_API_KEY environment variables.")
        exit(1)
        
    # Initialize the client with the specified Base URL and Key
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Run the 3 mandatory tasks defined in our scenarios folder
    tasks = ["easy_load_balance.yaml", "medium_fault_isolate.yaml", "hard_reroute.yaml"]
    
    final_scores = []
    for task in tasks:
        score = evaluate_task(task, client)
        final_scores.append(score)
        
    print("\n" + "="*40)
    print(f"🏆 TOTAL BENCHMARK SCORE: {sum(final_scores)/len(final_scores):.2f}/1.0")
    print("="*40)
