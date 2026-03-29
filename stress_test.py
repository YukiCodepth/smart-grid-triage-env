import random
from env.grid_env import SmartGridTriageEnv
from env.models import GridAction

def run_stress_test(task_file: str):
    print(f"\n⚡ STRESS TESTING: {task_file}")
    env = SmartGridTriageEnv(f"scenarios/{task_file}")
    obs = env.reset()
    
    # We loop for 30 steps to ensure it correctly triggers the 'is_done' 
    # flag when it hits your scenario's max_timesteps (10, 15, or 20)
    for i in range(30):
        # 1. Generate a completely random valid action
        action_type = random.choice(["toggle_breaker", "shed_load", "noop"])
        target = None
        
        if action_type == "toggle_breaker" and obs.edges:
            target = random.choice(obs.edges).id
        elif action_type == "shed_load" and obs.nodes:
            target = random.choice(obs.nodes).id
            
        action = GridAction(action_type=action_type, target_id=target)
        
        # 2. Fire it at the environment
        try:
            obs, reward, is_done, info = env.step(action)
            print(f"Step {obs.timestep}: Executed {action.action_type} on {action.target_id} | Reward: {reward:.2f}")
            
            if is_done:
                print(f"✅ SUCCESS: Episode terminated cleanly at step {obs.timestep}.")
                return
                
        except Exception as e:
            print(f"❌ CRITICAL CRASH during step {obs.timestep}: {e}")
            return

    print("❌ ERROR: Environment never returned is_done=True.")

if __name__ == "__main__":
    tasks = [
        "easy_load_balance.yaml", 
        "medium_fault_isolate.yaml", 
        "hard_reroute.yaml"
    ]
    for task in tasks:
        run_stress_test(task)
