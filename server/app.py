# app.py
from fastapi import FastAPI, HTTPException
from env.grid_env import SmartGridTriageEnv
from env.models import GridAction, GridObservation

app = FastAPI(title="SmartGridTriageEnv API")

# Initialize default task so the ping doesn't crash
current_env = SmartGridTriageEnv("scenarios/easy_load_balance.yaml")

@app.get("/")
def health_check():
    return {"status": 200, "message": "SmartGridTriageEnv is running"}

@app.post("/reset", response_model=GridObservation)
def reset_env():
    return current_env.reset()

@app.post("/step", response_model=dict)
def step_env(action: GridAction):
    try:
        obs, reward, is_done, info = current_env.step(action)
        return {"observation": obs.dict(), "reward": reward, "done": is_done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=GridObservation)
def get_state():
    return current_env.state()
