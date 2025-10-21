from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from datetime import datetime

app = FastAPI(title="Dynamic Load Balancer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ServerData(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_latency: float
    request_rate: float

# Input data ka structure define karo (tumhare actual columns ke hisab se)
class LoadBalancerData(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_bandwidth: float
    active_requests: float
    response_time: float
    # Time aur day automatically add karenge

# Model load karo
try:
    model = joblib.load('models/load_balancer_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    model = None

# Helper functions
def get_current_time_minutes():
    now = datetime.now()
    return now.hour * 60 + now.minute

def get_current_day_numeric():
    # Monday=1, Sunday=7
    return datetime.now().isoweekday()

@app.get("/")
async def root():
    return {"message": "Dynamic Load Balancer ML Model running. Use /predict POST."}

@app.post("/predict")
async def predict(data: ServerData):
    try:
        if model is None:
            return {"error": "Model not loaded"}
        
        # Current time aur day add karo
        current_time = get_current_time_minutes()
        current_day = get_current_day_numeric()
        
        # Data prepare karo - tumhare actual features ke hisab se
        input_data = pd.DataFrame([[
            data.cpu_usage,                    # CPU_Usage(%)
            data.memory_usage,                 # Memory_Usage(%) 
            data.network_latency,              # Network_bandwidth_usage
            data.request_rate,                 # Number_of_active_request
            50,                               # Response_time(ms) - default
            current_time,                     # time_minutes
            current_day                       # day_numeric
        ]], columns=[
            'CPU_Usage(%)', 'Memory_Usage(%)', 'Network_bandwidth_usage',
            'Number_of_active_request', 'Response_time(ms)', 'time_minutes', 'day_numeric'
        ])
        
        # Prediction karo
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()
        
        return {
            "prediction": str(prediction),
            "confidence": round(float(probability), 4),
            "input_data": {
                "cpu_usage": data.cpu_usage,
                "memory_usage": data.memory_usage,
                "network_latency": data.network_latency,
                "request_rate": data.request_rate,
                "current_time": current_time,
                "current_day": current_day
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/dashboard")
async def get_dashboard():
    return FileResponse("static/index.html")

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    return FileResponse(f"static/{file_path}")