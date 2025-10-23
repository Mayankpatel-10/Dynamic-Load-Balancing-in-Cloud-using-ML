# Dynamic-Load-Balancing-in-Cloud-using-ML
An intelligent cloud load balancer that leverages machine learning to optimize resource allocation, reduce latency, and improve system scalability. This project dynamically distributes workloads based on real-time performance metrics and predictive analysis, ensuring efficient cloud utilization.

STEP 1: Git Clone 
git clone https://github.com/Mayankpatel-10/Dynamic-Load-Balancing-in-Cloud-using-ML.git

STEP 2: Project Setup
cd "C:\Users\Mayank\Desktop\cloud project\Dynamic load balancer cloud"
.\venv\Scripts\Activate

STEP 3: Install Dependencies
pip install -r requirements.txt
pip install pandas numpy scikit-learn joblib fastapi uvicorn matplotlib xgboost requests

STEP 4: Data Check 
python check_data.py

STEP 5: Train Model
python train_model.py_

STEP 6: Server Start karo
python -m uvicorn serve_model:app --reload
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.

STEP 7: API Test karo
# new Terminal open kr (TERMINAL #2)
cd "C:\Users\Mayank\Desktop\cloud project\Dynamic load balancer cloud"
.\venv\Scripts\Activate

python -c "import requests; print(requests.get('http://127.0.0.1:8000/health').json())"

# Prediction test
python -c "
import requests
data = {
    'cpu_usage': 50,
    'memory_usage': 60, 
    'network_latency': 30,
    'request_rate': 100
}
response = requests.post('http://127.0.0.1:8000/predict', json=data)
print('Prediction:', response.json())
"

STEP 8: Frontend Check karo
Browser mein in URLs open karo:
API Status:  http://127.0.0.1:8000/
Swagger Docs:  http://127.0.0.1:8000/docs
Dashboard:  http://127.0.0.1:8000/dashboard
