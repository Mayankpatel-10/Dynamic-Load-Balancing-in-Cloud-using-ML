import requests
url = "http://127.0.0.1:8000/predict"
payload = {
  "CPU_Usage": 65,
  "Memory_Usage": 60,
  "Network_bandwidth_usage": 120,
  "Number_of_active_request": 180,
  "Disk_IO": 40
}
r = requests.post(url, json=payload)
print(r.status_code, r.json())
