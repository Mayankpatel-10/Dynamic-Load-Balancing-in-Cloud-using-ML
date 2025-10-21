import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("🚀 Starting model training for Dynamic Load Balancer...")

# Data load karo
try:
    df = pd.read_csv('data/data.csv')
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Data check karo
print("\n📊 Data Info:")
print(df.info())
print("\n🔍 First 5 rows:")
print(df.head())

print("\n🎯 Original data columns:")
for col in df.columns:
    print(f"{col}: {df[col].dtype} | Unique values: {df[col].nunique()}")

# 🔥 DATA PREPROCESSING - Ye important hai!

# 1. Time column ko numeric mein convert karo
print("\n⏰ Converting time_of_the_day to numeric...")
def time_to_minutes(time_str):
    try:
        if ':' in str(time_str):
            hours, minutes = map(int, str(time_str).split(':'))
            return hours * 60 + minutes
        else:
            return int(time_str)
    except:
        return 0

df['time_minutes'] = df['time_of_the_day'].apply(time_to_minutes)

# 2. Day column ko numeric mein convert karo (Label Encoding)
print("📅 Converting Day to numeric...")
day_mapping = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
    'Friday': 5, 'Saturday': 6, 'Sunday': 7
}
df['day_numeric'] = df['Day'].map(day_mapping)

# 3. Disk_I/O column saaf karo - ye tumhara target hai
print("💾 Cleaning Disk_I/O column...")
def clean_disk_io(value):
    try:
        # Text values ko handle karo
        if str(value).strip().lower() == 'thirteen':
            return 13
        # Negative values ko positive mein convert karo
        value = float(value)
        return abs(value)  # Always positive
    except:
        return 0  # Default value

df['disk_io_clean'] = df['Disk_I/O'].apply(clean_disk_io)

# 4. Features select karo - sirf numeric columns use karo
feature_columns = [
    'CPU_Usage(%)', 
    'Memory_Usage(%)', 
    'Network_bandwidth_usage',
    'Number_of_active_request', 
    'Response_time(ms)',
    'time_minutes', 
    'day_numeric'
]

X = df[feature_columns]

# 5. Target variable - Disk I/O ko categories mein convert karo
# Since Disk_I/O has too many unique values, let's categorize them
print("🎯 Creating target categories from Disk_I/O...")
def categorize_disk_io(value):
    if value < 20:
        return "Low"
    elif value < 50:
        return "Medium" 
    elif value < 80:
        return "High"
    else:
        return "Very_High"

y = df['disk_io_clean'].apply(categorize_disk_io)

print(f"\n📈 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"Target value counts:")
print(y.value_counts())

# Final data check
print("\n✅ Final features:")
print(X.head())
print(f"\n✅ Final target distribution:")
print(y.value_counts())

# Train-test split karo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📚 Training set: {X_train.shape}")
print(f"🧪 Test set: {X_test.shape}")

# Model train karo
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

# Predictions karo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Model save karo
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✅ Created directory: {models_dir}")

model_path = os.path.join(models_dir, 'load_balancer_model.pkl')
joblib.dump(model, model_path)
print(f"💾 Model saved to: {model_path}")

# Feature importance dikhao
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Feature Importance:")
print(feature_importance)

# Test prediction karo
print("\n🧪 Testing with sample data...")
sample_input = [[
    50,    # CPU_Usage(%)
    60,    # Memory_Usage(%) 
    120,   # Network_bandwidth_usage
    180,   # Number_of_active_request
    250,   # Response_time(ms)
    300,   # time_minutes (5:00 AM)
    1      # day_numeric (Monday)
]]

sample_pred = model.predict(sample_input)
sample_prob = model.predict_proba(sample_input)

print(f"Sample Prediction: {sample_pred[0]}")
print(f"Sample Probabilities: {dict(zip(model.classes_, sample_prob[0]))}")

# Mapping save karo (frontend ke liye important)
mapping_info = {
    'feature_columns': feature_columns,
    'target_classes': model.classes_.tolist(),
    'day_mapping': day_mapping,
    'input_mapping': {
        'cpu_usage': 'CPU_Usage(%)',
        'memory_usage': 'Memory_Usage(%)', 
        'network_latency': 'Network_bandwidth_usage',
        'request_rate': 'Number_of_active_request'
    }
}

print(f"\n📋 Input Mapping for Frontend:")
print(mapping_info['input_mapping'])

print("\n✅ Model training completed successfully!")
print("🎉 Now you can run: python -m uvicorn serve_model:app --reload")