import joblib
import pandas as pd
import numpy as np
import os

def simulate(model_path='models/load_balancer_model.joblib',
             data_path='data/data.csv',
             out_csv='data/simulation_results.csv'):
    
    # ✅ Step 1: Check if model file exists
    assert os.path.exists(model_path), f"❌ Model file not found: {model_path}"
    
    # ✅ Step 2: Load the saved model, scaler, and features
    obj = joblib.load(model_path)
    model = obj['model']
    scaler = obj['scaler']
    features = obj['features']

    # ✅ Step 3: Load dataset
    df = pd.read_csv(data_path)
    print(f"✅ Loaded dataset with {len(df)} rows")

    # ✅ Step 4: Clean data
    # Convert all feature columns to numeric (invalid entries → NaN)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing or invalid data
    df = df.dropna(subset=features + ['Response_time(ms)'])
    print(f"✅ Cleaned dataset → {len(df)} valid rows remain")

    # ✅ Step 5: Scale features
    X = df[features]
    X_scaled = scaler.transform(X)

    # ✅ Step 6: Predict response times
    df['Predicted_Response(ms)'] = model.predict(X_scaled)

    # ✅ Step 7: Compare predicted vs actual
    df['Delta_ms'] = df['Predicted_Response(ms)'] - df['Response_time(ms)']

    # ✅ Step 8: Save results
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("\n✅ Simulation complete! Results saved to:", out_csv)
    print("\n🔹 Preview:")
    print(df[['time_of_the_day', 'Response_time(ms)', 'Predicted_Response(ms)', 'Delta_ms']].head())

if __name__ == "__main__":
    simulate()
