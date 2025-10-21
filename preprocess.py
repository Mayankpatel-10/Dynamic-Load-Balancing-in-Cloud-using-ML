# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path='data/data.csv', test_size=0.2, random_state=42):
    df = pd.read_csv("data/data.csv")

# Clean textual numbers
    df.replace({
    ' one': 1, ' two': 2, ' three': 3, ' four': 4, ' five': 5,
    ' six': 6, ' seven': 7, ' eight': 8, ' nine': 9, ' ten': 10,
    ' eleven': 11, ' twelve': 12, ' thirteen': 13, ' fourteen': 14, ' fifteen': 15
    }, inplace=True)


    # Define features and target columns
    features = [
        'CPU_Usage(%)',
        'Memory_Usage(%)',
        'Network_bandwidth_usage',
        'Number_of_active_request',
        'Disk_I/O'
    ]
    target = 'Response_time(ms)'

    # Clean data: remove extra spaces, convert all to numeric
    for col in features + [target]:
        df[col] = df[col].astype(str).str.strip()              # remove spaces
        df[col] = pd.to_numeric(df[col], errors='coerce')      # convert words -> NaN

    # Drop rows where numeric conversion failed
    df = df.dropna(subset=features + [target])

    # Create input/output arrays
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, scaler, features
