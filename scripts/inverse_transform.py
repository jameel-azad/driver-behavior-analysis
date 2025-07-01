import joblib
import numpy as np

def invert_cluster_centers(approx_centers, scaler_path='scripts/scaler.pkl'):
    scaler = joblib.load(scaler_path)
    return scaler.inverse_transform(approx_centers)
