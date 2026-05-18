import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from src.db import get_telemetry_raw

def predict_maintenance(device_id: str, key: str, hours_history: int = 168, forecast_hours: int = 24):
    data = get_telemetry_raw(device_id, key, hours_history)
    
    if len(data) < 20:
        return {"error": "Not enough data for prediction", "min_required": 20, "got": len(data)}
    
    values = [d["value"] for d in data if d["value"] is not None]
    timestamps = [d["ts"] for d in data if d["value"] is not None]
    
    df = pd.DataFrame({"ts": timestamps, "value": values})
    df["ts_norm"] = (df["ts"] - df["ts"].min()) / (df["ts"].max() - df["ts"].min())
    
    X = df["ts_norm"].values.reshape(-1, 1)
    y = df["value"].values
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    last_ts = timestamps[-1]
    future_ts = [last_ts + (i * 3600 * 1000) for i in range(1, forecast_hours + 1)]
    ts_range = timestamps[-1] - timestamps[0]
    future_norm = [(ts - timestamps[0]) / ts_range for ts in future_ts]
    
    X_future = np.array(future_norm).reshape(-1, 1)
    X_future_poly = poly.transform(X_future)
    predictions = model.predict(X_future_poly)
    
    current_val = values[-1]
    mean_val = np.mean(values)
    std_val = np.std(values)
    trend = float(np.polyfit(range(len(values)), values, 1)[0])
    
    health_score = 100.0
    if abs(trend) > std_val * 0.1:
        health_score -= 20
    if current_val > mean_val + 2 * std_val:
        health_score -= 30
    if current_val < mean_val - 2 * std_val:
        health_score -= 30
    health_score = max(0, min(100, health_score))
    
    risk_level = "LOW" if health_score > 70 else "MEDIUM" if health_score > 40 else "HIGH"
    
    return {
        "device_id": device_id,
        "key": key,
        "current_value": float(current_val),
        "trend": trend,
        "trend_direction": "INCREASING" if trend > 0 else "DECREASING" if trend < 0 else "STABLE",
        "health_score": float(health_score),
        "risk_level": risk_level,
        "forecast": [
            {"ts": int(ts), "predicted_value": float(val)}
            for ts, val in zip(future_ts[:12], predictions[:12])
        ],
        "recommendation": get_recommendation(risk_level, trend, current_val, mean_val)
    }

def get_recommendation(risk, trend, current, mean):
    if risk == "HIGH":
        return "IMMEDIATE INSPECTION REQUIRED - Value significantly outside normal range"
    elif risk == "MEDIUM":
        return f"Schedule maintenance - {'Upward' if trend > 0 else 'Downward'} trend detected"
    else:
        return "Device operating normally - Continue regular monitoring"
