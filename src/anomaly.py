import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from src.db import get_telemetry_raw

def detect_anomalies(device_id: str, key: str, hours: int = 24, contamination: float = 0.05):
    data = get_telemetry_raw(device_id, key, hours)
    
    values = [d["value"] for d in data if d["value"] is not None]
    timestamps = [d["ts"] for d in data if d["value"] is not None]
    
    if len(values) < 5:
        return {"error": "Not enough valid data points", "got": len(values), "min_required": 5}
    
    X = np.array(values).reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(X_scaled)
    scores = model.score_samples(X_scaled)
    
    anomalies = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        if pred == -1:
            anomalies.append({
                "ts": timestamps[i],
                "value": values[i],
                "anomaly_score": float(score),
                "severity": "HIGH" if score < -0.3 else "MEDIUM"
            })
    
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    
    z_scores = [(v - mean_val) / std_val if std_val > 0 else 0 for v in values]
    z_anomalies = [
        {"ts": timestamps[i], "value": values[i], "z_score": float(z)}
        for i, z in enumerate(z_scores) if abs(z) > 3
    ]
    
    return {
        "device_id": device_id,
        "key": key,
        "hours_analyzed": hours,
        "total_points": len(values),
        "anomaly_count": len(anomalies),
        "anomaly_rate": len(anomalies) / len(values),
        "stats": {"mean": mean_val, "std": std_val, "min": float(np.min(values)), "max": float(np.max(values))},
        "isolation_forest_anomalies": anomalies[:20],
        "z_score_anomalies": z_anomalies[:20],
        "status": "ALERT" if len(anomalies) > len(values) * 0.1 else "NORMAL"
    }
