import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.db import get_telemetry_raw

def analyze_patterns(device_id: str, key: str, hours: int = 168, n_clusters: int = 3):
    data = get_telemetry_raw(device_id, key, hours)
    
    if len(data) < 20:
        return {"error": "Not enough data", "min_required": 20, "got": len(data)}
    
    values = [d["value"] for d in data if d["value"] is not None]
    timestamps = [d["ts"] for d in data if d["value"] is not None]
    
    hours_of_day = [(ts // 3600000) % 24 for ts in timestamps]
    days_of_week = [(ts // 86400000) % 7 for ts in timestamps]
    
    features = np.array([[v, h, d] for v, h, d in zip(values, hours_of_day, days_of_week)])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_clusters = min(n_clusters, len(values) // 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    clusters = []
    for i in range(n_clusters):
        cluster_vals = [values[j] for j in range(len(values)) if labels[j] == i]
        cluster_hours = [hours_of_day[j] for j in range(len(values)) if labels[j] == i]
        clusters.append({
            "cluster_id": i,
            "count": len(cluster_vals),
            "percentage": len(cluster_vals) / len(values) * 100,
            "avg_value": float(np.mean(cluster_vals)),
            "peak_hour": int(np.bincount(cluster_hours).argmax()),
            "label": classify_cluster(np.mean(cluster_vals), np.mean(values))
        })
    
    hourly_avg = {}
    for v, h in zip(values, hours_of_day):
        if h not in hourly_avg:
            hourly_avg[h] = []
        hourly_avg[h].append(v)
    
    hourly_pattern = [
        {"hour": h, "avg_value": float(np.mean(hourly_avg[h])), "count": len(hourly_avg[h])}
        for h in sorted(hourly_avg.keys())
    ]
    
    peak_hour = max(hourly_avg, key=lambda h: np.mean(hourly_avg[h]))
    low_hour = min(hourly_avg, key=lambda h: np.mean(hourly_avg[h]))
    
    return {
        "device_id": device_id,
        "key": key,
        "hours_analyzed": hours,
        "total_points": len(values),
        "clusters": clusters,
        "hourly_pattern": hourly_pattern,
        "insights": {
            "peak_hour": int(peak_hour),
            "low_hour": int(low_hour),
            "peak_value": float(np.mean(hourly_avg[peak_hour])),
            "variability": float(np.std(values) / np.mean(values) * 100) if np.mean(values) != 0 else 0
        }
    }

def classify_cluster(cluster_mean, overall_mean):
    ratio = cluster_mean / overall_mean if overall_mean != 0 else 1
    if ratio > 1.3:
        return "HIGH_USAGE"
    elif ratio < 0.7:
        return "LOW_USAGE"
    else:
        return "NORMAL_USAGE"
