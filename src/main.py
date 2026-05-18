import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from src.anomaly import detect_anomalies
from src.predictive import predict_maintenance
from src.patterns import analyze_patterns

load_dotenv()

app = FastAPI(title="Get220v ML Service", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "Get220v ML Service running", "version": "1.0.0"}

@app.get("/ml/anomaly/{device_id}/{key}")
def anomaly(device_id: str, key: str, hours: int = 24, contamination: float = 0.05):
    try:
        return detect_anomalies(device_id, key, hours, contamination)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/predict/{device_id}/{key}")
def predict(device_id: str, key: str, hours_history: int = 168, forecast_hours: int = 24):
    try:
        return predict_maintenance(device_id, key, hours_history, forecast_hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/patterns/{device_id}/{key}")
def patterns(device_id: str, key: str, hours: int = 168, n_clusters: int = 3):
    try:
        return analyze_patterns(device_id, key, hours, n_clusters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/full/{device_id}/{key}")
def full_analysis(device_id: str, key: str):
    results = {}
    try:
        results["anomaly"] = detect_anomalies(device_id, key, 24)
    except Exception as e:
        results["anomaly"] = {"error": str(e)}
    try:
        results["predictive"] = predict_maintenance(device_id, key, 168, 24)
    except Exception as e:
        results["predictive"] = {"error": str(e)}
    try:
        results["patterns"] = analyze_patterns(device_id, key, 168)
    except Exception as e:
        results["patterns"] = {"error": str(e)}
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 3003)))

import requests

@app.post("/ml/webhook/analyze")
async def webhook_analyze(payload: dict):
    device_id = payload.get("deviceId") or payload.get("originator", {}).get("id")
    key = payload.get("key", "temperature")
    
    if not device_id:
        return {"error": "deviceId required"}
    
    result = detect_anomalies(device_id, key, hours=24)
    
    if result.get("status") == "ALERT" or result.get("anomaly_count", 0) > 0:
        chat_api = os.getenv("CHAT_API_URL", "http://localhost:3001")
        try:
            requests.post(f"{chat_api}/api/notify/alarm", json={
                "type": f"ML Anomaly - {key}",
                "name": f"ML Anomaly - {key}",
                "originatorName": payload.get("deviceName", device_id),
                "severity": "CRITICAL" if result.get("anomaly_rate", 0) > 0.1 else "MAJOR",
                "status": "ACTIVE_UNACK",
                "mlDetails": result
            }, timeout=5)
            
            requests.post(f"{chat_api}/api/telegram/alert", json={
                "alarm": {
                    "type": f"ML Anomaly Detected - {key}",
                    "originatorName": payload.get("deviceName", device_id),
                    "severity": "CRITICAL" if result.get("anomaly_rate", 0) > 0.1 else "MAJOR",
                    "status": "ACTIVE",
                    "details": f"Anomalies: {result.get('anomaly_count', 0)}, Mean: {result.get('stats', {}).get('mean', 0):.1f}"
                }
            }, timeout=5)
        except Exception as e:
            print(f"Notification error: {e}")
    
    return result

@app.post("/ml/webhook/predict")
async def webhook_predict(payload: dict):
    device_id = payload.get("deviceId") or payload.get("originator", {}).get("id")
    key = payload.get("key", "temperature")
    device_name = payload.get("deviceName", device_id)
    
    if not device_id:
        return {"error": "deviceId required"}
    
    result = predict_maintenance(device_id, key, hours_history=168, forecast_hours=24)
    
    if "error" not in result and result.get("risk_level") in ["HIGH", "MEDIUM"]:
        chat_api = os.getenv("CHAT_API_URL", "http://localhost:3001")
        severity = "CRITICAL" if result.get("risk_level") == "HIGH" else "MAJOR"
        health = result.get("health_score", 100)
        trend = result.get("trend_direction", "STABLE")
        recommendation = result.get("recommendation", "")
        
        try:
            requests.post(f"{chat_api}/api/notify/alarm", json={
                "type": f"Predictive Alert - {key}",
                "name": f"Predictive Alert - {key}",
                "originatorName": device_name,
                "severity": severity,
                "status": "ACTIVE_UNACK"
            }, timeout=5)
            
            requests.post(f"{chat_api}/api/telegram/alert", json={
                "alarm": {
                    "type": f"Predictive Maintenance Alert",
                    "originatorName": device_name,
                    "severity": severity,
                    "status": "ACTIVE",
                    "details": f"Health: {health:.0f}% | Trend: {trend} | {recommendation}"
                }
            }, timeout=5)
        except Exception as e:
            print(f"Notification error: {e}")
    
    return result

@app.post("/ml/webhook/full")
async def webhook_full(payload: dict):
    device_id = payload.get("deviceId") or payload.get("originator", {}).get("id")
    key = payload.get("key", "temperature")
    device_name = payload.get("deviceName", device_id)
    
    if not device_id:
        return {"error": "deviceId required"}
    
    results = {}
    alerts = []
    
    try:
        anomaly = detect_anomalies(device_id, key, 24)
        results["anomaly"] = anomaly
        if anomaly.get("anomaly_count", 0) > 0:
            alerts.append(f"Anomalies: {anomaly['anomaly_count']} detected")
    except Exception as e:
        results["anomaly"] = {"error": str(e)}
    
    try:
        predictive = predict_maintenance(device_id, key, 168, 24)
        results["predictive"] = predictive
        if predictive.get("risk_level") in ["HIGH", "MEDIUM"]:
            alerts.append(f"Risk: {predictive['risk_level']} | Health: {predictive.get('health_score', 0):.0f}%")
    except Exception as e:
        results["predictive"] = {"error": str(e)}
    
    if alerts:
        chat_api = os.getenv("CHAT_API_URL", "http://localhost:3001")
        anomaly_count = results.get("anomaly", {}).get("anomaly_count", 0)
        risk = results.get("predictive", {}).get("risk_level", "LOW")
        severity = "CRITICAL" if anomaly_count > 2 or risk == "HIGH" else "MAJOR"
        
        try:
            requests.post(f"{chat_api}/api/telegram/alert", json={
                "alarm": {
                    "type": "ML Full Analysis Alert",
                    "originatorName": device_name,
                    "severity": severity,
                    "status": "ACTIVE",
                    "details": " | ".join(alerts)
                }
            }, timeout=5)
            
            requests.post(f"{chat_api}/api/notify/alarm", json={
                "type": "ML Full Analysis Alert",
                "name": "ML Full Analysis Alert",
                "originatorName": device_name,
                "severity": severity,
                "status": "ACTIVE_UNACK"
            }, timeout=5)
        except Exception as e:
            print(f"Notification error: {e}")
    
    results["device_id"] = device_id
    results["alerts_sent"] = len(alerts)
    return results
