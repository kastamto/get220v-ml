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
