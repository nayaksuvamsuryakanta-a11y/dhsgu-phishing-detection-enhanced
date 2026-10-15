"""FastAPI interface for local email screening."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from enhanced_phishing_detector import UNIVERSITY_NAME, load_artifact, log_prediction, normalize_record, parse_email_bytes, predict_record

artifact = None
@asynccontextmanager
async def lifespan(app):
    global artifact
    artifact = load_artifact()
    yield

app = FastAPI(title="HGU Phishing Detection API", version="1.0", lifespan=lifespan)

@app.get("/")
def index():
    return {"name":"HGU University Phishing Detection API","university":UNIVERSITY_NAME,"endpoints":["GET /","POST /predict-json","POST /predict-eml"]}

@app.post("/predict-json")
def predict_json(payload: dict):
    result=predict_record(normalize_record(payload),artifact); log_prediction(result); return result

@app.post("/predict-eml")
async def predict_eml(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Email file must be 10 MB or smaller.")
        record = parse_email_bytes(raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded email file.") from exc
    result=predict_record(record,artifact); log_prediction(result); return result
