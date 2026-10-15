"""FastAPI interface for local email screening."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from enhanced_phishing_detector import UNIVERSITY_NAME, load_artifact, log_prediction, normalize_record, parse_email_bytes, predict_record

MAX_EML_SIZE = 10 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 64 * 1024
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
        chunks = []; size = 0
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk: break
            size += len(chunk)
            if size > MAX_EML_SIZE:
                raise HTTPException(status_code=413, detail="Email file must be 10 MB or smaller.")
            chunks.append(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="The uploaded email file is empty.")
        raw = b"".join(chunks)
        record = parse_email_bytes(raw)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not parse the uploaded email file.") from exc
    result=predict_record(record,artifact); log_prediction(result); return result
