import json
import sqlite3
import shutil
import os
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).parent.resolve()
DB_PATH = ROOT / "db.sqlite"
UPLOAD_DIR = ROOT / "uploads"
MODELS_DIR = ROOT / "models"
ANALYSIS_DIR = ROOT / "analysis_outputs"
BEST_MODEL_TXT = ROOT / "best_model_path.txt"

# ensure dirs exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

app = FastAPI(title="Cycle Alarm Server (with ML)")

# --------- CORS (cho phép Flutter app truy cập) ---------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production -> lock down domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount static folder for analysis outputs (plots)
app.mount("/analysis_outputs", StaticFiles(directory=str(ANALYSIS_DIR)), name="analysis_outputs")

# --------- DB setup (survey) ---------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS surveys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            age_group TEXT,
            answers_json TEXT,
            created_at TEXT
        )
    """)
    con.commit()
    con.close()

init_db()

# --------- ML model (loaded at startup) ---------
model = None
best_model_path: Optional[str] = None

def try_load_model_from_path(path: str):
    """
    Try to load model using helper in TrainCNN6lop (if available),
    otherwise fallback to tf.keras.models.load_model.
    Returns loaded model or raises Exception.
    """
    # import locally to avoid heavy import at module import time
    try:
        # prefer custom loader if available
        from TrainCNN6lop import load_trained_model_for_inference
        m = load_trained_model_for_inference(path)
        print(f"✅ Model loaded via TrainCNN6lop.load_trained_model_for_inference: {path}")
        return m
    except Exception as e:
        print("⚠️ load_trained_model_for_inference not available or failed:", e)
        try:
            import tensorflow as tf
            m = tf.keras.models.load_model(path, compile=False)
            print(f"✅ Model loaded via tf.keras.models.load_model: {path}")
            return m
        except Exception as e2:
            print("❌ Fallback tf.keras.models.load_model failed:", e2)
            raise

def load_model_at_startup():
    global model, best_model_path
    try:
        if BEST_MODEL_TXT.exists():
            txt = BEST_MODEL_TXT.read_text().strip()
            if txt:
                cand = Path(txt)
                if not cand.exists():
                    # try models dir
                    cand2 = MODELS_DIR / cand.name
                    if cand2.exists():
                        best_model_path = str(cand2)
                    else:
                        print(f"Model path in best_model_path.txt not found: {txt}")
                        best_model_path = None
                else:
                    best_model_path = str(cand)
        if best_model_path:
            model = try_load_model_from_path(best_model_path)
        else:
            print("No best_model_path configured or model file missing. Model not loaded.")
    except Exception as e:
        print("Error loading model at startup:", e)
        model = None

# Load at import time
load_model_at_startup()

# --------- Helper: fallback heuristic alarms (if no model) ----------
import datetime as _dt
def heuristic_alarms(bed_time_str: str, wake_time_str: str):
    try:
        bed = _dt.datetime.strptime(bed_time_str, "%H:%M")
        wake = _dt.datetime.strptime(wake_time_str, "%H:%M")
    except Exception:
        bed = _dt.datetime.combine(_dt.date.today(), _dt.time(hour=22, minute=0))
        wake = bed + _dt.timedelta(hours=8)
    if wake <= bed:
        wake += _dt.timedelta(days=1)
    t = bed + _dt.timedelta(minutes=15)  # assume 15 min to fall asleep
    cycle = _dt.timedelta(minutes=90)
    out = []
    while t + _dt.timedelta(minutes=30) < wake and len(out) < 6:
        out.append(t.strftime("%H:%M"))
        t += cycle
    if not out:
        out = [wake.strftime("%H:%M")]
    # convert to 12h format with AM/PM for UI
    alarms = []
    for s in out:
        try:
            dt = _dt.datetime.strptime(s, "%H:%M")
            alarms.append({"time": dt.strftime("%I:%M %p").lstrip("0"), "description": "Heuristic (no model)"})
        except:
            alarms.append({"time": s, "description": "Heuristic (no model)"})
    return alarms

# --------- API Models ---------
class CalcRequest(BaseModel):
    filename: str
    bed_time: str  # "22:00"
    wake_time: Optional[str] = None
    age: Optional[str] = "30"
    gender: Optional[str] = "nam"
    mode: Optional[str] = "1"  # '1' = wake in light stage, '2' = cycles

# --------- Endpoints: upload EDF, upload model, calculate ----------
@app.post("/upload")
async def upload_edf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=400, detail="Only .edf files allowed")
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "filename": file.filename, "path": str(dest)}

# Alias: nếu client (nhầm) post tới /upload_sound thì xử lý giống /upload
@app.post("/upload_sound")
async def upload_sound_alias(file: UploadFile = File(...)):
    return await upload_edf(file=file)

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload model file (.keras, .h5, .pb). Saves to server/models and writes best_model_path.txt.
    After upload, tries to load model into memory.
    """
    name = file.filename
    if not name.lower().endswith((".keras", ".h5", ".pb")):
        raise HTTPException(status_code=400, detail="Model must be .keras/.h5/.pb")
    dest = MODELS_DIR / name
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # update best_model_path.txt to relative models path
    BEST_MODEL_TXT.write_text(str(dest))
    # try to load model immediately
    global model, best_model_path
    try:
        model = try_load_model_from_path(str(dest))
        best_model_path = str(dest)
        return {"ok": True, "model_path": str(dest)}
    except Exception as e:
        # keep model None but save path
        best_model_path = str(dest)
        return {"ok": False, "model_path": str(dest), "error": str(e)}

@app.post("/calculate")
async def calculate(req: CalcRequest):
    """
    Run analysis on a previously uploaded EDF file.
    Returns JSON with alarms and additional info (stage_counts, score, reports).
    """
    # check file exists
    edf_path = UPLOAD_DIR / req.filename
    if not edf_path.exists():
        raise HTTPException(status_code=404, detail="EDF file not found on server. Upload first.")

    # if model not loaded and no best_model_path, fallback to heuristic
    if model is None and (not BEST_MODEL_TXT.exists() or BEST_MODEL_TXT.read_text().strip() == ""):
        alarms = heuristic_alarms(req.bed_time, req.wake_time or "06:30")
        return {"alarms": alarms, "note": "No model loaded; returned heuristic alarms."}

    # Import analyze function (expects analyze_edf inside analyze_sleep_CNN.py)
    try:
        from analyze_sleep_CNN import analyze_edf
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze_sleep_CNN.py is missing or failed import: {e}")

    # call analyze_edf passing loaded model or model_path
    try:
        res = analyze_edf(
            edf_path=str(edf_path),
            model=model,
            model_path=best_model_path if best_model_path else (BEST_MODEL_TXT.read_text().strip() if BEST_MODEL_TXT.exists() else None),
            bed_time_str=req.bed_time,
            wake_time_str=req.wake_time or "06:30",
            age=req.age or "30",
            gender=req.gender or "nam",
            mode=req.mode or "1",
            output_dir=str(ANALYSIS_DIR)
        )
    except Exception as e:
        # show full error detail for easier debugging (in dev)
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

    # If analyze_edf returned timeline_path relative to server/analysis_outputs, convert to URL
    timeline_url = None
    try:
        timeline_path = res.get("timeline_path")
        if timeline_path:
            # derive relative filename and expose through /analysis_outputs
            fname = Path(timeline_path).name
            timeline_url = f"/analysis_outputs/{fname}"
    except Exception:
        timeline_url = None

    return {
        "alarms": res.get("alarms", []),
        "stage_counts": res.get("stage_counts", {}),
        "sleep_score": res.get("sleep_score"),
        "sleep_rating": res.get("sleep_rating"),
        "reports": {
            "sleep_quality_table": res.get("sleep_quality_table", []),
            "stage_impact_report": res.get("stage_impact_report", [])
        },
        "timeline_url": timeline_url
    }

# --------- keep your survey endpoints ----------
@app.post("/submit_survey")
async def submit_survey(request: Request):
    data = await request.json()
    gender = data.get("gender")
    age = data.get("age")
    group = data.get("group")
    answers = data.get("answers")

    if not all([gender, age, group, answers]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO surveys (gender, age, age_group, answers_json, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        gender,
        age,
        group,
        json.dumps(answers),
        _dt.datetime.utcnow().isoformat()
    ))
    con.commit()
    con.close()
    return {"ok": True, "message": "Survey saved successfully"}

@app.get("/surveys")
def get_all_surveys():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, gender, age, age_group, answers_json, created_at FROM surveys ORDER BY id DESC")
    rows = cur.fetchall()
    con.close()

    result = [
        {
            "id": r[0],
            "gender": r[1],
            "age": r[2],
            "group": r[3],
            "answers": json.loads(r[4]),
            "created_at": r[5],
        }
        for r in rows
    ]
    return {"ok": True, "surveys": result}

@app.get("/health")
def health():
    return {"ok": True, "message": "Server running and database ready.", "model_loaded": model is not None, "model_path": best_model_path}

# --------- run server ----------
if __name__ == "__main__":
    # use uvicorn to run the app
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
