"""
FastAPI backend for the Hybrid Spam Detector.

Trains the real model on startup, then serves:
  GET  /                          -> the frontend
  POST /api/signup                -> create an account, starts a session
  POST /api/login                 -> {"username","password"} -> starts a session
  POST /api/logout                -> clears the session
  GET  /api/me                    -> current signed-in user, or 401
  POST /api/analyze               -> (auth required) analyze a message, saved to history
  GET  /api/history                -> (auth required) list this user's past analyses
  GET  /api/history/{id}           -> (auth required) full detail of one entry (incl. narrative)
  DELETE /api/history/{id}         -> (auth required) remove one entry
  GET  /api/history/export         -> (auth required) download full evidence log as JSON
  GET  /api/history/{id}/report    -> (auth required) download a single case as a PDF evidence report

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload
    open http://127.0.0.1:8000
"""

import re
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, Request, Response, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

import db
import auth
from spam_detector import HybridSpamDetector, load_data
from narrative import generate_narrative
from pdf_report import build_case_report_pdf
from qr_reader import decode_qr

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Hybrid Spam Detector API")

# --- DB + model init, once, at startup --------------------------------------
db.init_db()

print("Training hybrid spam detector...")
_df = load_data(str(BASE_DIR / "spam_raw.csv"))
model = HybridSpamDetector(tactic_weight=0.35)
model.fit(_df["text"], _df["y"])
print(f"Model ready. Trained on {len(_df)} messages.")

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.\-]{3,32}$")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)

    @field_validator("username")
    @classmethod
    def valid_username(cls, v):
        if not USERNAME_RE.match(v):
            raise ValueError("Username must be 3-32 characters: letters, numbers, _ . -")
        return v


def _set_session_cookie(response: Response, user_id: int):
    token = auth.create_session_token(user_id)
    response.set_cookie(
        key=auth.COOKIE_NAME,
        value=token,
        max_age=auth.MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
    )


@app.post("/api/signup")
def signup(creds: Credentials, response: Response):
    user_id = db.create_user(creds.username, creds.password)
    if user_id is None:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    _set_session_cookie(response, user_id)
    return {"id": user_id, "username": creds.username}


@app.post("/api/login")
def login(creds: Credentials, response: Response):
    user = db.authenticate(creds.username, creds.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    _set_session_cookie(response, user["id"])
    return {"id": user["id"], "username": user["username"]}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME)
    return {"ok": True}


@app.get("/api/me")
def me(user_id: int = Depends(auth.get_current_user_id)):
    user = db.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return {"id": user["id"], "username": user["username"]}


# ---------------------------------------------------------------------------
# Analysis (auth required — every analysis is saved to the user's evidence log)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, user_id: int = Depends(auth.get_current_user_id)):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text cannot be empty.")
    result = model.analyze(text)
    result["source"] = "text"
    entry_id = db.add_history_entry(user_id, result)
    result["id"] = entry_id
    return result


@app.post("/api/analyze-qr")
def analyze_qr(file: UploadFile = File(...), user_id: int = Depends(auth.get_current_user_id)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    try:
        contents = file.file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image file.")

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    decoded_text = decode_qr(contents)
    if not decoded_text:
        raise HTTPException(
            status_code=400,
            detail="No QR code could be detected in the uploaded image. Please ensure the QR code is clear, well-lit, and unobstructed.",
        )

    result = model.analyze(decoded_text)
    result["source"] = "qr"
    entry_id = db.add_history_entry(user_id, result)
    result["id"] = entry_id
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "trained_on": int(len(_df))}


# ---------------------------------------------------------------------------
# Evidence log (history)
# ---------------------------------------------------------------------------

@app.get("/api/history")
def history(user_id: int = Depends(auth.get_current_user_id)):
    return db.list_history(user_id)


@app.get("/api/history/export")
def history_export(user_id: int = Depends(auth.get_current_user_id)):
    user = db.get_user_by_id(user_id)
    entries = db.export_history(user_id)
    filename = f"evidence_log_{user['username']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    return JSONResponse(
        content={"username": user["username"], "exported_at": datetime.now(timezone.utc).isoformat(),
                 "entries": entries},
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/history/{entry_id}")
def history_detail(entry_id: int, user_id: int = Depends(auth.get_current_user_id)):
    entry = db.get_history_entry(user_id, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="No such evidence entry.")
    entry["narrative"] = generate_narrative(entry)
    return entry


@app.get("/api/history/{entry_id}/report")
def history_report(entry_id: int, user_id: int = Depends(auth.get_current_user_id)):
    entry = db.get_history_entry(user_id, entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="No such evidence entry.")
    entry["narrative"] = generate_narrative(entry)
    user = db.get_user_by_id(user_id)

    pdf_bytes = build_case_report_pdf(entry, user["username"])
    filename = f"evidence_report_case_{entry_id:06d}_{entry['verdict'].lower()}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.delete("/api/history/{entry_id}")
def history_delete(entry_id: int, user_id: int = Depends(auth.get_current_user_id)):
    deleted = db.delete_history_entry(user_id, entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such evidence entry.")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    html_file = (static_dir / "index.html") if (static_dir.exists() and (static_dir / "index.html").exists()) else (BASE_DIR / "index.html")
    return FileResponse(str(html_file))

