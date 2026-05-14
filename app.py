import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import logic from the local directory
from bureau_equifax_us import parse_bureau_json, summary

def make_json_serializable(d):
    """Recursively convert NumPy and Pandas types to standard Python types for JSON serialization."""
    if isinstance(d, dict):
        return {str(k): make_json_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_json_serializable(v) for v in d]
    elif isinstance(d, (np.integer, np.int64)):
        return int(d)
    elif isinstance(d, (np.floating, np.float64)):
        return float(d)
    elif isinstance(d, pd.Timestamp):
        return str(d.date()) if pd.notnull(d) else None
    elif isinstance(d, (datetime.date, datetime.datetime)):
        return str(d)
    else:
        return d

app = FastAPI(title="Equifax Bureau Dashboard")

# Mount the static directory for CSS/JS/HTML
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    content = await file.read()
    try:
        raw_json = json.loads(content.decode("utf-8"))
        
        # Parse logic
        parsed = parse_bureau_json(raw_json)
        result = summary(parsed)
        
        # Attach filename reference to the payload response
        result["identity"]["filename"] = file.filename
        
        return make_json_serializable(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = static_path / "index.html"
    return Response(
        content=index_file.read_text(),
        media_type="text/html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
