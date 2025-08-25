import os
import json
import re
import pdfplumber
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# ✅ Correct initialization for new OpenAI SDK
client = OpenAI()  # Will automatically read from env var OPENAI_API_KEY

app = FastAPI(title="Billing PDF → Merged Schema Extractor")


# ✅ Removed RowNumber
class UnifiedRow(BaseModel):
    Name: Optional[str] = None
    MemberID: Optional[str] = None
    T1023AuthId: Optional[str] = None
    T1023Range: Optional[str] = None
    T1023BillDate: Optional[str] = None
    H0044AuthId: Optional[str] = None
    H0044Range: Optional[str] = None
    H0044BillDate: Optional[str] = None
    Paid: Optional[str] = None


@app.get("/")
async def root():
    return {"status": True, "message": "Billing PDF API is running 🚀"}


@app.post("/extract")
async def extract_merged(file: UploadFile = File(...)):
    """
    Upload any billing PDF (OHANA / ALOHA / HMSA).
    Returns: {"status": true/false, "data": [...]}
    """

    if not file.filename.lower().endswith(".pdf"):
        return {"status": False, "data": [], "error": "Please upload a PDF"}

    # 1) Extract text per page
    try:
        all_text_lines: List[str] = []
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    lines = [
                        re.sub(r"\s+", " ", ln).strip()
                        for ln in t.split("\n")
                        if ln.strip()
                    ]
                    all_text_lines.extend(lines)
    except Exception as e:
        return {"status": False, "data": [], "error": f"PDF read error: {e}"}

    if not all_text_lines:
        return {"status": False, "data": []}

    # 2) Heuristic: keep only the main table rows
    filtered_lines: List[str] = []
    stop_markers = (
        "AP'S OVERDUE",
        "AP'S DUE",
        "OVERDUE AP",
        "DUE CM",
        "SEPTEMBER",
        "ALL INTAKE",
        "NEEDS H0044",
    )
    for ln in all_text_lines:
        if any(m in ln.upper() for m in stop_markers):
            break
        if ("NAME" in ln.upper()) and (("MRN" in ln.upper()) or ("MBR" in ln.upper())):
            filtered_lines.append(ln)
            continue
        if re.match(r"^\d+\s", ln) or (
            "," in ln and not ln.upper().startswith("AUGUST")
        ):
            filtered_lines.append(ln)

    if not filtered_lines:
        filtered_lines = all_text_lines

    # 3) Build AI prompt (RowNumber removed from schema)
    system_prompt = (
        "You are a precise information extraction engine for billing tables. "
        "You will receive text lines from a PDF (header + rows). "
        'Return ONLY valid JSON of the form: {"rows": [ ... ]} with NO extra commentary.'
    )

    user_instructions = f"""
We have billing tables from different insurers with slightly different headers.
Unify each row into this MERGED SCHEMA (use strings; use null if missing):

- Name
- MemberID                (from MRN#, MRN, or MBR ID #)
- T1023AuthId             (from 'T1023 AUTH DATE(S)' column; values can be alphanumeric IDs)
- T1023Range              (from 'DATE RANGE' or 'B/D RANGE' that corresponds to T1023)
- T1023BillDate           (from 'BILLED' or 'BILL DATE' corresponding to T1023)
- H0044AuthId             (from 'H0044 AUTH DATE(S)')
- H0044Range              (from 'DATE RANGE' or 'B/D RANGE' corresponding to H0044)
- H0044BillDate           (from 'BILLED' or 'BILL DATE' corresponding to H0044)
- Paid

IMPORTANT RULES:
1) Pair RANGE/BILL columns correctly (T1023 vs H0044).
2) 'MemberID' comes from MRN#/MBR ID #.
3) Do not invent data. If a cell is blank, use null.
4) Keep date/range formats as found (e.g., '04/01-07/01').
5) Output strictly: {{"rows": [ UnifiedRow, ... ]}}.

LINES:
{json.dumps(filtered_lines, ensure_ascii=False)}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instructions},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        rows = data.get("rows", [])
    except Exception as e:
        return {"status": False, "data": [], "error": f"AI extraction failed: {e}"}

    normalized: List[UnifiedRow] = [UnifiedRow(**r) for r in rows]

    return {"status": True, "data": [row.dict() for row in normalized]}
