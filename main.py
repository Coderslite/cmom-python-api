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

# ✅ Initialize OpenAI client once (GLOBAL)
client = OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="Billing PDF → Merged Schema Extractor")


# Data model
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
    Upload a billing PDF, extract structured rows, and normalize into UnifiedRow schema.
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    text = ""
    with pdfplumber.open(temp_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Prompt for schema extraction
    prompt = f"""
    You are a PDF billing data parser. 
    Extract rows from the following text and map into JSON array of this schema:

    {{
      "Name": string,
      "MemberID": string,
      "T1023AuthId": string,
      "T1023Range": string,
      "T1023BillDate": string,
      "H0044AuthId": string,
      "H0044Range": string,
      "H0044BillDate": string,
      "Paid": string
    }}

    Text:
    {text}
    """

    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a structured data extractor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    # Parse JSON
    try:
        content = response.choices[0].message.content.strip()
        rows = json.loads(re.search(r"\[.*\]", content, re.S).group())
    except Exception as e:
        return {
            "error": f"Failed to parse AI response: {str(e)}",
            "raw": response.choices[0].message.content,
        }

    return {"rows": [UnifiedRow(**row).dict() for row in rows]}
