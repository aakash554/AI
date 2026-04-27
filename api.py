import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from core_logic import (
    save_uploaded_file,
    ingest_pdf,
    generate_flashcards,
    generate_revision_planner,
    generate_grounded_quiz,
    get_vectorstore,
    TEMP_DIR
)

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str

@app.get("/")
def read_root():
    return {"message": "Exam Revision Assistant Backend is running!"}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    saved_paths = []
    
    try:
        for file in files:
            path = os.path.join(TEMP_DIR, file.filename)
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(path)
            
        total_chunks = 0
        for path in saved_paths:
            chunks = ingest_pdf(path)
            total_chunks += chunks
            
        return {"message": f"Successfully processed {len(files)} files", "chunks": total_chunks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def get_status():
    vs = get_vectorstore()
    return {"has_knowledge_base": vs is not None}

@app.post("/generate/flashcards")
def get_flashcards(req: TopicRequest):
    if not get_vectorstore():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Please upload PDFs first.")
    
    result = generate_flashcards(req.topic)
    return {"data": result}

@app.post("/generate/planner")
def get_planner(req: TopicRequest):
    if not get_vectorstore():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Please upload PDFs first.")
    
    result = generate_revision_planner(req.topic)
    return {"data": result}

@app.post("/generate/quiz")
def get_quiz(req: TopicRequest):
    if not get_vectorstore():
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Please upload PDFs first.")
    
    result = generate_grounded_quiz(req.topic)
    return {"data": result}

