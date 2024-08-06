import os
import logging
import traceback
import sqlite3
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key not found. Make sure to set it in the environment variables.")
    raise RuntimeError("OpenAI API key not found. Make sure to set it in the environment variables.")
os.environ["OPENAI_API_KEY"] = api_key

# FastAPI app setup
app = FastAPI()

# CORS setup to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Define the path to the persisted ChromaDB
persist_directory = './chromadb_storage/db'
embedding_model = OpenAIEmbeddings()

# Create the folder if it doesn't exist
os.makedirs('chat_history_db', exist_ok=True)

# Database URL
DATABASE_URL = "sqlite:///./chat_history_db/conversations.db"

# Connect to the database
conn = sqlite3.connect('./chat_history_db/conversations.db', check_same_thread=False)
cursor = conn.cursor()

# Create conversations table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_query TEXT,
    answer TEXT,
    sources TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# Commit the changes
conn.commit()

class Query(BaseModel):
    question: str

# Function to load and split documents
def load_and_split_documents(directory):
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def initialize_chromadb(directory, persist_directory):
    try:
        logger.info(f"Loading and splitting documents from directory: {directory}")
        texts = load_and_split_documents(directory)
        logger.info(f"Loaded and split {len(texts)} documents")
        
        logger.info(f"Initializing ChromaDB with persist directory: {persist_directory}")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding_model,
                                         persist_directory=persist_directory)
        vectordb.persist()
        logger.info("ChromaDB initialized and persisted successfully")
        return vectordb
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to initialize ChromaDB")

def update_chromadb(texts, persist_directory, embedding_model):
    try:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        vectordb.add_documents(documents=texts)
        vectordb.persist()
        return vectordb
    except Exception as e:
        logger.error(f"Error updating ChromaDB: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to update ChromaDB")

# Check if the persist directory exists and initialize or load ChromaDB
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    vectordb = initialize_chromadb('./new_papers/', persist_directory)
else:
    try:
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embedding_model)
    except Exception as e:
        logger.error(f"Error loading ChromaDB: {e}")
        logger.error(traceback.format_exc())
        vectordb = initialize_chromadb('./new_papers/', persist_directory)

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        content = await file.read()
        file_path = f'./new_papers/{file.filename}'
        with open(file_path, 'wb') as f:
            f.write(content)

        texts = load_and_split_documents('./new_papers/')
        vectordb = update_chromadb(texts, persist_directory, embedding_model)
        return {"filename": file.filename, "status": "Processed"}
    except Exception as e:
        logger.error(f"Error uploading or processing PDF: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to upload or process PDF")
    
@app.post("/query/")
async def query_rag(query: Query):
    try:
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        response = qa_chain(query.question)

        cursor.execute("""
        INSERT INTO conversations (user_query, answer, sources)
        VALUES (?, ?, ?)
        """, (query.question, response['result'], ', '.join([doc.metadata['source'] for doc in response['source_documents']])))
        conn.commit()

        return {"answer": response['result'], "sources": [doc.metadata['source'] for doc in response['source_documents']]}
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to process the query")
    
@app.get("/conversations/", response_model=List[dict])
async def get_conversations():
    try:
        cursor.execute("SELECT * FROM conversations")
        rows = cursor.fetchall()
        conversations = [
            {"id": row[0], "user_query": row[1], "answer": row[2], "sources": row[3], "timestamp": row[4]}
            for row in rows
        ]
        return conversations
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int):
    try:
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        return {"message": "Conversation deleted"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to delete conversation")

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.exception_handler(Exception)
async def all_exception_handler(request, exc):
    logging.exception("An error occurred")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error. Please check the logs for more details."},
    )
