from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import os


load_dotenv()
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-3.1-flash-lite"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)

    return chunks

def clean_text(text):
    text = text.replace("\n", " ")

    text = text .replace("\t", " ")

    return text

def ask_llm(prompt):

    response = model.generate_content(prompt)

    return response.text

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    upload_path = f"uploads/{file.filename}"

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
# extracted_text
    reader = PdfReader(upload_path)
    extracted_text = ""

    for page in reader.pages:
        extracted_text+=page.extract_text()

# cleaning text
    extracted_text = clean_text(extracted_text)
# chunking text
    chunks = create_chunks(extracted_text)
#embedding
    embeddings = embedding_model.encode(chunks)

    for index, chunk in enumerate(chunks):
        collection.add(
            ids = [str(index)],
            documents = [chunk],
            embeddings = [embeddings[index].tolist()]
        )

    return {
    "message": "Embeddings stored successfully",
    "total_chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(question: str):

## retrieval
    question_embedding = embedding_model.encode(question)
    results = collection.query(
        query_embeddings = [question_embedding.tolist()],
        n_results = 3
    )

    retrieved_chunks  = results['documents'][0]

    context = "".join(retrieved_chunks)

    prompt = f""" 
    Answer the question based on the context below.

    Context: 
    {context}

    Question:
    {question}
    """
    # generation
    ai_response = ask_llm(prompt)

    return {
    "question": question,
    "answer": ai_response
}

embedding_model = SentenceTransformer(
    'paraphrase-MiniLM-L3-v2'
)

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name = "pdf_chunks"
)