from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Gemini
# genai.configure(
#     api_key=os.getenv("GEMINI_API_KEY")
# )

# model = genai.GenerativeModel(
#     "gemini-2.5-flash"
# )

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load embedding model ONLY ONCE
embedding_model = SentenceTransformer(
    'paraphrase-MiniLM-L3-v2'
)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_chunks"
)

# Health check
@app.get("/")
def health():
    return {
        "status": "running"
    }

# Utility functions
def get_embedding_model():
    return embedding_model

def create_chunks(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks

def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    return text

def ask_llm(prompt):

    # response = model.generate_content(prompt)
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
    )

    return response.text

# Upload API
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    try:
        os.makedirs("uploads", exist_ok=True)

        upload_path = f"uploads/{file.filename}"

        # Save uploaded PDF
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read PDF
        reader = PdfReader(upload_path)

        extracted_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:
                extracted_text += text

        # Clean text
        extracted_text = clean_text(extracted_text)

        # Validate extracted text
        if not extracted_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF"
            )

        # Chunking
        chunks = create_chunks(extracted_text)

        # Embeddings
        embedding_model = get_embedding_model()

        embeddings = embedding_model.encode(chunks)

        # Store in ChromaDB
        for index, chunk in enumerate(chunks):

            collection.add(
                ids=[f"{file.filename}_{index}"],
                documents=[chunk],
                embeddings=[embeddings[index].tolist()]
            )

        return {
            "message": "Embeddings stored successfully",
            "total_chunks": len(chunks)
        }

    except HTTPException as http_error:
        print("UPLOAD HTTP ERROR:", str(http_error.detail))
        raise http_error

    except Exception as e:
        print("UPLOAD ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Upload Error: {str(e)}"
        )

# Ask API
@app.post("/ask")
async def ask_question(question: str):

    try:
        print("Received question:", question)

        embedding_model = get_embedding_model()

        # Generate question embedding
        question_embedding = embedding_model.encode(question)

        print("Question embedding generated")

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=3
        )

        print("Chroma query successful")

        retrieved_chunks = results['documents'][0]

        if not retrieved_chunks:

            raise HTTPException(
                status_code=404,
                detail="No relevant chunks found"
            )

        # Create context
        context = " ".join(retrieved_chunks)

        print("Context length:", len(context))

        # Prompt
        prompt = f"""
        Answer the question based on the context below.

        Context:
        {context}

        Question:
        {question}
        """

        print("Calling Gemini API...")

        # LLM response
        ai_response = ask_llm(prompt)

        print("Gemini response received")

        return {
            "question": question,
            "answer": ai_response
        }

    except HTTPException as http_error:
        print("HTTP ERROR:", str(http_error.detail))
        raise http_error

    except Exception as e:
        print("ASK API ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error: {str(e)}"
        )