from fastapi import FastAPI, HTTPException, UploadFile, File
from qdrant_client import QdrantClient  
from qdrant_client.models import Distance, VectorParams, PointStruct
from PyPDF2 import PdfReader
import google.generativeai as gemini_client
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import List
import uuid
import re
import uvicorn
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('sentence-t5-base')

gemini_client.configure(api_key=os.getenv("GEMINI_API_KEY"))

qdrant_client = QdrantClient(url="http://localhost:6333", timeout=30)

collection_name = "sample"

try:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
except Exception as e:
    if "already exists" not in str(e):
        raise e

@app.post("/extract-pdf/")
async def extract_pdf(pdf_docs: List[UploadFile] = File(...)):
    # print(List[UploadFile])
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

        chunks = get_text_chunks(text)
        filenames = [pdf.filename for pdf in pdf_docs]
        new_filename=filenames[0]
        store_chunks_in_qdrant(chunks,new_filename)
        print(new_filename)


        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_text_chunks(text: str, chunk_size: int = 1000):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def store_chunks_in_qdrant(chunks: List[str],new_filename):
    if model and qdrant_client and collection_name:
        embeddings = model.encode(chunks)
        points = []

        for chunk, embed in zip(chunks, embeddings):
            doc_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embed,
                    payload={'data': chunk,"id":doc_id,"pdf_name":new_filename}
                )
            )

        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )

class SearchRequest(BaseModel):
    query: str

def generate_response(prompt: str) -> str:
    try:
        print("Generating response for prompt:", prompt)
        response = gemini_client.generate_text(prompt=prompt)
        
        # print("Response:", response)

        if response and hasattr(response, 'result'):
            generated_text = response.result
        else:
            generated_text = "No response text available"
        
        # print("Generated text:", generated_text)

        return generated_text
    except Exception as e:
        # print("Error generating response:", e)
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")



@app.post("/search/")
async def search_text(request: SearchRequest):
    try:
        print(f"Received search request: {request.query}")
        query_vector = model.encode([request.query])[0]

        search_result = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=3  
        )

        results = [{"id": result.id, "score": result.score, "data": result.payload["data"]}
                   for result in search_result]

        context = " ".join([result["data"] for result in results])
        # print(f"Context: {context}")

        prompt = f"""Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
                    the provided context just say,and provide the minimun 2 lines of contect "Answer is not available in the context", don't provide a wrong answer.\n\n
                    Context:\n{context}\n
                    Question:\n{request.query}\n
                """

        response = generate_response(prompt)
        # print(f"Generated response: {response}")

        return {"answer":response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)