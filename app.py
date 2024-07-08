from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever
from haystack.pipelines import DocumentSearchPipeline
from haystack.schema import Document
import os

app = FastAPI()

# Read book content from a text file
book_path = os.path.join(os.path.dirname(__file__), 'book.txt')
with open(book_path, 'r', encoding='utf-8') as file:
    book_content = file.read()

# Split the book content into paragraphs
paragraphs = book_content.strip().split("\n\n")  # Assuming paragraphs are separated by double newlines

# Create Document objects for each paragraph
documents = [Document(content=para) for para in paragraphs]

# Initialize DocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Write documents to the DocumentStore
document_store.write_documents(documents)

# Initialize Retriever
retriever = BM25Retriever(document_store=document_store)

# Create Document Search Pipeline
pipeline = DocumentSearchPipeline(retriever)

class QueryModel(BaseModel):
    query: str

@app.post("/get_contexts/")
async def get_contexts(query: QueryModel):
    try:
        # Retrieve top 5 contexts
        results = pipeline.run(query=query.query, params={"Retriever": {"top_k": 5}})
        
        # Extract contexts
        contexts = [doc.content for doc in results['documents']]
        
        return {"contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
