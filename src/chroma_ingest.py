# DS 4300 Example - Modified for ChromaDB

import ollama
import chromadb
import numpy as np
import os
import fitz

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "document_embeddings"

VECTOR_DIM = 768


# Clear the existing collection if it exists
def clear_chroma_store():
    print("Clearing existing ChromaDB collection...")
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
    print("ChromaDB collection cleared.")


# Create a new collection in ChromaDB
def create_chroma_collection():
    collection = chroma_client.create_collection(name=collection_name)
    print("ChromaDB collection created successfully.")
    return collection


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in ChromaDB
def store_embedding(collection, file: str, page: str, chunk: str, embedding: list):
    # Create a unique ID for this chunk
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk) % 10000}"

    # Store in ChromaDB
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{
            "file": file,
            "page": page,
            "chunk_text": chunk
        }],
        documents=[chunk]
    )
    print(f"Stored embedding for: {doc_id}")


# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(collection, data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        collection=collection,
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


# Query ChromaDB for similar documents
def query_chroma(collection, query_text: str):
    embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    for i, doc_id in enumerate(results['ids'][0]):
        print(f"{doc_id} \n ----> {results['distances'][0][i]}\n")


def main():
    clear_chroma_store()
    collection = create_chroma_collection()

    process_pdfs(collection, "../data/")
    print("\n---Done processing PDFs---\n")
    query_chroma(collection, "What is the capital of France?")
    # At the end of main() function in ingest_chroma.py
    print("Available collections:", chroma_client.list_collections())


if __name__ == "__main__":
    main()