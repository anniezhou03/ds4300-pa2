import faiss
import ollama
import numpy as np
import os
import fitz
import json
import time
import psutil

VECTOR_DIM = 384
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = faiss.METRIC_INNER_PRODUCT


# Create a HNSW index in FAISS
def create_faiss_index():
    index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    index.is_trained = True
    return index



# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "all-minilm") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# store the embedding in FAISS
def store_embedding(index, chunk: str, embedding: list, metadata: list):
    embedding_arr = np.array(embedding, dtype=np.float32)
    index.add(embedding_arr.reshape(1, VECTOR_DIM))
    metadata.append(chunk)
    print(f"Stored embedding for: {chunk[:50]}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir, index):
    metadata = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:

                # print(f"--- Extracted text from Page {page_num + 1} of {file_name} ---")
                # print(text)
                # print("\n" + "-" * 80 + "\n")

                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(index, chunk, embedding, metadata)


            print(f" -----> Processed {file_name}")
    return metadata


def save_index(index, metadata, index_filename='faiss_index.index', metadata_filename='metadata.json'):
    faiss.write_index(index, index_filename)
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f)
    print(f'FAISS index and metadata saved to {index_filename}, {metadata_filename}')


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    return memory.rss / (1024 * 1024)

def main():
    # clear_redis_store()
    # create_hnsw_index()
    index = create_faiss_index()

    metadata = process_pdfs("../data/", index)
    print("\n---Done processing PDFs---\n")
    #query_redis("What is the capital of France?")
    #query_faiss(index, "What is the capital of France?")
    save_index(index, metadata)

if __name__ == "__main__":

    start_time = time.time()
    start_memory = get_memory_usage()

    main()

    end_time = time.time()
    end_memory = get_memory_usage()

    execution_time = end_time - start_time
    memory_used = abs(end_memory - start_memory)

    print(f"Execution Time: {execution_time} seconds")
    print(f"Memory Usage: {memory_used} MB")
