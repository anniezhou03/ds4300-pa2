import faiss
import json
import numpy as np
import ollama
import time
import psutil
import os


VECTOR_DIM = 384
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = faiss.METRIC_INNER_PRODUCT
INDEX_FILE = "faiss_index.index"


def create_faiss_index(index_filename=INDEX_FILE):
    index = faiss.read_index(index_filename) #faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    #index.is_trained = True
    return index


def get_embedding(text: str, model: str = "all-minilm") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(index, query, k=3):
    embedding = get_embedding(query)
    embedding_arr = np.array(embedding, dtype=np.float32).reshape(1, VECTOR_DIM)

    distances, indices = index.search(embedding_arr, k)

    print(f"Query Results'{query}':")

    top_results = []
    for i in range(k):
        print(f"ID: {indices[0][i]}, Distance: {distances[0][i]}")
        top_results.append({
            "id": indices[0][i],
            "similarity": distances[0][i],
        })

    return top_results


def generate_rag_response(query, context_results, metadata):
    # Prepare context string with metadata
    context_str = "\n".join(
        [
            f"Result {result['id']} with similarity {float(result['similarity']):.2f}, Context: {metadata[result['id']]}"
            # Retrieve metadata for the chunk
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct the prompt for the RAG model
    prompt = f"""You are a helpful AI assistant.
        Use the following context to answer the query as accurately as possible. If the context is
        not relevant to the query, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    return memory.rss / (1024 * 1024)


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    faiss_index = create_faiss_index()

    with open("metadata.json", "r") as f:
        metadata = json.load(f)

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        start_time = time.time()
        start_memory = get_memory_usage()

        # Search for relevant embeddings
        context_results = search_embeddings(faiss_index, query)

        # Generate RAG response
        response = generate_rag_response(query, context_results, metadata)

        end_time = time.time()
        end_memory = get_memory_usage()
        execution_time = end_time - start_time
        memory_used = abs(end_memory - start_memory)

        print("\n--- Response ---")
        print(response)

        print(f"\nExecution Time: {execution_time} seconds")
        print(f"Memory Usage: {memory_used} MB")


if __name__ == "__main__":
    interactive_search()
