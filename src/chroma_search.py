import chromadb
import numpy as np
import ollama
import time
import psutil
import os
import csv
from datetime import datetime


# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "document_embeddings"

# all-minilm
# mxbai-embed-large
# nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    try:
        # Get the collection
        collection = chroma_client.get_collection(name=collection_name)

        # Create embedding for the query
        query_embedding = get_embedding(query)

        # Search for similar documents
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Transform results into the expected format
        top_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            top_results.append({
                "file": metadata["file"],
                "page": metadata["page"],
                "chunk": results["documents"][0][i][:50] + "...",  # Preview of the text
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            })

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Similarity: {result['similarity']:.4f}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}: {result.get('chunk', '')}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
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

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    return memory.rss / (1024 * 1024)  # in MB


def interactive_search():
    """Interactive search interface."""
    print("ChromaDB RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break
        start_time = time.time()
        start_memory = get_memory_usage_mb()
        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)
        end_time = time.time()
        end_memory = get_memory_usage_mb()

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory

        print("\n--- Response ---")
        print(response)

        print(f"\nExecution Time: {execution_time} seconds")
        print(f"Memory Usage: {memory_used} MB")


if __name__ == "__main__":
    interactive_search()