# ds4300-pa2

## Overview
The goal of this project is to develop a RAG system that allows students to ask queries and interact with the course notes. This can be achieved using different variables such as text chunking, vector databases, embedding models, and LLMs. Through different combinations of these variables, students can analyze efficiency and accuracy to get an optimal combination. 

## How to Execute
- Download all of the folders inside of the repository. The data folder contains all of the files we used to ingest, but you can add or take away whichever files.
- In the src folder, we have a set of ingest.py and search.py files for each vector DB we experimented with (Redis, FAISS, Chroma). ingest.py and search.py are for Redis, ingest_faiss and search_faiss are for FAISS, chroma_ingest and chroma_search are for Chroma.
- For ingest, our RAG supports nomic-embed-text, all-minilm, and mxbai-embed-large as embedding models. These are coded in as strings within the get_embedding function and can be switched out to whichever model you would like to use. These are also Ollama models, so you would also need to pull from Ollama if you don't already have them. For search, our RAG supports mistral:latest and llama3.2 as our embedding models. Similarly, these are coded in as strings within the get_embedding function and can be switched out. Again, these must also be pulled from Ollama first.
- In order to vary the embedding model in the ingest_faiss.py file, the vector dimension also needs to be changed. Change the VECTOR_DIM variable at the top of the file to be 384 for all-minilm, 768 for nomic-embed-text, and 1024 for mxbai-embed-large.
- To test different chunk sizes and overlap sizes, you can simply switch out the number in the split_text_into_chunks function.
- To test the responses of our RAG, simply run your preferred search file. It will ask you to type in your question, and after pressing "enter", the RAG model will begin looking for related, processed information and will generate a response. You can keep asking questions until you manually exit or stop the running process.
