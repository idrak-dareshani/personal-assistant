# rag_utils.py
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
import os
import json
import requests

def load_documents(uploaded_files, upload_dir="data/uploads"):
    all_docs = []
    os.makedirs(upload_dir, exist_ok=True)

    # Remove existing files in the upload directory
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

def create_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    all_chunks = []
    chunk_summary = {}

    for doc in documents:
        chunks = splitter.split_documents([doc])
        file_name = os.path.basename(doc.metadata.get("source", "unknown"))
        chunk_summary[file_name] = chunk_summary.get(file_name, 0) + len(chunks)
        all_chunks.extend(chunks)

    vectorstore = FAISS.from_documents(all_chunks, embedding=embeddings)
    return vectorstore, chunk_summary
