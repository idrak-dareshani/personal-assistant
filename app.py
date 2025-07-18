# app.py
import os
import time
import requests
import streamlit as st
from rag_utils import list_ollama_models, get_model_info
from rag_utils import load_documents, create_vectorstore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

st.markdown("""
    <style>
        /* Reduce padding in main content */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Remove extra spacing at the top of sidebar */
        section[data-testid="stSidebarHeader"] {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="🧠 Personal ChatGPT", layout="wide")
st.title("🧠 Personal ChatGPT - RAG + Multi-model")

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

if "vs" not in st.session_state:
    st.session_state.vs = None

with st.sidebar:
    available_models = list_ollama_models()
    # Sidebar: Model selection and clean metadata
    st.sidebar.markdown("### 🤖 Model Selection")
    selected_model = st.sidebar.selectbox("Choose a model", available_models)

    if selected_model:
        model_info = get_model_info(selected_model)
        with st.sidebar.expander("ℹ️ Model Summary", expanded=False):
            st.markdown(f"**Name**: {model_info['name']}")
            st.markdown(f"**Parameters**: {model_info['parameter_size']}")
            st.markdown(f"**Size GB**: {model_info['size_gb']}")
            st.markdown(f"**Family**: `{model_info['family']}`")
            st.markdown(f"**Quantization**: {model_info['quantization']}")

    st.markdown("### 📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files", type=["txt", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.docs = load_documents(uploaded_files)
        vectorstore, chunk_summary = create_vectorstore(st.session_state.docs)
        st.session_state.vectorstore = vectorstore
        st.session_state.chunk_summary = chunk_summary
        st.success(f"📚 {len(uploaded_files)} file(s) uploaded.")

    if st.session_state.get("docs"):
        with st.expander("📁 Uploaded Files Summary", expanded=False):
            chunk_info = st.session_state.get("chunk_summary", {})
            if chunk_info:
                for file_name, chunk_count in chunk_info.items():
                    st.markdown(f"- **{file_name}** — 🧩 `{chunk_count}` chunks")
            else:
                st.markdown("No documents uploaded.")
                    
# Chat interface
query = st.chat_input("💬 Ask me anything...")

if query:
    start_time = time.time()

    history_prompt = ""
    for q, a in st.session_state.chat_memory[-3:]:
        history_prompt += f"User: {q}\nAssistant: {a}\n"

    if st.session_state.get("vectorstore"):
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt_template = PromptTemplate.from_template(
            """
            You are a helpful assistant. Use the context below to answer the user question.

            Context:
            {context}

            Conversation History:
            {history}

            Question: {question}
            Answer:
            """
        )

        llm = OllamaLLM(model=selected_model, base_url="http://localhost:11434")

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        docs = retriever.invoke(query)
        doc_text = "\n".join([doc.page_content for doc in docs])

        final_prompt = prompt_template.format(
            context=doc_text, history=history_prompt, question=query
        )

    else:
        final_prompt = f"""
            You are a helpful assistant. Use the chat history to respond.

            Chat History:
            {history_prompt}

            User: {query}
            Assistant:"""

    # Call Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": selected_model, "prompt": final_prompt, "stream": False}
    )

    duration = time.time() - start_time

    reply = response.json()["response"]
    st.session_state.chat_memory.append((query, reply))

    for user_msg, bot_msg in st.session_state.chat_memory:
        st.chat_message("user").write(user_msg)
        st.chat_message("assistant").write(bot_msg)

    st.caption(f"⏱️ Response time: {duration:.2f} seconds")