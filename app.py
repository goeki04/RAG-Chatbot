import streamlit as st
import requests
import os
from langchain_community.llms import Ollama
#from qdrant_client import QdrantClient
#from langchain_community.embeddings import FastEmbedEmbeddings

st.title("CatGPT says Hello!")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
#embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#vectorstore = QdrantVectorStore(client=client, collection_name="Berichtsheft", embeddings=embeddings)
#retriever = vectorstore.as_retriever()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Waiting for prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        answer = response.json().get("response", "Fehler: Keine Antwort erhalten.")
    except Exception as e:
        answer = f"Verbindung zu Ollama fehlgeschlagen: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})