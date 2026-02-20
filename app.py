import streamlit as st
import requests
import os
from langchain_community.llms import Ollama
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"
st.title("CatGPT says Hello!")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
client =QdrantClient(host="vdb_1", port=6333)

collection_name = "Berichtsheft"

if not client.collection_exists(collection_name):
    from qdrant_client.http import models
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384, # Größe für paraphrase-multilingual-MiniLM-L12-v2
            distance=models.Distance.COSINE
        ),
    )
    st.info(f"Collection '{collection_name}' wurde neu erstellt.")

vectorstore = QdrantVectorStore(client=client, collection_name="Berichtsheft", embedding=embeddings)
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