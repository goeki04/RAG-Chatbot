import streamlit as st
import requests
import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["FASTEMBED_CACHE_PATH"] = "/tmp/fastembed_cache"

VDB_SOURCES = [
    {"host": "vdb_1", "collection": "datenbank_eins"},
    {"host": "vdb_2", "collection": "datenbank_zwei"}
]

st.set_page_config(page_title="CatGPT", layout="wide")
st.title("🐱 CatGPT says Hello!")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = get_embeddings()
@st.cache_resource
def get_all_vectorstores(_embeddings):
    vectorstores = []
    for source in VDB_SOURCES:
        client = QdrantClient(host=source["host"], port=6333)
        if not client.collection_exists(source["collection"]):
            client.create_collection(
                collection_name=source["collection"],
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        vs = QdrantVectorStore(client=client, collection_name=source["collection"], embedding=_embeddings)
        vectorstores.append(vs)
    return vectorstores


vectorstores = get_all_vectorstores(embeddings)

def ingest_pdfs(folder_path, target_vectorstore): 
    with st.spinner(f"Verarbeite PDFs aus {folder_path}..."):
        loader = PyPDFDirectoryLoader(folder_path)
        docs = loader.load()
        
        if not docs:
            st.error(f"Keine Dateien in {folder_path} gefunden!")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        target_vectorstore.add_documents(chunks)
        st.sidebar.success(f" {len(chunks)} Abschnitte importiert!")

with st.sidebar:
    st.header("⚙️ Verwaltung")
    if st.button("📚 Daten aus db1 einlesen"):
        ingest_pdfs("/app/data/db1", vectorstores[0])
    
    if st.button("📚 Daten aus db2 einlesen"):
        ingest_pdfs("/app/data/db2", vectorstores[1])
    
    if st.button("🗑️ Datenbanken leeren"):
        for source in VDB_SOURCES:
            tmp_client = QdrantClient(host=source["host"], port=6333)
            if tmp_client.collection_exists(source["collection"]):
                tmp_client.delete_collection(source["collection"])
            tmp_client.create_collection(
                collection_name=source["collection"],
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        st.cache_resource.clear()
        st.sidebar.success("Beide Datenbanken blitzblank geputzt!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Frag mich etwas über deine Dokumente..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    all_results = []
    for vs in vectorstores:
        results = vs.similarity_search_with_score(prompt, k=3)
        all_results.extend(results)
    
    all_results.sort(key=lambda x: x[1], reverse=True)
    
    top_results = all_results[:4]
    
    docs = [doc for doc, score in top_results]
    
    context = "\n---\n".join([doc.page_content for doc in docs])

    full_prompt = f"""Du bist ein intelligenter Assistent für einen Angestellten. 
    Beantworte die Frage präzise und ausschließlich auf Basis der folgenden Informationen.
    
    INFORMATIONEN:
    {context}
    
    FRAGE:
    {prompt}"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            res = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "llama3.2:3b", "prompt": full_prompt, "stream": True},
                stream=True
            )
            
            for line in res.iter_lines():
                if line:
                    chunk = json.loads(line)
                    full_response += chunk.get("response", "")
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            if top_results:
                with st.expander("Verwendete Quellen & Debug-Info"):
                    for doc, score in top_results:
                        st.write(f"- **Score: {score:.4f}** | {doc.metadata.get('source', 'Unbekannt')}")
                    
                    st.divider()
                    st.markdown("**Das hat die KI tatsächlich gelesen:**")
                    st.info(context)
                        
        except Exception as e:
            st.error(f"Fehler: {e}")