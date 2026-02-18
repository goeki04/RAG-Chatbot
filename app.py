import os
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
# Konfiguration über Umgebungsvariablen (für Docker Compose)
QDRANT_URL = os.getenv("QDRANT_URL", "http://vdb_1:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "datenbank_eins")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")

st.set_page_config(page_title="Local RAG", page_icon="🤖")

@st.cache_resource
def init_rag():
    embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL
    )
    llm = ChatOllama(model="llama3.1", base_url=OLLAMA_URL, temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

st.title("🛡️ Interner Wissens-Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Frag mich etwas über die Dokumente..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            qa = init_rag()
        except Exception as e:
            st.error(str(e))
        res = qa.invoke({"query": prompt})
        answer = res["result"]
        
        st.markdown(answer)
        with st.expander("Quellen"):
            for doc in res["source_documents"]:
                st.write(f"- {doc.metadata.get('source', 'Unbekannt')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})