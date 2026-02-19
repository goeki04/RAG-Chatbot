import os
import time
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
no_proxy = os.getenv('NO_PROXY', '') + ',vdb_1,vdb_2,localhost,127.0.0.1'
os.environ['NO_PROXY'] = no_proxy
os.environ['no_proxy'] = no_proxy

def wait_for_qdrant(url, attempts=15):
    client = QdrantClient(url=url)
    for i in range(attempts):
        try:
            client.get_collections()
            print(f" Qdrant ({url}) ist bereit!")
            return True
        except Exception:
            print(f" Versuche {i+1}/{attempts}: Warte auf Qdrant...")
            time.sleep(2)
    return False

embeddings = FastEmbedEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

def load_documents_from_directory(directory_path, file_extension="**/*.txt"):
    """
    Lädt alle Dateien aus einem Verzeichnis.
    
    Unterstützte Formate:
    - .txt: **/*.txt
    - .pdf: **/*.pdf
    - .md: **/*.md
    """
    print(f" Lade Dateien aus: {directory_path} ({file_extension})")
    
    if file_extension.endswith('.pdf'):
        loader = DirectoryLoader(
            directory_path,
            glob=file_extension,
            loader_cls=PyPDFLoader,
            show_progress=True
        )
    else:
        loader = DirectoryLoader(
            directory_path,
            glob=file_extension,
            loader_cls=TextLoader,
            show_progress=True
        )
    
    documents = loader.load()
    print(f"{len(documents)} Dokumente geladen")
    return documents

def ingest_to_collection(target_url, data_path, collection_name, file_pattern="**/*.txt"):
    print(f"\n--- Starte Ingestion für {collection_name} auf {target_url} ---")
    
    data = load_documents_from_directory(data_path, file_pattern)
    
    if not data:
        print(f"Keine Dateien gefunden in {data_path} mit Pattern {file_pattern}")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    print(f"{len(docs)} Chunks erstellt")
    
    print(f"Erstelle Embeddings und speichere in Qdrant...")
    QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=target_url,
        collection_name=collection_name,
        force_recreate=True
    )
    print(f"Erfolgreich {len(docs)} Chunks in '{collection_name}' gespeichert!\n")

if __name__ == "__main__":
    if wait_for_qdrant("http://vdb_1:6333") and wait_for_qdrant("http://vdb_2:6333"):
        data_dir = "/app/data"
        
        ingest_to_collection(
            "http://vdb_1:6333",
            f"{data_dir}/db1",
            "datenbank_eins",
            file_pattern="**/*.pdf"
        )
        
        ingest_to_collection(
            "http://vdb_2:6333",
            f"{data_dir}/db2",
            "datenbank_zwei",
            file_pattern="**/*.pdf"
        )