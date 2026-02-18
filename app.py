import streamlit as st
import requests
import os

st.title("Ollama Remote Interface")

# Adresse des Ollama-Servers (aus Docker-Netzwerk oder Umgebung)
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Nachrichten anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Eingabe
if prompt := st.chat_input("Schreib Ollama etwas..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Anfrage an den Ollama Container senden
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
