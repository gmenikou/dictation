import streamlit as st
import numpy as np
import threading
import requests
from faster_whisper import WhisperModel

from audio_stream import start_stream, audio_queue

st.title("🏥 Offline Dictation System")

# =========================
# MODEL
# =========================
model = WhisperModel("base", compute_type="int8")

# =========================
# SESSION STATE
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

if "buffer" not in st.session_state:
    st.session_state.buffer = []

# =========================
# WHISPER FUNCTION
# =========================
def transcribe(audio):
    segments, _ = model.transcribe(audio, language="el")
    return " ".join([s.text for s in segments])

# =========================
# OLLAMA FUNCTION
# =========================
def correct(text):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"""
You are a Greek radiology dictation corrector.

Rules:
- do NOT change meaning
- only fix grammar
- only normalize medical terms

Text:
{text}
""",
                "stream": False
            }
        )
        return r.json()["response"]
    except:
        return text

# =========================
# PROCESS LOOP
# =========================
def process_audio():
    while st.session_state.running:
        audio = audio_queue.get()

        audio = np.squeeze(audio).astype(np.float32)

        text = transcribe(audio)
        if not text.strip():
            continue

        corrected = correct(text)

        st.session_state.buffer.append(corrected)

# =========================
# UI
# =========================
col1, col2 = st.columns(2)

with col1:
    if st.button("🎤 Start Dictation"):
        st.session_state.running = True
        start_stream()
        threading.Thread(target=process_audio, daemon=True).start()

with col2:
    if st.button("⛔ Stop"):
        st.session_state.running = False

st.divider()

st.subheader("🧾 Live Report")

st.text_area(
    "Report",
    value="\n".join(st.session_state.buffer),
    height=300
)

st.subheader("📋 Final Copy")

final_text = "\n".join(st.session_state.buffer)

st.code(final_text)

st.download_button(
    "⬇️ Download Report",
    final_text,
    file_name="report.txt"
)
