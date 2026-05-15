import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import requests
import webrtcvad
from faster_whisper import WhisperModel
import time

# =====================================================
# CONFIG
# =====================================================
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# =====================================================
# QUEUES (THREAD SAFE PIPELINE)
# =====================================================
audio_queue = queue.Queue()
text_queue = queue.Queue()
ui_queue = queue.Queue()

# =====================================================
# SESSION STATE
# =====================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "buffer" not in st.session_state:
    st.session_state.buffer = []

# =====================================================
# WHISPER MODEL
# =====================================================
model = WhisperModel("base", compute_type="int8")

def transcribe(audio):
    segments, _ = model.transcribe(audio, language="el")
    return " ".join([s.text for s in segments]).strip()

# =====================================================
# LLM CORRECTION (OPTIONAL)
# =====================================================
def correct(text):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": f"""
You are a Greek radiology dictation assistant.

RULES:
- Do NOT change meaning
- Fix grammar only
- Normalize medical terminology
- Output clean radiology report style text

TEXT:
{text}
""",
                "stream": False
            },
            timeout=10
        )
        return r.json().get("response", text).strip()
    except:
        return text

# =====================================================
# VAD
# =====================================================
vad = webrtcvad.Vad(2)

speech_buffer = []
recording = False

def is_speech(pcm_chunk: bytes):
    try:
        return vad.is_speech(pcm_chunk, SAMPLE_RATE)
    except:
        return False

def audio_callback(indata, frames, time_info, status):
    global speech_buffer, recording

    pcm16 = (indata[:, 0] * 32767).astype(np.int16)

    for i in range(0, len(pcm16), FRAME_SAMPLES):
        chunk = pcm16[i:i + FRAME_SAMPLES]

        if len(chunk) < FRAME_SAMPLES:
            continue

        if is_speech(chunk.tobytes()):
            recording = True
            speech_buffer.append(chunk.copy())
        else:
            if recording and len(speech_buffer) > 0:
                audio_queue.put(np.concatenate(speech_buffer))
                speech_buffer = []
            recording = False

# =====================================================
# AUDIO STREAM
# =====================================================
def start_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    stream.start()
    return stream

# =====================================================
# STT WORKER
# =====================================================
def stt_worker():
    while st.session_state.running:
        try:
            audio = audio_queue.get(timeout=1)

            audio = audio.astype(np.float32).flatten()

            # avoid whisper garbage on tiny chunks
            if len(audio) / SAMPLE_RATE < 1.5:
                continue

            text = transcribe(audio)

            if text.strip():
                text_queue.put(text)

        except:
            continue

# =====================================================
# OPTIONAL LLM WORKER
# =====================================================
def llm_worker():
    while st.session_state.running:
        try:
            text = text_queue.get(timeout=1)
            corrected = correct(text)
            ui_queue.put(corrected)
        except:
            continue

# =====================================================
# UI UPDATE LOOP (MAIN THREAD)
# =====================================================
def drain_ui_queue():
    while not ui_queue.empty():
        st.session_state.buffer.append(ui_queue.get())

# =====================================================
# UI
# =====================================================
st.title("🏥 Offline Greek Radiology Dictation System")

col1, col2, col3 = st.columns(3)

stream_ref = {"stream": None}

with col1:
    if st.button("🎤 Start Dictation"):
        st.session_state.running = True

        stream_ref["stream"] = start_stream()

        threading.Thread(target=stt_worker, daemon=True).start()
        threading.Thread(target=llm_worker, daemon=True).start()

with col2:
    if st.button("⛔ Stop"):
        st.session_state.running = False

        try:
            if stream_ref["stream"]:
                stream_ref["stream"].stop()
                stream_ref["stream"].close()
        except:
            pass

with col3:
    if st.button("🧹 Clear"):
        st.session_state.buffer = []

st.divider()

# =====================================================
# DRAIN QUEUES SAFELY
# =====================================================
drain_ui_queue()

# =====================================================
# LIVE REPORT
# =====================================================
st.subheader("🧾 Live Report")

live_text = "\n".join(st.session_state.buffer)

st.text_area("Draft Report", live_text, height=300)

# =====================================================
# FINAL REPORT
# =====================================================
st.subheader("📋 Final Copy-Ready Report")

st.text_area("Editable Final Report", live_text, height=300)

st.code(live_text)

st.download_button(
    "⬇️ Download Report",
    live_text,
    file_name="radiology_report.txt"
)

# =====================================================
# AUTO REFRESH WHILE RUNNING
# =====================================================
if st.session_state.running:
    time.sleep(0.2)
    st.rerun()
