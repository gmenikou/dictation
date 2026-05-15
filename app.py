import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import requests
import webrtcvad
from faster_whisper import WhisperModel

# =====================================================
# CONFIG
# =====================================================
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024

audio_queue = queue.Queue()

# =====================================================
# SESSION STATE
# =====================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "buffer" not in st.session_state:
    st.session_state.buffer = []

# =====================================================
# WHISPER (LOCAL)
# =====================================================
model = WhisperModel("base", compute_type="int8")

def transcribe(audio):
    segments, _ = model.transcribe(audio, language="el")
    return " ".join([s.text for s in segments])

# =====================================================
# OLLAMA CORRECTION
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
- Only fix grammar
- Normalize medical terminology
- Output clean radiology report style text

TEXT:
{text}
""",
                "stream": False
            }
        )
        return r.json()["response"]
    except:
        return text

# =====================================================
# VAD (VOICE ACTIVITY DETECTION)
# =====================================================
vad = webrtcvad.Vad(2)

recording = False
speech_buffer = []

def is_speech(frame):
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except:
        return False

def audio_callback(indata, frames, time, status):
    global recording, speech_buffer

    frame = indata.copy()

    # convert to 16-bit PCM for VAD
    pcm = (frame * 32767).astype(np.int16).tobytes()

    if is_speech(pcm[:320]):
        recording = True
        speech_buffer.append(frame.copy())
    else:
        if recording:
            if len(speech_buffer) > 0:
                audio_queue.put(np.concatenate(speech_buffer))
                speech_buffer = []
            recording = False

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
# PROCESSING LOOP
# =====================================================
def process_loop():
    while st.session_state.running:
        try:
            audio = audio_queue.get(timeout=1)

            audio = np.squeeze(audio).astype(np.float32)

            # 1. STT
            text = transcribe(audio)
            if not text.strip():
                continue

            # 2. LLM correction
            corrected = correct(text)

            # 3. Append to report
            st.session_state.buffer.append(corrected)

        except:
            continue

# =====================================================
# UI
# =====================================================
st.title("🏥 Offline Greek Radiology Dictation System")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎤 Start Dictation"):
        st.session_state.running = True
        start_stream()
        threading.Thread(target=process_loop, daemon=True).start()

with col2:
    if st.button("⛔ Stop"):
        st.session_state.running = False

with col3:
    if st.button("🧹 Clear"):
        st.session_state.buffer = []

st.divider()

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

final_text = live_text

st.text_area("Editable Final Report", final_text, height=300)

st.code(final_text)

st.download_button(
    "⬇️ Download Report",
    final_text,
    file_name="radiology_report.txt"
)
