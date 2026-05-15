import webrtcvad
import collections
import numpy as np
import sounddevice as sd
import queue

SAMPLE_RATE = 16000

vad = webrtcvad.Vad(2)  # 0-3 (higher = more aggressive filtering)

audio_queue = queue.Queue()

# buffer for speech segments
ring_buffer = collections.deque(maxlen=30)

speech_frames = []
recording = False

def is_speech(frame_bytes):
    return vad.is_speech(frame_bytes, SAMPLE_RATE)

def callback(indata, frames, time, status):
    global recording, speech_frames

    audio = indata.copy().tobytes()

    if is_speech(audio[:320]):  # small frame check
        recording = True
        speech_frames.append(indata.copy())
    else:
        if recording:
            # silence detected → finalize segment
            if len(speech_frames) > 0:
                audio_queue.put(np.concatenate(speech_frames))
                speech_frames = []
            recording = False

def start_vad_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=1024
    )
    stream.start()
    return stream
