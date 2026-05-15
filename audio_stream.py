import sounddevice as sd
import queue
import numpy as np

SAMPLE_RATE = 16000
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def start_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback,
        blocksize=1024
    )
    stream.start()
    return stream
