"""
sound_utils.py
---------------
Generates a short two-tone siren beep entirely in code (no audio file
needed) and returns it as a base64 string, so it can be embedded directly
into an HTML <audio> tag inside the Streamlit app.
"""

import io
import wave
import struct
import math
import base64


def generate_alert_beep_base64(duration_ms: int = 900, sample_rate: int = 44100) -> str:
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    wf = wave.open(buf, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)

    segment = sample_rate // 4  # switch tone every quarter second -> siren feel
    for i in range(n_samples):
        t = i / sample_rate
        freq = 950.0 if (i // segment) % 2 == 0 else 1300.0
        value = int(0.5 * 32767 * math.sin(2 * math.pi * freq * t))
        wf.writeframes(struct.pack("<h", value))
    wf.close()

    return base64.b64encode(buf.getvalue()).decode("utf-8")
