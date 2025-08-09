# jarvis/audio.py
import time, asyncio
from jarvis.console import sprint, c, GRAY

def capture_microphone(cfg, state, loop, audio_queue):
    stream = cfg.pa.open(format=cfg.FORMAT, channels=cfg.CHANNELS, rate=cfg.RATE,
                         input=True, frames_per_buffer=cfg.CHUNK, start=True)
    sprint(c("🎤 Micrófono listo.", GRAY))
    try:
        while True:
            if not state.shared_state.is_speaking:
                data = stream.read(cfg.CHUNK, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)
            else:
                time.sleep(0.005)
    finally:
        stream.stop_stream()
        stream.close()
