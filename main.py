# main.py
import asyncio, threading
from jarvis.config import load_config, cleanup   # <-- IMPORTA cleanup
from jarvis.state import make_state, history_init
from jarvis.audio import capture_microphone
from jarvis.asr import transcribe_audio
from jarvis.llm import llm_streamer
from jarvis.console import sprint, c, GRAY

async def main():
    cfg = load_config()
    state = await make_state()
    await history_init(state)

    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    mic_thread = threading.Thread(
        target=capture_microphone, args=(cfg, state, loop, audio_queue), daemon=True
    )
    mic_thread.start()

    try:
        loop.create_task(llm_streamer(cfg, state, loop))
        await transcribe_audio(cfg, state, loop, audio_queue)
    finally:
        # <-- la limpieza va aquí, donde SÍ existe cfg
        cleanup(cfg)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sprint(c("\n\n👋 Asistente detenido.", GRAY))
