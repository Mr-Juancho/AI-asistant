import asyncio
import os
from dotenv import load_dotenv
import pyaudio
import websockets
import json
import time
import threading
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Inicializa los clientes de las APIs
# LÍNEA CORREGIDA
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
# LÍNEA CORREGIDA
elevenlabs_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

# Configuración del audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
# ─── NUEVO ────────────────────────────────────────────────────────────────
class SharedState:
    """Guarda si el asistente está hablando."""
    def __init__(self):
        self.is_speaking = False

state      = SharedState()      # instancia global
speak_lock = asyncio.Lock()     # asegura que solo hable una tarea a la vez

# --- 2. CAPTURA DE MICRÓFONO (PATRÓN CORREGIDO) ---
def capture_microphone(loop, audio_queue, state):
    """Lee el micrófono y envía audio a la cola SOLO cuando el asistente calla."""
    

    pa      = pyaudio.PyAudio()
    stream  = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                      input=True, frames_per_buffer=CHUNK)
    print("🎤 Micrófono listo.")

    try:
        while True:
            if not state.is_speaking:                       # ← aquí el mute
                data = stream.read(CHUNK, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)
            else:
                time.sleep(0.005)                           # no saturar CPU
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()


# --- 3. FUNCIÓN DEL CEREBRO (LLM) Y BOCA (TTS) ---
async def process_llm_and_speak(text: str, audio_queue, state):
    """Genera respuesta y la lee; desactiva el micrófono durante la locución."""
    async with speak_lock:                      # nunca dos voces a la vez
        state.is_speaking = True                # 🔇  silencia el micrófono
        try:
            # 🧠 GPT-4o ------------------------------------------------------
            resp = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "Eres JARVIS, el asistente personal de Juan, un ingeniero brillante con visión futurista. Respondes con precisión, ingenio y respeto, combinando eficiencia con sutileza. Anticipas necesidades, resuelves problemas técnicos, y nunca repites innecesariamente. Dirígete a él como “Juan”, mantén un tono profesional pero cercano, y actúa como un verdadero copiloto de inteligencia artificial"},
                    {"role": "user", "content": text}
                ]
            )
            answer = resp.choices[0].message.content.strip()
            print(f"Asistente: {answer}")

            # 👂 Vacía restos que quedaron en la cola de audio
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except asyncio.QueueEmpty: break

            # 🗣️ ElevenLabs --------------------------------------------------
            VOICE_ID  = "IKne3meq5aSn9XLyUdCD"
            VOICE_OPTS= {"stability":0.75,"similarity_boost":0.75,
                         "style":0.45,"use_speaker_boost":True}

            pcm_gen = elevenlabs_client.text_to_speech.stream(
                text=answer,
                voice_id=VOICE_ID,
                model_id="eleven_multilingual_v2",
                output_format="pcm_16000",
                voice_settings=VOICE_OPTS
            )

            pa  = pyaudio.PyAudio()
            out = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
            async for chunk in pcm_gen:
                if chunk: out.write(chunk)
            out.stop_stream(); out.close(); pa.terminate()

        finally:
            state.is_speaking = False           # 🔈  reactiva el micrófono
        
# --- 4. FUNCIÓN PRINCIPAL DE TRANSCRIPCIÓN (OÍDOS) ---
async def transcribe_audio(loop, audio_queue, state):
    """Envía audio a Deepgram y pasa frases completas a GPT-4o."""
    import websockets, json                     # imports locales
    DEEPGRAM_URL = (f"wss://api.deepgram.com/v1/listen?"
                    f"encoding=linear16&sample_rate=16000&channels=1&language=es")

    async with websockets.connect(
        DEEPGRAM_URL, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    ) as ws:
        print("🟢 Conectado a Deepgram. ¡Habla ahora!")

        # -------------- NUEVO sender() --------------
        async def sender():
            KEEPALIVE_EVERY = 3          # segundos (Deepgram corta a los ~10 s)
            last_sent = time.monotonic()

            while True:
                if not state.is_speaking:
                    data = await audio_queue.get()      # audio real
                    await ws.send(data)
                    last_sent = time.monotonic()
                else:
                    # Durante la locución envía solo KeepAlive
                    if time.monotonic() - last_sent >= KEEPALIVE_EVERY:
                        await ws.send('{"type":"KeepAlive"}')
                        last_sent = time.monotonic()
                    await asyncio.sleep(0.05)           # no satures CPU
        # --------------------------------------------


        async def receiver():
            full = ""
            async for msg in ws:
                res   = json.loads(msg)
                trans = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                if trans and res.get("is_final", False):
                    full += trans + " "
                    print(f"Tú: {full}")
                    if not state.is_speaking:   # no interrumpir la locución
                        loop.create_task(process_llm_and_speak(full, audio_queue, state))
                    full = ""

        await asyncio.gather(sender(), receiver())


# --- 5. PUNTO DE ENTRADA PRINCIPAL (PATRÓN CORREGIDO) ---
async def main():
    audio_queue = asyncio.Queue()
    loop        = asyncio.get_running_loop()

    # Hilo de micrófono
    mic_thread = threading.Thread(
        target=capture_microphone,
        args=(loop, audio_queue, state),
        daemon=True
    )
    mic_thread.start()

    await transcribe_audio(loop, audio_queue, state)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Asistente detenido.")
