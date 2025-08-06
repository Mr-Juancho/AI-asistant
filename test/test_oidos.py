import asyncio
import os
from dotenv import load_dotenv
import pyaudio
import websockets
import json
import threading

# --- 1. CONFIGURACIÓN ---
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Configuración del audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
audio_queue = asyncio.Queue()

# --- 2. CAPTURA DE MICRÓFONO (CORREGIDO) ---
# Ahora acepta el 'loop' como argumento
def capture_microphone(loop):
    """Captura audio del mic y lo pone en la cola usando el loop del hilo principal."""
    import time
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎤 Micrófono listo. Habla para empezar la prueba.")
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            # Usa el 'loop' que le pasamos para comunicarse de forma segura
            asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)
            time.sleep(0.001)  # Pequeño delay para evitar saturar el CPU
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo captura de audio...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# --- 3. TRANSCRIPCIÓN EN TIEMPO REAL ---
async def transcribe_audio():
    """Conecta a Deepgram y muestra la transcripción."""
    DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={RATE}&channels={CHANNELS}&interim_results=true"
    
    try:
        async with websockets.connect(DEEPGRAM_URL, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
            print("🟢 Conectado a Deepgram. ¡Escuchando!")

            async def sender(ws):
                while True:
                    data = await audio_queue.get()
                    await ws.send(data)

            async def receiver(ws):
                async for msg in ws:
                    res = json.loads(msg)
                    transcript = res.get('channel', {}).get('alternatives', [{}])[0].get('transcript', '')
                    if transcript:
                        print(f"Escuchando: {transcript}", end='\r', flush=True)

            await asyncio.gather(sender(ws), receiver(ws))
    except Exception as e:
        print(f"\nError de conexión con Deepgram: {e}")

# --- PUNTO DE ENTRADA (CORREGIDO) ---
async def main():
    # Obtenemos el loop del chef (hilo principal) ANTES de crear el otro hilo
    loop = asyncio.get_running_loop()
    
    # Inicia la captura del micrófono, pasándole el 'loop' (el walkie-talkie)
    mic_thread = threading.Thread(target=capture_microphone, args=(loop,), daemon=True)
    mic_thread.start()
    
    await transcribe_audio()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Prueba detenida.")