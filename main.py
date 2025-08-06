import asyncio
import os
from dotenv import load_dotenv
import pyaudio
import websockets
import json
import threading
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import stream, play

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

# --- 2. CAPTURA DE MICRÓFONO (PATRÓN CORREGIDO) ---
def capture_microphone(loop, audio_queue):
    """Captura audio del mic y lo pone en la cola usando el loop del hilo principal."""
    import time
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("🎤 Micrófono listo.")
    
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)
            time.sleep(0.001)  # Pequeño delay para evitar saturar el CPU
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo captura de audio...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# --- 3. FUNCIÓN DEL CEREBRO (LLM) Y BOCA (TTS) ---
# Reemplaza tu función entera con esta versión final y correcta
async def process_llm_and_speak(text):
    """Envía el texto a OpenAI y reproduce la respuesta de ElevenLabs en streaming."""
    print("🧠 Pensando...")
    
    try:
        response_stream = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres JARVIS, un asistente de IA conversacional, directo, ingenioso y conciso."},
                {"role": "user", "content": text}
            ],
            stream=True
        )
        
        async def text_iterator():
            async for chunk in response_stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    yield delta
        
        print("Asistente: ", end="", flush=True)

        # LA LLAMADA FINAL Y CORRECTA CON EL CLIENTE ASÍNCRONO
        audio_stream = await elevenlabs_client.generate(
            text=text_iterator(),
            voice="Adam", # Ahora el nombre "Adam" funcionará
            model="eleven_multilingual_v2",
            stream=True
        )
        
        # 'stream' es una función de ayuda que consume el stream de audio
        stream(audio_stream)
        print("\n")

    except Exception as e:
        print(f"\nError en process_llm_and_speak: {e}")
        
# --- 4. FUNCIÓN PRINCIPAL DE TRANSCRIPCIÓN (OÍDOS) ---
async def transcribe_audio(loop, audio_queue):
    """Escucha, envía a Deepgram y, al finalizar una frase, la pasa al cerebro."""
    # AÑADIMOS EL PARÁMETRO DE IDIOMA
    # ESTA ES LA LÍNEA CORREGIDA
    DEEPGRAM_URL = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={RATE}&channels={CHANNELS}&language=es"
    
    try:
        async with websockets.connect(DEEPGRAM_URL, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
            print("🟢 Conectado a Deepgram. ¡Habla ahora!")

            async def sender(ws):
                while True:
                    data = await audio_queue.get()
                    await ws.send(data)

            async def receiver(ws):
                full_transcript = ""
                async for msg in ws:
                    res = json.loads(msg)
                    transcript = res.get('channel', {}).get('alternatives', [{}])[0].get('transcript', '')
                    
                    # Esta lógica es buena: esperamos a que termines la frase para no enviar texto a medias
                    if transcript and res.get('is_final', False):
                        full_transcript += transcript + " "
                        print(f"Tú: {full_transcript}")
                        # Una vez que tenemos una frase completa, la enviamos al cerebro
                        # Usamos create_task para que no bloquee la recepción de más mensajes
                        loop.create_task(process_llm_and_speak(full_transcript))
                        full_transcript = ""

            await asyncio.gather(sender(ws), receiver(ws))
            
    except Exception as e:
        print(f"Error de conexión con Deepgram: {e}")

# --- 5. PUNTO DE ENTRADA PRINCIPAL (PATRÓN CORREGIDO) ---
async def main():
    audio_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    
    mic_thread = threading.Thread(target=capture_microphone, args=(loop, audio_queue), daemon=True)
    mic_thread.start()
    
    await transcribe_audio(loop, audio_queue)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Asistente detenido.")