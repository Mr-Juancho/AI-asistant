# voice_tester_elevenlabs.py (Corregido)
import asyncio
import os
from dotenv import load_dotenv
import pyaudio
from elevenlabs.client import AsyncElevenLabs
from elevenlabs.client import ElevenLabs

# --- 1. CONFIGURACIÓN ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    print("Error: La variable de entorno ELEVENLABS_API_KEY no está configurada.")
    exit()

try:
    client_sync = ElevenLabs(api_key=ELEVENLABS_API_KEY)
except Exception as e:
    print(f"Error al inicializar el cliente de ElevenLabs: {e}")
    print("Verifica que tu API Key sea correcta.")
    exit()

client_async = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
pa = pyaudio.PyAudio()

async def test_voice():
    """
    Función principal para listar, seleccionar y probar voces.
    """
    print("Obteniendo lista de voces de ElevenLabs...")
    try:
        voices = client_sync.voices.get_all().voices
    except Exception as e:
        print(f"\nError al obtener las voces: {e}")
        print("Asegúrate de que tu plan de ElevenLabs permite el acceso a la API.")
        return
        
    print("─" * 50)
    print("Voces Disponibles:")
    for i, voice in enumerate(voices):
        labels = ", ".join(voice.labels.values()) if voice.labels else "N/A"
        print(f"  {i+1: >2}) {voice.name: <20} | Labels: {labels: <25} | ID: {voice.voice_id}")
    print("─" * 50)

    while True:
        try:
            choice_num = input("\nElige el número de la voz que quieres probar (o 'q' para salir): ")
            if choice_num.lower() == 'q':
                break
            
            voice_index = int(choice_num) - 1
            if not 0 <= voice_index < len(voices):
                print("Número de voz inválido. Inténtalo de nuevo.")
                continue

            selected_voice = voices[voice_index]
            print(f"Has seleccionado: {selected_voice.name}")

            text_to_speak = input("Escribe el texto que quieres escuchar: ")
            if not text_to_speak.strip():
                print("No se ha introducido texto.")
                continue

            print(f"\nGenerando audio con la voz '{selected_voice.name}'...")

            stream_out = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                frames_per_buffer=CHUNK
            )

            # --- LÍNEA CORREGIDA ---
            # Se ha eliminado el 'await'. La función devuelve un generador, no una corrutina.
            pcm_stream = client_async.text_to_speech.stream(
                text=text_to_speak,
                voice_id=selected_voice.voice_id,
                model_id="eleven_turbo_v2",
                output_format=f"pcm_{RATE}"
            )

            stream_out.start_stream()
            # El bucle 'async for' es el que consume el generador
            async for chunk in pcm_stream:
                if chunk:
                    stream_out.write(chunk)
            
            stream_out.stop_stream()
            stream_out.close()
            print("Reproducción finalizada.")

        except ValueError:
            print("Entrada inválida. Por favor, introduce un número.")
        except KeyboardInterrupt:
            print("\nSaliendo del probador de voces.")
            break
        except Exception as e:
            print(f"\nOcurrió un error inesperado: {e}")
            break

if __name__ == "__main__":
    try:
        asyncio.run(test_voice())
    finally:
        pa.terminate()
        print("\n👋 Programa terminado.")