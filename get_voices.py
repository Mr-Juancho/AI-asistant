# get_voices.py
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Carga la API Key
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    raise ValueError("API Key no encontrada. Revisa tu archivo .env")

# Inicializa el cliente (el síncrono es más simple para esta tarea)
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

print("Buscando voces predefinidas en ElevenLabs...")
print("-" * 50)

try:
    # Obtenemos todas las voces
    voices_response = client.voices.get_all()

    # Iteramos y mostramos la información relevante
    for voice in voices_response.voices:
        # Mostramos solo las voces predefinidas
        if voice.category == 'premade':
            print(f"Nombre: {voice.name}")
            print(f"  ID:   {voice.voice_id}")
            # Buscamos la etiqueta de acento para dar una pista del idioma
            labels = voice.labels if voice.labels else {}
            print(f"  Info: {labels.get('description', 'N/A')}")
            print("-" * 20)
except Exception as e:
    print(f"Ocurrió un error al contactar a la API: {e}")