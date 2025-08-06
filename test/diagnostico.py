import os
import elevenlabs
from dotenv import load_dotenv

print("--- INICIANDO DIAGNÓSTICO FINAL DE ELEVENLABS ---")

try:
    # Imprime la versión para estar seguros
    print(f"Versión instalada: {elevenlabs.__version__}")
    print("-" * 50)

    # Carga la API Key
    load_dotenv()
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

    if not ELEVENLABS_API_KEY:
        print("ERROR: No se encontró la ELEVENLABS_API_KEY en el archivo .env")
    else:
        # Usamos el cliente Asíncrono, ya que el entorno es async
        client = elevenlabs.client.AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

        # --- LA PARTE NUEVA E IMPORTANTE ---
        # Accedemos al sub-módulo que nos ha estado dando problemas
        tts_sub_client = client.text_to_speech

        print(f"Tipo de objeto del sub-cliente TTS: {type(tts_sub_client)}")
        print("-" * 50)

        # Imprime TODOS los métodos y atributos disponibles en 'text_to_speech'
        print("Métodos y atributos disponibles en 'text_to_speech':")
        from pprint import pprint
        pprint(dir(tts_sub_client))
        print("-" * 50)

except Exception as e:
    print(f"Ocurrió un error durante el diagnóstico: {e}")

print("\n--- DIAGNÓSTICO FINALIZADO ---")