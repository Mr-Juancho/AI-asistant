import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def test_model_latency(model_name: str, prompt: str = "Hola, ¿cómo estás?"):
    """
    Mide la latencia de primera respuesta de un modelo dado.
    Compatible con GPT-4 y GPT-5 (parámetros correctos).
    """
    print(f"\n⏱ Probando latencia con modelo: {model_name}")
    start_time = time.perf_counter()

    # Detectar si es GPT-5
    if model_name.startswith("gpt-5"):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=50,   # nuevo parámetro
            temperature=1               # obligatorio en GPT-5
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    print(f"📥 Respuesta: {response.choices[0].message.content.strip()}")
    print(f"⚡ Latencia total: {latency_ms:.2f} ms")
    return latency_ms


if __name__ == "__main__":
    modelo_actual = os.getenv("OPENAI_LLM", "gpt-4o")
    test_model_latency(modelo_actual)
    test_model_latency("gpt-5-mini")

