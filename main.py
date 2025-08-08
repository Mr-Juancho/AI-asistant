# realtime_assistant_step3_tts_por_trozos.py
# Paso 3: TTS por trozos (empieza a hablar con los primeros tokens)
# - El LLM stream se trocea por oraciones/longitud y se manda a ElevenLabs en piezas.
# - Latencia real baja: el audio arranca con los primeros tokens.
# - Mantiene la consola con colores ([TÚ] rojo, [ASISTENTE] azul) y previews de Paso 2.
# - Previews se cancelan cuando llega el final y arranca el TTS.

import asyncio
import os
import re
from dotenv import load_dotenv
import pyaudio
import websockets
import json
import time
import threading
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from memory import MemoryManager

#comentario adicional
# ───────────────────────────────────────────────────────────────────────────
# 1) CONFIG INICIAL
# ───────────────────────────────────────────────────────────────────────────
load_dotenv()
memory = MemoryManager()

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY")

openai_client       = AsyncOpenAI(api_key=OPENAI_API_KEY)
elevenlabs_client   = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

# Audio I/O
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000
CHUNK    = 480          # 10–20 ms
pa       = pyaudio.PyAudio()
FRAME_BYTES = CHUNK * 2
TAIL_SILENCE_FRAMES = CHUNK * 4

out = pa.open(format=pyaudio.paInt16,
              channels=1,
              rate=RATE,
              output=True,
              frames_per_buffer=CHUNK,
              start=False)

# Historial (RAM por ahora)
conversation_history = []
MAX_HISTORY_TURNS = 5

# Globals
current_llm_task = None
current_preview_task = None
tts_started_event = asyncio.Event()
llm_text_queue = asyncio.Queue(maxsize=8)

grace_mode = False
suppress_user_partials = False   # ⬅️ NUEVO: no pintes [TÚ] mientras habla el asistente
assistant_printing = False  # suspende la línea [TÚ] mientras el asistente imprime



# Estado compartido
class SharedState:
    def __init__(self):
        self.is_speaking = False      # True solo al arrancar TTS

state      = SharedState()
speak_lock = asyncio.Lock()

# ───────────────────────────────────────────────────────────────────────────
# Utilidades de impresión sincronizada + colores
# ───────────────────────────────────────────────────────────────────────────
print_lock = threading.Lock()

# ANSI
RESET = "\x1b[0m"
BOLD  = "\x1b[1m"
DIM   = "\x1b[2m"
RED   = "\x1b[31m"
BLUE  = "\x1b[34m"
GRAY  = "\x1b[90m"

def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def tag(role: str, color: str, dim: bool=False) -> str:
    style = (DIM if dim else BOLD)
    return f"{style}{color}[{role}]{RESET}"

def clear_partial_line(width: int = 140):
    with print_lock:
        print("\r" + (" " * width) + "\r", end="", flush=True)

def sprint(msg: str = "", end: str = "\n"):
    with print_lock:
        print(msg, end=end, flush=True)

def render_user_partial(text: str, width: int = 120):
    # No re-pintar la línea del usuario si el asistente está imprimiendo o si la voz está sonando
    if suppress_user_partials or assistant_printing:
        return
    with print_lock:
        lbl = tag("TÚ", RED)
        print(f"\r{lbl}: {text:<{width}}", end="", flush=True)


# ⬇️ PÉGALO AQUÍ
def merge_with_overlap(prev: str, nxt: str, max_overlap: int = 24) -> str:
    """
    Une prev + nxt eliminando solapes (p.ej., 'Tiene el micrófono' + 'micrófono distante').
    Compara sufijo de prev con prefijo de nxt, sin sensibilidad a mayúsculas.
    """
    prev = prev.rstrip()
    nxt  = nxt.lstrip()
    max_len = min(max_overlap, len(prev), len(nxt))
    for k in range(max_len, 0, -1):
        if prev[-k:].lower() == nxt[:k].lower():
            return (prev + nxt[k:]).strip()
    # sin solape claro, mete un espacio si hace falta
    joiner = "" if (prev.endswith((" ", "\n")) or nxt.startswith((" ", "\n"))) else " "
    return (prev + joiner + nxt).strip()

async def fetch_tts_pcm(text: str, voice_id: str, voice_opts: dict, model_id: str = "eleven_multilingual_v2") -> bytes:
    """
    Descarga por completo el PCM de un trozo de texto usando ElevenLabs, devolviendo bytes.
    Sirve para prefetch del siguiente trozo y evitar huecos entre streams.
    """
    pcm_gen = elevenlabs_client.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model_id=model_id,
        output_format="pcm_16000",
        voice_settings=voice_opts,
        optimize_streaming_latency=1  # 1 = estable/rápido; cambia a 2 si la red es variable
    )
    buf = bytearray()
    async for chunk in pcm_gen:
        if chunk:
            buf.extend(chunk)
    return bytes(buf)


# ───────────────────────────────────────────────────────────────────────────
# 2) MICRÓFONO
# ───────────────────────────────────────────────────────────────────────────
def capture_microphone(loop, audio_queue, state):
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                     input=True, frames_per_buffer=CHUNK, start=True)
    sprint(c("🎤 Micrófono listo.", GRAY))
    try:
        while True:
            if not state.is_speaking:
                data = stream.read(CHUNK, exception_on_overflow=False)
                asyncio.run_coroutine_threadsafe(audio_queue.put(data), loop)
            else:
                time.sleep(0.005)
    finally:
        stream.stop_stream()
        stream.close()

# ───────────────────────────────────────────────────────────────────────────
# 3) PIPELINE LLM → TTS sincronizado (una sola pasada, sin desync)
# ───────────────────────────────────────────────────────────────────────────
TTS_MIN_CHARS       = 120   # antes 100
TTS_MAX_CHARS       = 300   # antes 260
TTS_TIME_FLUSH_MS   = 700   # antes 600

FIRST_CHUNK_TIME_FLUSH_MS = 260  # antes 280
FIRST_CHUNK_MIN_CHARS     = 18   # antes 16



async def speak_worker(tts_queue: asyncio.Queue, state):
    global out, FRAME_BYTES, TAIL_SILENCE_FRAMES

    VOICE_ID   = "IKne3meq5aSn9XLyUdCD"
    VOICE_OPTS = {
        "stability": 0.75,
        "similarity_boost": 0.75,
        "style": 0.45,
        "use_speaker_boost": True,
        "speed": 1.15,
    }
    MODEL_ID = "eleven_multilingual_v2"

    PREBUFFER_FIRST_MS = 160   # prebuffer solo al inicio
    LOW_WATER_MS       = 120   # cuando el buffer cae por debajo, “top-up” rápido
    def frames_for_ms(ms: int) -> int:
        return int(RATE * (ms / 1000.0)) * 2

    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=96)

    async def producer():
        first = True
        while True:
            text = await tts_queue.get()
            if text is None:
                break
            ol = 0 if first else 2  # 0 = arranque agresivo; 2 = robusto/estable
            pcm_gen = elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                output_format="pcm_16000",
                voice_settings=VOICE_OPTS,
                optimize_streaming_latency=ol
            )
            async for b in pcm_gen:
                if b:
                    await audio_q.put(b)
            first = False
        await audio_q.put(None)

    async def consumer():
        first_audio = False
        buf = bytearray()
        eof = False
        INITIAL = frames_for_ms(PREBUFFER_FIRST_MS)
        LOW     = frames_for_ms(LOW_WATER_MS)

        def write_from_buf():
            nonlocal buf, first_audio
            while len(buf) >= FRAME_BYTES:
                chunk = bytes(buf[:FRAME_BYTES]); del buf[:FRAME_BYTES]
                if not out.is_active():
                    out.start_stream()
                if not first_audio:
                    state.is_speaking = True
                    tts_started_event.set()
                    first_audio = True
                out.write(chunk)

        # prebuffer SOLO al principio
        while len(buf) < INITIAL and not eof:
            item = await audio_q.get()
            if item is None:
                eof = True; break
            buf.extend(item)
        write_from_buf()

        # luego, flujo continuo
        while True:
            if len(buf) >= FRAME_BYTES:
                write_from_buf()
            else:
                # intenta traer algo rápido sin bloquear
                pulled = False
                for _ in range(4):
                    try:
                        item = audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        eof = True; break
                    buf.extend(item); pulled = True
                    if len(buf) >= FRAME_BYTES:
                        break
                if not pulled:
                    await asyncio.sleep(0.004)

            if eof and len(buf) < FRAME_BYTES:
                if len(buf):
                    out.write(bytes(buf)); buf.clear()
                break

            if len(buf) < LOW and not eof:
                try:
                    item = await asyncio.wait_for(audio_q.get(), timeout=0.03)
                    if item is None:
                        eof = True
                    else:
                        buf.extend(item)
                except asyncio.TimeoutError:
                    pass

        out.write(b"\x00" * TAIL_SILENCE_FRAMES * 2)
        if out.is_active():
            out.stop_stream()
        state.is_speaking = False

    await asyncio.gather(asyncio.create_task(producer()),
                         asyncio.create_task(consumer()))


async def process_llm_and_speak(text: str, audio_queue, state):
    global conversation_history, assistant_printing

    clear_partial_line(); sprint()
    retrieved_context = await memory.get_context(text)

    augmented_prompt = (
        "Considera el siguiente contexto sobre mí. Úsalo únicamente si es directamente relevante. "
        "Si no lo es, ignóralo por completo.\n"
        "--- CONTEXTO ---\n"
        f"{retrieved_context}\n"
        "--- FIN DEL CONTEXTO ---\n\n"
        f"Pregunta del usuario: {text}"
    )

    messages_to_send = [
        {"role": "system",
         "content": ("Eres JARVIS, el asistente personal del señor, un ingeniero brillante con visión futurista. "
                     "Respondes con precisión, de forma corta (3-5 líneas máx), ingenio y respeto.")},
        *conversation_history,
        {"role": "user", "content": augmented_prompt},
    ]

    sprint(tag("ASISTENTE", BLUE) + ": ", end="")

    # cola para TTS
    tts_queue: asyncio.Queue = asyncio.Queue()
    speaker_task = asyncio.create_task(speak_worker(tts_queue, state))

    stream = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages_to_send,
        stream=True
    )

    # impresión por lotes para no bloquear el loop
    assistant_printing = True
    CONSOLE_FLUSH_MS = 35
    buf, console_buf = "", ""
    full_answer = ""
    last_flush = time.monotonic()
    first_token_time = None
    first_chunk_sent = False

    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        # acumula para consola
        console_buf += token
        now = time.monotonic()
        if any(ch in console_buf[-3:] for ch in ".!?…") or len(console_buf) >= 48 or (now-last_flush)*1000 >= CONSOLE_FLUSH_MS:
            sprint(c(console_buf, BLUE), end="")
            console_buf = ""
            last_flush = now

        # segmentación para TTS
        if first_token_time is None:
            first_token_time = now
        buf += token
        full_answer += token

        flush_piece = False
        if not first_chunk_sent:
            if (now - first_token_time)*1000 >= 260 and len(buf) >= 18:
                flush_piece = True
        else:
            if any(ch in buf[-3:] for ch in ".!?…") and len(buf) >= 120:
                flush_piece = True
            elif len(buf) >= 300:
                flush_piece = True
            elif (now - last_flush)*1000 >= 700 and len(buf) >= 24:
                flush_piece = True

        if flush_piece:
            await tts_queue.put(buf.strip())
            buf = ""
            first_chunk_sent = True

    if console_buf:
        sprint(c(console_buf, BLUE), end="")

    tail = buf.strip()
    if tail:
        await tts_queue.put(tail)

    await tts_queue.put(None)
    await speaker_task

    assistant_printing = False

    if full_answer.strip():
        conversation_history.append({"role": "user", "content": text})
        conversation_history.append({"role": "assistant", "content": full_answer})
        if len(conversation_history) > MAX_HISTORY_TURNS * 2:
            conversation_history[:] = conversation_history[-(MAX_HISTORY_TURNS * 2):]


# ───────────────────────────────────────────────────────────────────────────
# 3.b) PREVIEW DEL LLM (SOLO CONSOLA, SIN TTS) — (Paso 2)
# ───────────────────────────────────────────────────────────────────────────
async def preview_llm(text: str):
    global assistant_printing
    assistant_printing = True
    try:
        messages = [
            {"role": "system",
             "content": ("Eres JARVIS en modo previsualización. Da una respuesta preliminar muy breve "
                         "(1-2 líneas), sin conclusiones definitivas, hasta que llegue el mensaje final.")},
            {"role": "user", "content": text}
        ]
        sprint("\n" + tag("ASISTENTE·preview", BLUE, dim=True) + ": ", end="")
        stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        # imprime en bloques, no token a token
        buf, last = "", time.monotonic()
        FLUSH_MS = 40
        async for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if not content:
                continue
            buf += content
            now = time.monotonic()
            if any(ch in buf[-3:] for ch in ".!?…") or len(buf) >= 64 or (now-last)*1000 >= FLUSH_MS:
                sprint(c(buf, BLUE + DIM), end="")
                buf, last = "", now
        if buf:
            sprint(c(buf, BLUE + DIM), end="")
        sprint()
    except asyncio.CancelledError:
        raise
    finally:
        assistant_printing = False


# ───────────────────────────────────────────────────────────────────────────
# 3.c) CONSUMIDOR DE COLA DE PARCIALES → LANZA/CANCELA PREVIEW
# ───────────────────────────────────────────────────────────────────────────
async def llm_streamer(loop):
    global current_preview_task, grace_mode

    MIN_MS_BETWEEN_PREVIEWS = 600
    MIN_PREVIEW_CHARS = 18
    last_preview_ts = 0.0
    last_text = ""

    def looks_like_short_command(t: str) -> bool:
        words = t.strip().split()
        if len(words) <= 8 and words:
            first = words[0].lower()
            return first in {
                "abre","pon","pausa","reanuda","busca","cierra","enciende",
                "apaga","ve","ir","muestra","llama","escribe","play","stop"
            }
        return False

    while True:
        partial_text = await llm_text_queue.get()
        if grace_mode:
            continue
        if len(partial_text) < MIN_PREVIEW_CHARS:
            continue
        if looks_like_short_command(partial_text):
            continue
        if partial_text == last_text:
            continue
        now = time.monotonic()
        if (now - last_preview_ts) * 1000 < MIN_MS_BETWEEN_PREVIEWS:
            continue
        last_preview_ts = now
        last_text = partial_text

        if current_preview_task and not current_preview_task.done():
            current_preview_task.cancel()
            try:
                await current_preview_task
            except asyncio.CancelledError:
                pass
        current_preview_task = loop.create_task(preview_llm(partial_text))

# ───────────────────────────────────────────────────────────────────────────
# 4) TRANSCRIPCIÓN DEEPGRAM: PARCIALES + FINAL CON GRACIA (idéntico al tuyo con fixes)
# ───────────────────────────────────────────────────────────────────────────
async def transcribe_audio(loop, audio_queue, state):
    """
    Deepgram en streaming con:
      - Parciales con debounce + primer parcial rápido.
      - Finales con coalescing + tail-merge (con deduplicación de solape).
      - Ventana de gracia que PAUSA previews.
      - Reset de estado al enviar la respuesta final.
      - Consola: usuario en rojo.
    """
    global grace_mode, suppress_user_partials  # ⬅️ añadimos el global aquí

    ENDPOINTING_MS       = 900
    FINAL_GRACE_MS       = 900
    MIN_FINAL_CHARS      = 12
    PARTIAL_MIN_CHARS    = 4
    PARTIAL_DEBOUNCE_MS  = 90

    DEEPGRAM_URL = (
        "wss://api.deepgram.com/v1/listen?"
        f"encoding=linear16&sample_rate={RATE}&channels=1"
        "&language=es&smart_format=true"
        "&interim_results=true"
        f"&endpointing={ENDPOINTING_MS}"
    )

    last_partial_ts    = 0.0
    last_partial_text  = ""
    first_partial_done = False

    pending_final_text   = ""
    last_final_time_mono = 0.0
    finalize_task        = None
    final_seq            = 0

    def print_user_final_line(text: str):
        lbl = tag("TÚ", RED)
        sprint(f"\n{lbl}: {text}\n")

    async def finalize_if_quiet(my_seq: int):
        nonlocal pending_final_text, last_final_time_mono, finalize_task
        nonlocal first_partial_done, last_partial_ts, last_partial_text, final_seq
        global grace_mode, suppress_user_partials, current_llm_task  # ⬅️ aquí sí global

        await asyncio.sleep(FINAL_GRACE_MS / 1000.0)
        if my_seq != final_seq:
            finalize_task = None; return
        if (time.monotonic() - last_final_time_mono) * 1000 < FINAL_GRACE_MS:
            finalize_task = None; return

        text = pending_final_text.strip()
        if not text:
            finalize_task = None; return
        if not (len(text) >= MIN_FINAL_CHARS or text.endswith((".", "?", "!", "…"))):
            finalize_task = None; return

        # cancelar preview activo
        if current_preview_task and not current_preview_task.done():
            current_preview_task.cancel()
            try:
                await current_preview_task
            except asyncio.CancelledError:
                pass

        # ⬇️ bien alineado (antes estaba mal indentado)
        clear_partial_line()
        print_user_final_line(text)

        # preparar señal de arranque de audio
        tts_started_event.clear()

        # Lanza la respuesta final (con TTS por trozos) y bloquea parciales de [TÚ]
        suppress_user_partials = True
        current_llm_task = loop.create_task(process_llm_and_speak(text, audio_queue, state))

        # Cuando termine la respuesta, volvemos a permitir parciales
        async def _release_after_assistant():
            global suppress_user_partials   # ⬅️ era nonlocal, debe ser global
            try:
                await current_llm_task
            finally:
                suppress_user_partials = False
        asyncio.create_task(_release_after_assistant())

        # Mantén 'grace_mode' hasta que arranque el TTS (evita que [TÚ] pise a [ASISTENTE])
        async def _drop_grace_on_tts_start():
            global grace_mode
            try:
                await asyncio.wait_for(tts_started_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            grace_mode = False
        asyncio.create_task(_drop_grace_on_tts_start())

        # RESET de estado del turno
        pending_final_text = ""
        first_partial_done = False
        last_partial_text = ""
        last_partial_ts = 0.0
        final_seq = 0
        finalize_task = None

    async with websockets.connect(
        DEEPGRAM_URL,
        additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    ) as ws:
        sprint(c("🟢 Conectado a Deepgram. ¡Habla ahora!", GRAY))

        async def sender():
            # Warmup 100 ms de silencio
            try:
                await ws.send(b"\x00" * int(RATE * 0.1) * 2)
            except Exception:
                pass
            KEEPALIVE_EVERY = 3.0
            while True:
                try:
                    data = await asyncio.wait_for(audio_queue.get(), timeout=KEEPALIVE_EVERY)
                    await ws.send(data)
                except asyncio.TimeoutError:
                    await ws.send('{"type":"KeepAlive"}')
                    await asyncio.sleep(0.05)

        async def receiver():
            nonlocal last_partial_ts, last_partial_text, first_partial_done
            nonlocal pending_final_text, last_final_time_mono, finalize_task, final_seq
            global grace_mode

            async for msg in ws:
                res  = json.loads(msg)
                alt  = res.get("channel", {}).get("alternatives", [{}])[0]
                trans= alt.get("transcript", "") or ""
                if not trans:
                    continue

                if res.get("is_final", False):
                    grace_mode = True
                    pending_final_text = (
                        merge_with_overlap(pending_final_text, trans)
                        if pending_final_text else trans.strip()
                    )
                    last_final_time_mono = time.monotonic()

                    final_seq += 1
                    my_seq = final_seq
                    finalize_task = asyncio.create_task(finalize_if_quiet(my_seq))

                else:
                    now = time.monotonic()

                    # Tail-merge durante gracia con deduplicación
                    if grace_mode and (now - last_final_time_mono) * 1000 < FINAL_GRACE_MS:
                        pending_final_text = merge_with_overlap(pending_final_text, trans)
                        last_final_time_mono = now
                        continue  # no preview en gracia

                    # Primer parcial sin debounce
                    if not first_partial_done:
                        if suppress_user_partials:
                            continue
                        render_user_partial(trans, width=120)
                        first_partial_done = True
                        last_partial_text = trans
                        last_partial_ts = now
                        if not grace_mode:
                            try:
                                llm_text_queue.put_nowait(trans)
                            except asyncio.QueueFull:
                                try: _ = llm_text_queue.get_nowait()
                                except Exception: pass
                                await llm_text_queue.put(trans)
                        continue

                    # Siguientes parciales con debounce
                    if len(trans) >= PARTIAL_MIN_CHARS and (now - last_partial_ts) * 1000 >= PARTIAL_DEBOUNCE_MS:
                        if trans != last_partial_text:
                            if suppress_user_partials:
                                continue
                            render_user_partial(trans, width=120)
                            last_partial_text = trans
                            last_partial_ts = now
                            if not grace_mode:
                                try:
                                    llm_text_queue.put_nowait(trans)
                                except asyncio.QueueFull:
                                    try: _ = llm_text_queue.get_nowait()
                                    except Exception: pass
                                    await llm_text_queue.put(trans)

        await asyncio.gather(sender(), receiver())


# ───────────────────────────────────────────────────────────────────────────
# 5) MAIN
# ───────────────────────────────────────────────────────────────────────────
_streamer_started = False  # lanzamos el streamer una sola vez

async def main():
    global _streamer_started

    audio_queue = asyncio.Queue()
    loop        = asyncio.get_running_loop()

    # Hilo de micrófono
    mic_thread = threading.Thread(
        target=capture_microphone,
        args=(loop, audio_queue, state),
        daemon=True
    )
    mic_thread.start()

    # Consumidor de parciales para preview del LLM (una sola vez)
    if not _streamer_started:
        loop.create_task(llm_streamer(loop))
        _streamer_started = True

    await transcribe_audio(loop, audio_queue, state)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sprint(c("\n\n👋 Asistente detenido.", GRAY))
    finally:
        try:
            if out.is_active():
                out.stop_stream()
            out.close()
        except Exception:
            pass
        pa.terminate()
