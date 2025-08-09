# jarvis/llm.py
# LLM streaming + segmentación para TTS + previews

from __future__ import annotations
import asyncio
import time
from typing import Optional
import os
from openai import OpenAI

from jarvis.console import sprint, c, tag, BLUE, DIM
from jarvis.memory_store import summarize_and_store
from jarvis.state import history_append, AppState
from jarvis.tts import speak_worker

# ---- Parámetros de segmentación / consola ----
TTS_MIN_CHARS            = 120
TTS_MAX_CHARS            = 300
TTS_TIME_FLUSH_MS        = 700
FIRST_CHUNK_TIME_MS      = 260
FIRST_CHUNK_MIN_CHARS    = 18
CONSOLE_FLUSH_MS         = 35

# ---- Previews (desde cola de parciales) ----
MIN_MS_BETWEEN_PREVIEWS  = 600
MIN_PREVIEW_CHARS        = 18

# Cliente OpenAI directo para selección dinámica de modelo
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _sanitize_messages(raw_msgs):
    """Deja solo las claves válidas para el API: role y content.
       Filtra cualquier mensaje que no sea system/user/assistant o que no tenga string content.
    """
    clean = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content})
    return clean


async def _preview_llm(cfg, state: AppState, text: str):
    """Preview ultrarrápido (sin stream) con timeout corto, usando GPT-5 nano.
       Evita encabezados vacíos y llaves extra en mensajes.
    """
    state.assistant_printing = True
    TIMEOUT_S = 0.45
    try:
        msgs = _sanitize_messages([
            {
                "role": "system",
                "content": ("Eres JARVIS en modo previsualización. Responde con una frase breve, "
                            "sin conclusiones definitivas.")
            },
            {"role": "user", "content": text},
        ])

        sprint("\n" + tag("ASISTENTE·preview", BLUE, dim=True) + ": ", end="")

        async def _ask_once():
            # Responses API no-stream (más estable con GPT-5)
            return await cfg.openai_client.responses.create(
                model="gpt-5-nano",
                input=msgs,
            )

        content = ""
        try:
            resp = await asyncio.wait_for(_ask_once(), timeout=TIMEOUT_S)
            content = (getattr(resp, "output_text", "") or "").strip()
        except asyncio.TimeoutError:
            content = ""

        if content:
            line = content.splitlines()[0].strip()
            if len(line) < 40:
                parts = [p.strip() for p in content.splitlines() if p.strip()]
                if len(parts) >= 2:
                    line = (line + " " + parts[1]).strip()
            if len(line) > 140:
                line = line[:140].rstrip() + "…"
            sprint(c(line, BLUE + DIM), end="")
        sprint()

    except asyncio.CancelledError:
        sprint()
        raise
    finally:
        state.assistant_printing = False





def _looks_like_short_command(t: str) -> bool:
    words = t.strip().split()
    if len(words) <= 8 and words:
        first = words[0].lower()
        return first in {
            "abre", "pon", "pausa", "reanuda", "busca", "cierra", "enciende",
            "apaga", "ve", "ir", "muestra", "llama", "escribe", "play", "stop"
        }
    return False


async def llm_streamer(cfg, state: AppState, loop):
    """Lee parciales de la cola y lanza/cancela previews."""
    last_preview_ts = 0.0
    last_text = ""

    while True:
        partial_text = await state.llm_text_queue.get()

        if state.grace_mode:
            continue
        if len(partial_text) < MIN_PREVIEW_CHARS:
            continue
        if _looks_like_short_command(partial_text):
            continue
        if partial_text == last_text:
            continue

        now = time.monotonic()
        if (now - last_preview_ts) * 1000 < MIN_MS_BETWEEN_PREVIEWS:
            continue
        last_preview_ts = now
        last_text = partial_text

        if state.current_preview_task and not state.current_preview_task.done():
            state.current_preview_task.cancel()
            try:
                await state.current_preview_task
            except asyncio.CancelledError:
                pass

        state.current_preview_task = loop.create_task(_preview_llm(cfg, state, partial_text))


async def process_llm_and_speak(cfg, state: AppState, text: str, _audio_queue=None):
    """
    Stream LLM -> consola + TTS.
    - GPT-5 (mini/nano): Responses API (stream de eventos) + mensajes saneados.
    - GPT-4.x: Chat Completions (como antes).
    """
    # ---------- Contexto ----------
    kb_ctx = ""
    try:
        kb_ctx = await state.kb.get_context(text)
    except Exception:
        kb_ctx = ""

    N = 2
    turns = [t for t in state.full_history if t.get("role") in ("user", "assistant")]
    recent = "\n".join(f"{t['role'].upper()}: {t.get('content','')}" for t in turns[-(N * 2):])

    ctx_parts = []
    if kb_ctx:
        ctx_parts.append("## KB PRIVADA\n" + kb_ctx)
    if recent:
        ctx_parts.append("## ÚLTIMOS TURNOS (RESUMEN CRUDO)\n" + recent)
    ctx = "\n\n".join(ctx_parts).strip()

    base_msgs = [
        {
            "role": "system",
            "content": (
                "Eres JARVIS, el asistente personal del señor, un ingeniero brillante con visión futurista. "
                "Respondes con precisión, de forma corta (3–5 líneas), ingenio y respeto. "
                "Refiérete al usuario como 'señor'. "
                "Usa el contexto SOLO si es relevante. "
                f"--- CONTEXTO ---\n{ctx}\n--- FIN DEL CONTEXTO ---"
            ),
        },
        *state.conversation_history,   # puede traer 'ts' u otras claves → SANEAR
        {"role": "user", "content": text},
    ]
    msgs = _sanitize_messages(base_msgs)

    # ---------- Prefijo consola ----------
    sprint(tag("ASISTENTE", BLUE) + ": ", end="")

    # ---------- TTS ----------
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
    speaker_task = asyncio.create_task(speak_worker(cfg, state, tts_queue))

    # ---------- Selección de modelo ----------
    if len(text.strip()) < 40 or _looks_like_short_command(text):
        model_name = "gpt-5-nano"
    else:
        model_name = "gpt-5-mini"

    # ---------- Estados de segmentación / impresión ----------
    state.assistant_printing = True
    buf = ""
    console_buf = ""
    full_answer = ""
    first_token_time: Optional[float] = None
    first_chunk_sent = False
    last_flush_for_tts = time.monotonic()
    last_console_flush = time.monotonic()

    async def _flush_console(now):
        nonlocal console_buf, last_console_flush
        if console_buf:
            sprint(c(console_buf, BLUE), end="")
            console_buf = ""
            last_console_flush = now

    # ---------- Stream según endpoint ----------
    if model_name.startswith("gpt-5"):
        # GPT-5: Responses API (estable en streaming). ¡NO pasar claves extra!
        stream = await cfg.openai_client.responses.create(
            model=model_name,
            input=msgs,
            stream=True,
        )
        async for event in stream:
            et = getattr(event, "type", "")
            if et == "response.output_text.delta":
                token = getattr(event, "delta", "") or ""
                if not token:
                    continue

                # consola por lotes
                console_buf += token
                now = time.monotonic()
                if (any(ch in console_buf[-3:] for ch in ".!?…")
                        or len(console_buf) >= 48
                        or (now - last_console_flush) * 1000 >= CONSOLE_FLUSH_MS):
                    await _flush_console(now)

                # TTS
                if first_token_time is None:
                    first_token_time = now
                buf += token
                full_answer += token

                flush_piece = False
                if not first_chunk_sent:
                    if (now - first_token_time) * 1000 >= FIRST_CHUNK_TIME_MS and len(buf) >= FIRST_CHUNK_MIN_CHARS:
                        flush_piece = True
                else:
                    if any(ch in buf[-3:] for ch in ".!?…") and len(buf) >= TTS_MIN_CHARS:
                        flush_piece = True
                    elif len(buf) >= TTS_MAX_CHARS:
                        flush_piece = True
                    elif (now - last_flush_for_tts) * 1000 >= TTS_TIME_FLUSH_MS and len(buf) >= 24:
                        flush_piece = True

                if flush_piece:
                    await tts_queue.put(buf.strip())
                    buf = ""
                    first_chunk_sent = True
                    last_flush_for_tts = now

            elif et == "response.completed":
                break
            elif et == "response.error":
                err_msg = getattr(event, "error", None)
                if err_msg:
                    sprint(c(f"\n[error LLM]: {err_msg}", BLUE), end="")
                break

    else:
        # GPT-4.x: Chat Completions (mantén tu lógica anterior)
        stream = await cfg.openai_client.chat.completions.create(
            model=model_name,
            messages=msgs,             # ya saneados
            stream=True,
            max_tokens=300,
            temperature=0.7
        )
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            console_buf += token
            now = time.monotonic()
            if (any(ch in console_buf[-3:] for ch in ".!?…")
                    or len(console_buf) >= 48
                    or (now - last_console_flush) * 1000 >= CONSOLE_FLUSH_MS):
                await _flush_console(now)

            if first_token_time is None:
                first_token_time = now
            buf += token
            full_answer += token

            flush_piece = False
            if not first_chunk_sent:
                if (now - first_token_time) * 1000 >= FIRST_CHUNK_TIME_MS and len(buf) >= FIRST_CHUNK_MIN_CHARS:
                    flush_piece = True
            else:
                if any(ch in buf[-3:] for ch in ".!?…") and len(buf) >= TTS_MIN_CHARS:
                    flush_piece = True
                elif len(buf) >= TTS_MAX_CHARS:
                    flush_piece = True
                elif (now - last_flush_for_tts) * 1000 >= TTS_TIME_FLUSH_MS and len(buf) >= 24:
                    flush_piece = True

            if flush_piece:
                await tts_queue.put(buf.strip())
                buf = ""
                first_chunk_sent = True
                last_flush_for_tts = now

    # Volcados finales
    now = time.monotonic()
    await _flush_console(now)

    tail = buf.strip()
    if tail:
        await tts_queue.put(tail)

    await tts_queue.put(None)
    await speaker_task
    state.assistant_printing = False

    # ---------- Persistencia ----------
    await history_append(state, "assistant", full_answer)
    state.conversation_history.append({"role": "user", "content": text})
    state.conversation_history.append({"role": "assistant", "content": full_answer})
    await summarize_and_store(cfg, state, text, full_answer)
