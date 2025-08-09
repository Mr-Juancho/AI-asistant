# jarvis/asr.py
import asyncio, json, time, websockets
from jarvis.console import c, tag, sprint, clear_partial_line, render_user_partial, GRAY, RED
from jarvis.llm import process_llm_and_speak
from jarvis.state import history_append

# Utilidad para fusionar finales con solapes (evita repeticiones al “tail-merge”)
def merge_with_overlap(prev: str, nxt: str, max_overlap: int = 24) -> str:
    prev = (prev or "").rstrip()
    nxt  = (nxt or "").lstrip()
    if not prev:
        return nxt
    max_len = min(max_overlap, len(prev), len(nxt))
    for k in range(max_len, 0, -1):
        if prev[-k:].lower() == nxt[:k].lower():
            return (prev + nxt[k:]).strip()
    joiner = "" if (prev.endswith((" ", "\n")) or nxt.startswith((" ", "\n"))) else " "
    return (prev + joiner + nxt).strip()

async def transcribe_audio(cfg, state, loop, audio_queue: asyncio.Queue):
    """
    Deepgram en streaming con:
      - Parciales con debounce + primer parcial rápido.
      - Finales con coalescing + tail-merge (deduplicación de solape).
      - Ventana de gracia que PAUSA previews.
      - Persistencia de usuario/assistant (se hace en LLM).
      - Consola: usuario en rojo, asistente en azul (en llm).
    """
    ENDPOINTING_MS       = 900
    FINAL_GRACE_MS       = 900
    MIN_FINAL_CHARS      = 12
    PARTIAL_MIN_CHARS    = 4
    PARTIAL_DEBOUNCE_MS  = 90

    DEEPGRAM_URL = (
        "wss://api.deepgram.com/v1/listen?"
        f"encoding=linear16&sample_rate={cfg.RATE}&channels=1"
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

        await asyncio.sleep(FINAL_GRACE_MS / 1000.0)

        # otro final más nuevo lo invalida
        if my_seq != final_seq:
            finalize_task = None
            return

        # todavía dentro de la ventana de gracia → no cierres
        if (time.monotonic() - last_final_time_mono) * 1000 < FINAL_GRACE_MS:
            finalize_task = None
            return

        text = (pending_final_text or "").strip()
        if not text:
            finalize_task = None
            return
        if not (len(text) >= MIN_FINAL_CHARS or text.endswith((".", "?", "!", "…"))):
            finalize_task = None
            return

        # Cancela preview activo si lo hay
        if getattr(state, "current_preview_task", None) and not state.current_preview_task.done():
            state.current_preview_task.cancel()
            try:
                await state.current_preview_task
            except asyncio.CancelledError:
                pass

        # Limpieza de consola y pinta línea de usuario
        clear_partial_line()
        print_user_final_line(text)

        # Persistir turno de usuario
        await history_append(state, "user", text)

        # Preparar señal de arranque de TTS
        state.tts_started_event.clear()

        # No pintes [TÚ] durante el habla del asistente
        state.suppress_user_partials = True

        # Lanza respuesta final (LLM→TTS)
        task = loop.create_task(process_llm_and_speak(cfg, state, text, audio_queue))

        # Al terminar, vuelve a permitir parciales
        async def _release_after_assistant():
            try:
                await task
            finally:
                state.suppress_user_partials = False
        loop.create_task(_release_after_assistant())

        # Mantén grace_mode hasta que arranque el TTS (evita solapes con [TÚ])
        async def _drop_grace_on_tts_start():
            try:
                await asyncio.wait_for(state.tts_started_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            state.grace_mode = False
        loop.create_task(_drop_grace_on_tts_start())

        # RESET de estado del turno
        state.grace_mode = True  # todavía en gracia hasta que arranque TTS
        pending_final_text = ""
        first_partial_done = False
        last_partial_text = ""
        last_partial_ts = 0.0
        final_seq = 0
        finalize_task = None

    # Conexión WS
    async with websockets.connect(
        DEEPGRAM_URL,
        additional_headers={"Authorization": f"Token {cfg.DEEPGRAM_API_KEY}"}
    ) as ws:
        sprint(c("🟢 Conectado a Deepgram. ¡Habla ahora!", GRAY))

        # Sender: envía frames del micro + keepalive
        async def sender():
            # Warmup ~100 ms silencio
            try:
                await ws.send(b"\x00" * int(cfg.RATE * 0.1) * 2)
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

        # Receiver: procesa parciales y finales
        async def receiver():
            nonlocal last_partial_ts, last_partial_text, first_partial_done
            nonlocal pending_final_text, last_final_time_mono, finalize_task, final_seq

            async for msg in ws:
                res  = json.loads(msg)
                alt  = res.get("channel", {}).get("alternatives", [{}])[0]
                trans= alt.get("transcript", "") or ""
                if not trans:
                    continue

                if res.get("is_final", False):
                    state.grace_mode = True

                    # Coalesce finales con deduplicación por solape
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

                    # Tail-merge dentro de la gracia (dedup)
                    if state.grace_mode and (now - last_final_time_mono) * 1000 < FINAL_GRACE_MS:
                        pending_final_text = merge_with_overlap(pending_final_text, trans)
                        last_final_time_mono = now
                        continue  # no preview en gracia

                    # Primer parcial SIN debounce
                    if not first_partial_done:
                        if state.suppress_user_partials or state.assistant_printing:
                            continue
                        render_user_partial(state, trans, width=120)
                        first_partial_done = True
                        last_partial_text = trans
                        last_partial_ts = now

                        # empuja al streamer de previews
                        if not state.grace_mode:
                            try:
                                state.llm_text_queue.put_nowait(trans)
                            except asyncio.QueueFull:
                                try:
                                    _ = state.llm_text_queue.get_nowait()
                                except Exception:
                                    pass
                                await state.llm_text_queue.put(trans)
                        continue

                    # Siguientes parciales con debounce
                    if len(trans) >= PARTIAL_MIN_CHARS and (now - last_partial_ts) * 1000 >= PARTIAL_DEBOUNCE_MS:
                        if trans != last_partial_text:
                            if state.suppress_user_partials or state.assistant_printing:
                                continue
                            render_user_partial(state, trans, width=120)
                            last_partial_text = trans
                            last_partial_ts = now
                            if not state.grace_mode:
                                try:
                                    state.llm_text_queue.put_nowait(trans)
                                except asyncio.QueueFull:
                                    try:
                                        _ = state.llm_text_queue.get_nowait()
                                    except Exception:
                                        pass
                                    await state.llm_text_queue.put(trans)

        await asyncio.gather(sender(), receiver())
