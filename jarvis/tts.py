# jarvis/tts.py
import asyncio

# Reproduce audio continuo con baja latencia:
# - 1er trozo: streaming agresivo (latencia mínima)
# - Siguientes: streaming robusto + búfer en consumidor (evita microcortes)
async def speak_worker(cfg, state, tts_queue: asyncio.Queue):
    """
    cfg: Config (jarvis.config.load_config)
    state: AppState (jarvis.state.make_state)
    tts_queue: cola de trozos de texto (str) a sintetizar. Termina con None.
    """
    VOICE_ID   = cfg.VOICE_ID
    MODEL_ID   = cfg.ELEVEN_MODEL_ID
    VOICE_OPTS = {
        "stability": 0.75,
        "similarity_boost": 0.75,
        "style": 0.45,
        "use_speaker_boost": True,
        "speed": 1.15,
    }

    PREBUFFER_FIRST_MS = 160   # prebuffer antes de soltar el primer audio
    LOW_WATER_MS       = 120   # cuando el buffer cae por debajo, rellenamos rápido

    def frames_for_ms(ms: int) -> int:
        # 16 kHz * 16-bit mono (2 bytes)
        return int(cfg.RATE * (ms / 1000.0)) * 2

    # Cola interna de BYTES ya listos para reproducir
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=96)

    async def producer():
        """Convierte texto → bytes PCM y los encola en audio_q."""
        first = True
        while True:
            text = await tts_queue.get()
            if text is None:
                break

            # optimize_streaming_latency:
            # 0 = agresivo (arranca YA), 1 = rápido estable, 2 = más robusto (más búfer servidor)
            ol = 0 if first else 2

            pcm_gen = cfg.eleven_client.text_to_speech.stream(
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

        await audio_q.put(None)  # sentinel

    async def consumer():
        """Lee bytes de audio_q, hace prebuffer y reproduce continuo con PyAudio."""
        first_audio = False
        buf = bytearray()
        eof = False
        INITIAL = frames_for_ms(PREBUFFER_FIRST_MS)
        LOW     = frames_for_ms(LOW_WATER_MS)

        out = getattr(cfg, "out", None)

        def out_is_active() -> bool:
            try:
                return (out is not None and hasattr(out, "is_active") and out.is_active())
            except Exception:
                return False

        def out_start_safe():
            if out is None:
                return
            try:
                # Si el stream no está activo, intenta arrancarlo
                if hasattr(out, "start_stream") and not out_is_active():
                    out.start_stream()
            except Exception:
                # Si no se puede arrancar, no reventamos el flujo
                pass

        def out_write_safe(data: bytes) -> bool:
            """Devuelve True si escribió, False si no se pudo (stream parado o error)."""
            if out is None:
                return False
            try:
                if not out_is_active():
                    out_start_safe()
                if not out_is_active():
                    return False
                out.write(data)
                return True
            except Exception:
                return False

        def write_from_buf():
            nonlocal buf, first_audio
            # Escribe por frames completos
            while len(buf) >= cfg.FRAME_BYTES:
                chunk = bytes(buf[:cfg.FRAME_BYTES])
                del buf[:cfg.FRAME_BYTES]
                # Arranca stream si es necesario y escribe con salvaguarda
                if not out_write_safe(chunk):
                    # Si no podemos escribir (stream parado), abortamos consumo
                    return False
                if not first_audio:
                    state.shared_state.is_speaking = True
                    state.tts_started_event.set()
                    first_audio = True
            return True

        # Prebuffer SOLO al principio
        while len(buf) < INITIAL and not eof:
            item = await audio_q.get()
            if item is None:
                eof = True
                break
            buf.extend(item)

        if not write_from_buf():
            # No se pudo escribir (stream parado), terminamos consumidor
            eof = True

        # Flujo continuo
        while True:
            if len(buf) >= cfg.FRAME_BYTES:
                if not write_from_buf():
                    # No se pudo escribir (stream parado)
                    eof = True
            else:
                # intenta traer sin bloquear
                pulled = False
                for _ in range(4):
                    try:
                        item = audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        eof = True
                        break
                    buf.extend(item)
                    pulled = True
                    if len(buf) >= cfg.FRAME_BYTES:
                        break
                if not pulled:
                    await asyncio.sleep(0.004)

            # Fin natural
            if eof and len(buf) < cfg.FRAME_BYTES:
                if len(buf):
                    # Últimos bytes residuales
                    out_write_safe(bytes(buf))
                    buf.clear()
                break

            # Low-water: si el buffer baja, esperamos un “top-up” corto
            if len(buf) < LOW and not eof:
                try:
                    item = await asyncio.wait_for(audio_q.get(), timeout=0.03)
                    if item is None:
                        eof = True
                    else:
                        buf.extend(item)
                except asyncio.TimeoutError:
                    pass

        # Coda de silencio y cierre (solo si el stream sigue activo)
        try:
            if out_is_active():
                out_write_safe(b"\x00" * (cfg.TAIL_SILENCE_FRAMES * 2))
                if hasattr(out, "stop_stream") and out.is_active():
                    out.stop_stream()
        except Exception:
            pass

        state.shared_state.is_speaking = False

    # Ejecutar productor y consumidor en paralelo (robusto ante excepciones internas)
    prod = asyncio.create_task(producer())
    cons = asyncio.create_task(consumer())
    try:
        await asyncio.gather(prod, cons, return_exceptions=True)
    except asyncio.CancelledError:
        # Cancelación limpia (barge-in/cierre)
        pass
