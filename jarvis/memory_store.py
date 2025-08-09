# jarvis/memory_store.py
# Memoria semántica sencilla + resúmenes de turno
# - Lee tu KB local (myKnowledgeBase/*.txt|*.md|*.json)
# - Devuelve contexto relevante para el prompt
# - Después de cada respuesta, resume y persiste {"role":"summary", ...}

from __future__ import annotations
import os, json, re, asyncio, glob
from dataclasses import dataclass
from typing import List, Optional, Tuple

from jarvis.console import c, GRAY
from jarvis.state import history_append, HISTORY_FILE, AppState  # usa el mismo historial

# Carpeta con notas/conocimientos del usuario
KB_DIR = "myKnowledgeBase"

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def _flatten_json(obj, prefix="") -> str:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(_flatten_json(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.append(_flatten_json(v, f"{prefix}{i}."))
    else:
        out.append(f"{prefix}{obj}")
    return "\n".join([x for x in out if x])

def _read_json_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _flatten_json(data)
    except Exception:
        return _read_text_file(path)  # fallback

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()

def _score(query: str, text: str) -> float:
    """Scoring muy simple (coincidencias de palabras + boost por frase exacta)."""
    q = _normalize(query)
    t = _normalize(text)
    if not q or not t:
        return 0.0
    words = [w for w in re.split(r"[^\wáéíóúñü]+", q) if w]
    hits = sum(1 for w in words if w in t)
    phrase = 3.0 if q in t else 0.0
    return hits + phrase

@dataclass
class KBItem:
    name: str
    text: str

# ------------------------------------------------------------
# Memory Manager
# ------------------------------------------------------------
class MemoryManager:
    def __init__(self, kb_dir: str = KB_DIR, include_recent_turns: int = 2):
        self.kb_dir = kb_dir
        self.include_recent_turns = include_recent_turns
        self._kb_cache: List[KBItem] = []
        self._kb_loaded = False

    # ---------- Carga KB ----------
    def _load_kb(self):
        if self._kb_loaded:
            return
        items: List[KBItem] = []

        if os.path.isdir(self.kb_dir):
            # txt / md
            for path in glob.glob(os.path.join(self.kb_dir, "*.txt")) + \
                        glob.glob(os.path.join(self.kb_dir, "*.md")):
                txt = _read_text_file(path)
                if txt.strip():
                    items.append(KBItem(os.path.basename(path), txt))

            # json (se aplana)
            for path in glob.glob(os.path.join(self.kb_dir, "*.json")):
                # No dupliques conversation_history.json en la KB de contexto
                if os.path.basename(path) == os.path.basename(HISTORY_FILE):
                    continue
                txt = _read_json_file(path)
                if txt.strip():
                    items.append(KBItem(os.path.basename(path), txt))

        self._kb_cache = items
        self._kb_loaded = True

    # ---------- Recupera últimos turnos ----------
    async def _read_recent_turns(self, state: Optional[AppState], n_pairs: int) -> str:
        """
        Devuelve los últimos n_pairs pares (user+assistant) del historial.
        Usa el 'state' si está disponible; si no, lee directo de HISTORY_FILE.
        """
        turns: List[dict] = []
        if state and getattr(state, "full_history", None):
            turns = [t for t in state.full_history if t.get("role") in ("user", "assistant")]
        else:
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                turns = [t for t in data if t.get("role") in ("user", "assistant")]
            except Exception:
                turns = []

        if not turns:
            return ""

        snippet = []
        # últimos n_pairs*2 elementos
        for t in turns[-(n_pairs*2):]:
            role = t.get("role")
            content = t.get("content", "")
            snippet.append(f"{role.upper()}: {content}")

        return "\n".join(snippet)

    # ---------- Contexto para el prompt ----------
    async def get_context(
        self,
        query: str,
        *,
        state: Optional[AppState] = None,
        top_k: int = 3,
        max_chars: int = 1200,
    ) -> str:
        """
        Devuelve contexto de la KB + (opcional) últimos turnos.
        No bloqueante; imprime logs ligeros.
        """
        self._load_kb()

        # rank por score
        scored: List[Tuple[float, KBItem]] = []
        for it in self._kb_cache:
            s = _score(query, it.text)
            if s > 0:
                scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked = [it for _, it in scored[:top_k]]

        parts: List[str] = []
        # Troceo por cuota
        used = 0
        for it in picked:
            quota = max(200, max_chars // max(1, len(picked)))
            snippet = it.text.strip()[:quota]
            parts.append(f"## {it.name}\n{snippet}")
            used += len(snippet)
            if used >= max_chars:
                break

        # Últimos turnos
        if self.include_recent_turns > 0:
            recent = await self._read_recent_turns(state, self.include_recent_turns)
            if recent:
                parts.append("\n## ÚLTIMOS TURNOS (RESUMEN CRUDO)\n" + recent)

        ctx = ("\n\n").join(parts).strip()

        # Log amigable
        if ctx:
            print(c(f"🔍 Buscando en la memoria sobre: '{query}'", GRAY))
            print(c("✅ Contexto encontrado.", GRAY))
        else:
            print(c(f"🔍 Buscando en la memoria sobre: '{query}'", GRAY))
            print(c("⚪ Sin contexto relevante.", GRAY))
        return ctx

# ------------------------------------------------------------
# Resumen de turno y persistencia como "summary"
# ------------------------------------------------------------
async def summarize_and_store(cfg, state: AppState, user_text: str, assistant_text: str):
    """
    Resume el turno (usuario + asistente) y guarda una entrada:
      { "role": "summary", "content": "<JSON>", "ts": ... }
    en conversation_history.json (misma ruta de HISTORY_FILE).
    """
    system = (
        "Eres un asistente que devuelve SOLO JSON válido. "
        "Resume breve el turno y extrae hechos clave. "
        "Esquema:\n"
        "{"
        "\"summary\": string, "
        "\"topics\": [string], "
        "\"facts\": [{\"key\": string, \"value\": string, \"confidence\": number}]"
        "}"
    )
    user = (
        f"Usuario dijo: {user_text}\n"
        f"Asistente respondió: {assistant_text}\n"
        "Devuelve ÚNICAMENTE el JSON del esquema. Sin texto extra."
    )

    try:
        resp = await cfg.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except Exception:
            data = {"summary": raw.strip(), "topics": [], "facts": []}
    except Exception as e:
        data = {"summary": f"(fallback) {user_text[:140]}", "topics": [], "facts": []}

    await history_append(state, "summary", json.dumps(data, ensure_ascii=False))
    print(c(f"📝 Resumen guardado → {data.get('summary','<sin resumen>')[:80]}", GRAY))

# Singleton de conveniencia (si quieres importarlo directo)
memory = MemoryManager()
