# jarvis/state.py
from __future__ import annotations
from dataclasses import dataclass, field
import asyncio, time, json, os
from typing import List, Dict, Any, Optional

# Tu RAG (Chroma) vive en memoria.py en la raíz del repo
from memory import MemoryManager

# ===== Configuración de historial persistente =====
MAX_HISTORY_TURNS = 5
HISTORY_FILE = "myKnowledgeBase/conversation_history.json"


class SharedState:
    """Estado compartido de audio (para pausar el micrófono mientras suena TTS)."""
    def __init__(self):
        self.is_speaking: bool = False


@dataclass
class AppState:
    # --- Memorias ---
    kb: MemoryManager                                 # tu RAG privado (Chroma)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)  # últimos turnos para el prompt
    full_history: List[Dict[str, Any]] = field(default_factory=list)          # TODO el JSON persistente

    # --- Señales / colas ---
    tts_started_event: asyncio.Event = field(default_factory=asyncio.Event)
    llm_text_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=8))

    # --- Flags de UI/flujo ---
    grace_mode: bool = False
    suppress_user_partials: bool = False
    assistant_printing: bool = False

    # --- Tareas en curso (para cancelar previews, etc.) ---
    current_preview_task: Optional[asyncio.Task] = None
    current_llm_task: Optional[asyncio.Task] = None

    # --- Locks ---
    history_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # --- Audio compartido ---
    shared_state: SharedState = field(default_factory=SharedState)


async def make_state() -> AppState:
    """Construye el AppState con tu MemoryManager (RAG)."""
    kb = MemoryManager()  # inicializa embeddings / Chroma retriever
    return AppState(kb=kb)


# ===== Persistencia simple a JSON =====
def _read_history_file_sync() -> list:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _write_history_file_sync(history: list):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


async def history_init(state: AppState):
    """Carga todo el historial desde disco y siembra la ventana reciente para el prompt."""
    state.full_history = _read_history_file_sync()
    recent = [t for t in state.full_history if t.get("role") in ("user", "assistant")]
    state.conversation_history = recent[-(MAX_HISTORY_TURNS * 2):]


async def history_append(state: AppState, role: str, content: str):
    """Añade un turno al historial completo y persiste a JSON."""
    turn = {"role": role, "content": content, "ts": time.time()}
    state.full_history.append(turn)
    async with state.history_lock:
        await asyncio.to_thread(_write_history_file_sync, state.full_history)
