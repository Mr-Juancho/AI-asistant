# jarvis/config.py
from dataclasses import dataclass
import os
from dotenv import load_dotenv
import pyaudio
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs

load_dotenv()

@dataclass
class Config:
    OPENAI_API_KEY: str
    DEEPGRAM_API_KEY: str
    ELEVENLABS_API_KEY: str

    # audio
    FORMAT: int
    CHANNELS: int
    RATE: int
    CHUNK: int
    FRAME_BYTES: int
    TAIL_SILENCE_FRAMES: int

    VOICE_ID: str
    ELEVEN_MODEL_ID: str

    pa: pyaudio.PyAudio
    out: any  # PyAudio stream

    openai_client: AsyncOpenAI
    eleven_client: AsyncElevenLabs

def load_config() -> Config:
    OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 480
    FRAME_BYTES = CHUNK * 2
    TAIL_SILENCE_FRAMES = CHUNK * 4

    pa = pyaudio.PyAudio()
    out = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                  output=True, frames_per_buffer=CHUNK, start=False)

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    eleven_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)

    return Config(
        OPENAI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY,
        FORMAT, CHANNELS, RATE, CHUNK, FRAME_BYTES, TAIL_SILENCE_FRAMES,
        VOICE_ID="IKne3meq5aSn9XLyUdCD",
        ELEVEN_MODEL_ID="eleven_multilingual_v2",
        pa=pa, out=out,
        openai_client=openai_client,
        eleven_client=eleven_client,
    )

def cleanup(cfg):
    try:
        if cfg.out.is_active():
            cfg.out.stop_stream()
        cfg.out.close()
    except Exception:
        pass
    try:
        cfg.pa.terminate()
    except Exception:
        pass

