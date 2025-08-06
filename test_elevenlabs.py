import asyncio, os, pyaudio
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs

load_dotenv()
client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"        # Adam
VOICE_OPTS = {                           # opcional
    "stability": 0.45,
    "similarity_boost": 0.85,
    "style": 0.15,
    "use_speaker_boost": True,
}

async def main():
    stream_gen = client.text_to_speech.stream(
        text="Hi Sir, this is the correct way to generate audio with ElevenLabs.",
        voice_id=VOICE_ID,
        model_id="eleven_turbo_v2",
        output_format="pcm_16000",
        voice_settings=VOICE_OPTS,
    )

    audio = b"".join([chunk async for chunk in stream_gen])

    pa = pyaudio.PyAudio()
    out = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    out.write(audio)
    out.stop_stream(); out.close(); pa.terminate()

if __name__ == "__main__":
    asyncio.run(main())
