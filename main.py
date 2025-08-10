from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import io
import os
import asyncio
import json
import ffmpeg
from datetime import datetime, timedelta
from starlette.websockets import WebSocketState


# Ensure FFmpeg is available
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

load_dotenv()
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_DIR = "audio_logs"
os.makedirs(AUDIO_DIR, exist_ok=True)


def cleanup_old_files(directory: str, minutes: int = 10, extensions: tuple = (".wav",)):
    cutoff = datetime.now() - timedelta(minutes=minutes)
    for filename in os.listdir(directory):
        if not filename.lower().endswith(extensions):
            continue
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)
                print(f"[CLEANUP] Removed old file: {filename}")


def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    input_stream = io.BytesIO(webm_bytes)
    process = (
        ffmpeg
        .input("pipe:0")
        .output("pipe:1", format="wav", ac=1, ar=16000)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate(input=input_stream.read())
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {err.decode()}")
    return out


@app.websocket("/ws/voice")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps({"status": "connected"}))
    await websocket.send_text(json.dumps({"debug": "hello from backend"}))

    voice_param = websocket.query_params.get("voice", "default")
    voice_map = {
        "default": "coral",
        "female": "nova",
        "male": "echo",
        "robot": "onyx"
    }
    selected_voice = voice_map.get(voice_param, "coral")
    print(f"[INFO] Selected voice: {selected_voice}")

    try:
        cleanup_old_files(AUDIO_DIR, minutes=10, extensions=(".wav",))

        while True:
            audio_buffer = io.BytesIO()

            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30)
                except asyncio.TimeoutError:
                    print("[TIMEOUT] No audio received")
                    await websocket.send_text(json.dumps({"error": "Timeout waiting for audio"}))
                    break

                if data == b"__END__":
                    print("[INFO] End of recording received")
                    break
                elif data == b"__CLOSE__":
                    print("[INFO] Close signal received")
                    await websocket.send_text(json.dumps({"status": "closing"}))
                    await websocket.close(code=1000)
                    return

                print(f"[CHUNK] Received {len(data)} bytes")
                audio_buffer.write(data)

            audio_buffer.seek(0)
            raw_bytes = audio_buffer.read()
            print(f"[INFO] Total WebM audio received: {len(raw_bytes)} bytes")

            if len(raw_bytes) == 0:
                await websocket.send_text(json.dumps({"error": "No audio received"}))
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_prefix = f"session_{timestamp}"

            try:
                wav_bytes = convert_webm_to_wav(raw_bytes)

                if wav_bytes is None:
                    print("[ERROR] FFmpeg conversion failed: No output")
                    await websocket.send_text(json.dumps({"error": "FFmpeg conversion failed"}))
                    continue

                print(f"[INFO] Converted to WAV: {len(wav_bytes)} bytes")
            except Exception as e:
                await websocket.send_text(json.dumps({"error": "FFmpeg conversion failed"}))
                print("[ERROR] FFmpeg conversion failed:", e)
                continue

            wav_path = os.path.join(AUDIO_DIR, f"{session_prefix}.wav")
            # with open(wav_path, "wb") as f:
            #     f.write(wav_bytes)
            # print(f"[INFO] Saved WAV to {wav_path}")

            wav_buffer = io.BytesIO(wav_bytes)
            wav_buffer.name = "audio.wav"

            await websocket.send_text(json.dumps({"progress": "starting transcription"}))
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=wav_buffer
                )
            except Exception as e:
                print("[ERROR] Transcription failed:", e)
                await websocket.send_text(json.dumps({"error": "Transcription failed"}))
                continue

            user_text = transcription.text
            print(f"[INFO] Transcription: {user_text}")
            await websocket.send_text(json.dumps({"transcript": user_text}))
            await websocket.send_text(json.dumps({"progress": "starting LLM"}))

            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_text}
                ],
                stream=True
            )

            full_response = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta

            print(f"[INFO] LLM Response: {full_response}")
            await websocket.send_text(json.dumps({"progress": "starting TTS"}))

            tts_wave_data = b""
            try:
                with client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    voice=selected_voice,
                    input=full_response,
                    instructions="Speak in a cheerful and positive tone.",
                ) as response:
                    for audio_chunk in response.iter_bytes():
                        tts_wave_data += audio_chunk
            except Exception as e:
                print("[ERROR] TTS generation failed:", e)
                await websocket.send_text(json.dumps({"error": "TTS generation failed"}))
                continue

            ffmpeg_process = (
                ffmpeg
                .input("pipe:0")
                .output("pipe:1", format="wav", ac=1, ar=24000)
                .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
            )
            wav_data, err = ffmpeg_process.communicate(input=tts_wave_data)
            if ffmpeg_process.returncode != 0:
                await websocket.send_text(json.dumps({"error": "FFmpeg conversion failed"}))
                print("[ERROR] FFmpeg TTS conversion error:", err.decode())
                continue

            chunk_size = 4096
            for i in range(0, len(wav_data), chunk_size):
                await websocket.send_bytes(wav_data[i:i+chunk_size])
                print(f"[DEBUG] Sent audio chunk {i//chunk_size + 1}")

            tts_wav_path = os.path.join(AUDIO_DIR, f"{session_prefix}_tts.wav")
            # with open(tts_wav_path, "wb") as f:
            #     f.write(wav_data)
            # print(f"[INFO] Saved TTS WAV to {tts_wav_path}")

            await asyncio.sleep(0.2)
            await websocket.send_text(json.dumps({"audio_done": True}))

    except WebSocketDisconnect as e:
        print(f"[INFO] WebSocket disconnected: code={e.code}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"error": "Server error occurred"}))
        except Exception as send_err:
            print(f"[WARN] Failed to send error message: {send_err}")
