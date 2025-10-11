from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import tempfile, os, subprocess, traceback, base64
from faster_whisper import WhisperModel

# -------------------------------
# ⚙️ Flask + SocketIO setup
# -------------------------------
app = Flask(__name__)

# ✅ Allow Ionic (localhost:8101) + Render frontend
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ SocketIO must also allow CORS
socketio = SocketIO(app, cors_allowed_origins="*")

# -------------------------------
# 🧠 Load Whisper model once
# -------------------------------
print("⏳ Loading Whisper model (tiny)...")
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("✅ Whisper model loaded successfully.")


# -------------------------------
# 🌐 Routes
# -------------------------------
@app.route("/")
def home():
    return jsonify({"status": "running", "message": "✅ Faster Whisper WebSocket Server"})


# -------------------------------
# 🎧 SocketIO Events
# -------------------------------
@socketio.on("connect")
def handle_connect():
    print("🟢 Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("🔴 Client disconnected")


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Receives base64 audio chunk (WebM or WAV) from the client,
    converts it with ffmpeg, then transcribes.
    """
    try:
        audio_bytes = base64.b64decode(data)

        # Save base64 chunk to .webm file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name

        # Convert WebM → WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            temp_output_path = temp_output.name

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", temp_input_path,
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le", temp_output_path
        ]

        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print("🔴 FFmpeg conversion failed:")
            print(result.stderr.decode())
            return

        # ✅ Now transcribe clean WAV
        segments, _ = model.transcribe(temp_output_path)
        text = " ".join([s.text for s in segments]).strip()

        if text:
            emit("transcript", {"text": text}, broadcast=True)
            print(f"🗣 {text}")

    except Exception as e:
        print(f"❌ Error in handle_audio_chunk: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except Exception:
            pass



# -------------------------------
# 🚀 Start Server
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)
