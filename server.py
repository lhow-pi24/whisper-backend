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
    Expects base64-encoded audio chunk (WebM or WAV)
    """
    try:
        # Decode the base64 audio data
        audio_bytes = base64.b64decode(data)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            temp_input.write(audio_bytes)
            temp_input_path = temp_input.name

        # 🔍 Transcribe using Whisper
        segments, _ = model.transcribe(
            temp_input_path,
            beam_size=3,
            best_of=3,
            condition_on_previous_text=False
        )

        # Combine text
        text = " ".join([s.text for s in segments]).strip()
        os.remove(temp_input_path)

        # Send result to all clients
        if text:
            emit("transcript", {"text": text}, broadcast=True)
            print(f"🗣 {text}")

    except Exception as e:
        print(f"❌ Error in handle_audio_chunk: {e}")
        traceback.print_exc()


# -------------------------------
# 🚀 Start Server
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)
