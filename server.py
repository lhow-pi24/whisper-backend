from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os
import subprocess
import ffmpeg
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            audio.save(temp_audio.name)
            wav_path = temp_audio.name.replace(".webm", ".wav")

            result = subprocess.run(
                ["ffmpeg", "-y", "-i", temp_audio.name, "-ar", "16000", "-ac", "1", wav_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                print("🔴 FFmpeg error:", result.stderr.decode())
                return jsonify({"error": "FFmpeg failed", "details": result.stderr.decode()}), 500

            segments, info = model.transcribe(wav_path)
            text = " ".join([segment.text for segment in segments])

            os.remove(temp_audio.name)
            os.remove(wav_path)

            return jsonify({"text": text})

    except Exception as e:
        print("❌ Exception:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
