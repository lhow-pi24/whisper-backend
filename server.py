from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, subprocess
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

# 🧩 Load lightweight model
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    try:
        # Save uploaded .webm audio temporarily
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            audio.save(temp_audio.name)
            wav_path = temp_audio.name.replace(".webm", ".wav")

            # 🔁 Convert WebM → WAV using ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_audio.name, "-ar", "16000", "-ac", "1", wav_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 🧠 Transcribe with faster-whisper
            segments, info = model.transcribe(wav_path)
            text = " ".join([segment.text for segment in segments])

            # 🧹 Clean up
            os.remove(temp_audio.name)
            os.remove(wav_path)

            return jsonify({"text": text})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"FFmpeg failed: {e.stderr.decode()}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
