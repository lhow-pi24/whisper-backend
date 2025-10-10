from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, subprocess, traceback
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

# Load Whisper model once at startup
try:
    print("⏳ Loading Whisper model (tiny)...")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    print("✅ Whisper model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    try:
        # Save webm file temporarily
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio:
            audio.save(temp_audio.name)
            wav_path = temp_audio.name.replace(".webm", ".wav")

            # Convert to wav using ffmpeg
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", temp_audio.name,
                    "-ar", "16000", "-ac", "1",
                    wav_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.returncode != 0:
                print("🔴 FFmpeg error:\n", result.stderr.decode())
                return jsonify({"error": "FFmpeg failed", "details": result.stderr.decode()}), 500

            # Transcribe
            segments, info = model.transcribe(wav_path)
            text = " ".join([segment.text for segment in segments])

            # Clean up
            os.remove(temp_audio.name)
            os.remove(wav_path)

            print("✅ Transcription success:", text[:60])
            return jsonify({"text": text})

    except Exception as e:
        print("❌ Exception during transcription:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
