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
        # Save the uploaded blob to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_in:
            audio.save(temp_in.name)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            # ✅ FIXED: let ffmpeg auto-detect input
            process = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", temp_in.name,
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    temp_out.name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.returncode != 0:
                print("🔴 FFmpeg error:\n", process.stderr.decode())
                return jsonify({"error": "FFmpeg failed", "details": process.stderr.decode()}), 500

            # Run Whisper transcription
            segments, info = model.transcribe(temp_out.name)
            text = " ".join([segment.text for segment in segments])

            os.remove(temp_in.name)
            os.remove(temp_out.name)

            print("✅ Transcription success:", text[:60])
            return jsonify({"text": text})

    except Exception as e:
        print("❌ Exception during transcription:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    try:
        # 🪣 Save uploaded file first
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_in:
            audio.save(temp_in.name)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            # 🎧 Convert webm -> wav using FFmpeg
            process = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "webm",        # specify input format
                    "-i", temp_in.name,  # input file
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "wav",
                    temp_out.name
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.returncode != 0:
                print("🔴 FFmpeg error:\n", process.stderr.decode())
                return jsonify({"error": "FFmpeg failed", "details": process.stderr.decode()}), 500

            # 🧠 Run Whisper
            segments, info = model.transcribe(temp_out.name)
            text = " ".join([segment.text for segment in segments])

            os.remove(temp_in.name)
            os.remove(temp_out.name)

            print("✅ Transcription success:", text[:60])
            return jsonify({"text": text})

    except Exception as e:
        print("❌ Exception during transcription:", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio = request.files["file"]

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            # 🪄 Convert directly from stdin -> wav
            process = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "webm",
                    "-i", "pipe:0",     # read from stdin
                    "-ar", "16000",     # sample rate
                    "-ac", "1",         # mono
                    "-f", "wav",        # output format
                    temp_wav.name
                ],
                input=audio.read(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.returncode != 0:
                print("🔴 FFmpeg error:\n", process.stderr.decode())
                return jsonify({"error": "FFmpeg failed", "details": process.stderr.decode()}), 500

            # 🧠 Transcribe
            segments, info = model.transcribe(temp_wav.name)
            text = " ".join([segment.text for segment in segments])

            os.remove(temp_wav.name)
            print("✅ Transcription success:", text[:60])
            return jsonify({"text": text})

    except Exception as e:
        print("❌ Exception during transcription:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
