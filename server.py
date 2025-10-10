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
    
    # Check if file is empty
    if audio.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_input:
            audio.save(temp_input.name)
            temp_input_path = temp_input.name

        # Validate
        if not os.path.exists(temp_input_path) or os.path.getsize(temp_input_path) < 1000:
            print("❌ Received invalid or empty audio file")
            return jsonify({"error": "Uploaded file is empty or corrupted"}), 400


        # Convert to WAV using FFmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
            temp_output_path = temp_output.name

        try:
            # FFmpeg command to convert WebM to WAV
            ffmpeg_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-f", "webm", "-i", temp_input_path,
                "-ar", "16000", "-ac", "1",
                "-acodec", "pcm_s16le", "-f", "wav", temp_output_path
            ]

            process = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30  # Add timeout to prevent hanging
            )

            if process.returncode != 0:
                error_details = process.stderr.decode('utf-8', errors='ignore')
                print(f"🔴 FFmpeg error (return code: {process.returncode}):\n{error_details}")
                
                # Try alternative approach if first fails
                print("🔄 Trying alternative FFmpeg approach...")
                alt_cmd = [
                    "ffmpeg", "-y",
                    "-f", "webm",       # Force input format
                    "-i", temp_input_path,
                    "-ar", "16000",
                    "-ac", "1",
                    "-acodec", "pcm_s16le",
                    temp_output_path
                ]
                
                alt_process = subprocess.run(
                    alt_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
                
                if alt_process.returncode != 0:
                    alt_error = alt_process.stderr.decode('utf-8', errors='ignore')
                    print(f"🔴 Alternative FFmpeg also failed:\n{alt_error}")
                    return jsonify({
                        "error": "Audio conversion failed", 
                        "details": f"Primary: {error_details[:200]}... Alternative: {alt_error[:200]}..."
                    }), 500

            # Check if output file was created and has content
            if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
                return jsonify({"error": "Converted audio file is empty"}), 500

            # Transcribe with Whisper
            segments, info = model.transcribe(
                temp_output_path,
                beam_size=5,
                best_of=5,
                patience=1,
                condition_on_previous_text=False
            )
            
            text = " ".join([segment.text for segment in segments]).strip()

            print(f"✅ Transcription success: {text[:80]}{'...' if len(text) > 80 else ''}")

            return jsonify({
                "text": text,
                "language": info.language,
                "language_probability": info.language_probability
            })

        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")

    except subprocess.TimeoutExpired:
        print("🔴 FFmpeg timeout")
        return jsonify({"error": "Audio processing timeout"}), 500
    except Exception as e:
        print(f"❌ Exception during transcription: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)