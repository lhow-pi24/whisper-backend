from flask import Flask, request, jsonify
import tempfile, os
from faster_whisper import WhisperModel

app = Flask(__name__)

# Use the lightest model possible
model = WhisperModel("tiny", device="cpu", compute_type="int8")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    audio = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio.save(temp_audio.name)
        segments, info = model.transcribe(temp_audio.name)
        text = " ".join([segment.text for segment in segments])
        return jsonify({"text": text})

@app.route("/")
def home():
    return "✅ Faster Whisper Backend is Running"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
