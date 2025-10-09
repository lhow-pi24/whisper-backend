from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("base")  # You can change to "small" for faster loading

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    audio = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio.save(temp_audio.name)
        result = model.transcribe(temp_audio.name)
        return jsonify({"text": result["text"]})

@app.route("/", methods=["GET"])
def home():
    return "✅ Whisper Backend is Running"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
