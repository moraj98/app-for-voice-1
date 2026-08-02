"""Simple AI voice generator: clone a voice from a sample and speak any text, exported as MP3."""
import os
import uuid

from flask import Flask, render_template, request, send_file, jsonify
from pydub import AudioSegment

from prompt_generator import INTENT_ROLES, generate_prompt, refine_prompt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, "voice_samples")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_SAMPLE_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
DEFAULT_SAMPLE_PATH = os.path.join(BASE_DIR, "voice_samples", "mohit_sample.mp3")

os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_tts_model = None


def get_tts_model():
    """Lazily load the voice-cloning model (large download on first use)."""
    global _tts_model
    if _tts_model is None:
        from TTS.api import TTS
        _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prompt")
def prompt_page():
    return render_template("prompt.html", intents=sorted(INTENT_ROLES))


@app.route("/api/prompt", methods=["POST"])
def api_prompt():
    payload = request.get_json(silent=True) or {}
    words = (payload.get("words") or "").strip()
    intent = (payload.get("intent") or "auto").strip()
    existing = (payload.get("prompt") or "").strip()

    try:
        if payload.get("mode") == "refine" and existing:
            result = refine_prompt(existing, words, intent=intent)
        else:
            result = generate_prompt(words, intent=intent)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@app.route("/generate", methods=["POST"])
def generate():
    text = request.form.get("text", "").strip()
    language = request.form.get("language", "en").strip() or "en"
    voice_sample = request.files.get("voice_sample")

    if not text:
        return jsonify({"error": "Please enter some text to speak."}), 400

    job_id = uuid.uuid4().hex
    uploaded_sample_path = None

    if voice_sample and voice_sample.filename:
        ext = os.path.splitext(voice_sample.filename)[1].lower()
        if ext not in ALLOWED_SAMPLE_EXTENSIONS:
            return jsonify({"error": f"Unsupported voice sample type: {ext}"}), 400
        uploaded_sample_path = os.path.join(VOICE_SAMPLES_DIR, f"{job_id}{ext}")
        voice_sample.save(uploaded_sample_path)
        sample_path = uploaded_sample_path
    elif os.path.exists(DEFAULT_SAMPLE_PATH):
        sample_path = DEFAULT_SAMPLE_PATH
    else:
        return jsonify({"error": "Please upload a voice sample."}), 400

    wav_path = os.path.join(OUTPUTS_DIR, f"{job_id}.wav")
    mp3_path = os.path.join(OUTPUTS_DIR, f"{job_id}.mp3")

    try:
        model = get_tts_model()
        model.tts_to_file(
            text=text,
            speaker_wav=sample_path,
            language=language,
            file_path=wav_path,
        )
        AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3")
    except Exception as exc:
        return jsonify({"error": f"Voice generation failed: {exc}"}), 500
    finally:
        for path in (uploaded_sample_path, wav_path):
            if path and os.path.exists(path):
                os.remove(path)

    return send_file(
        mp3_path,
        as_attachment=True,
        download_name="ai_voice.mp3",
        mimetype="audio/mpeg",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
