# AI Voice Generator

A simple text-to-speech web app that clones your voice from a short sample
and speaks any text you give it, exported as an MP3.

Runs entirely on your own machine using the open-source
[Coqui XTTS-v2](https://github.com/coqui-ai/TTS) model — no API key, no
account, no per-use cost.

## Requirements

- Python 3.9–3.11
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
  (needed to export MP3)
- ~2 GB free disk space (the voice model downloads automatically on first run)
- A CPU works, but generation is faster with a GPU (CUDA)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

1. Upload a short sample of your voice (a clean 5–15 second WAV or MP3 clip
   works best, no background noise).
2. Type the text you want spoken.
3. Choose a language.
4. Click **Generate MP3** — the first run downloads the voice model
   (a few GB), so it will take longer than later runs.
5. The generated MP3 plays in the browser and downloads automatically.

Uploaded samples and intermediate audio files are deleted from the server
right after each generation; only the MP3 is returned to you.
