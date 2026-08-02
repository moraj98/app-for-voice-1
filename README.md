# AI Voice Generator

A simple text-to-speech web app that clones your voice from a short sample
and speaks any text you give it, exported as an MP3.

Runs entirely on your own machine using the open-source
[Coqui XTTS-v2](https://github.com/coqui-ai/TTS) model — no API key, no
account, no per-use cost.

## Easiest free option: Google Colab (no installation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/moraj98/app-for-voice-1/blob/claude/tts-ai-voice-generator-410s1c/voice_clone_colab.ipynb)

Click the badge above → switch the runtime to the free **T4 GPU**
(Runtime → Change runtime type) → run the cells top to bottom. It clones the
voice from the sample in this repo and downloads an MP3 — completely free.

## Run on your own computer

### Requirements

- Python 3.9–3.11
- [ffmpeg](https://ffmpeg.org/download.html) installed and on your `PATH`
  (needed to export MP3)
- ~2 GB free disk space (the voice model downloads automatically on first run)
- A CPU works, but generation is faster with a GPU (CUDA)

### Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

### Usage

1. Upload a short sample of your voice (a clean 5–15 second WAV or MP3 clip
   works best, no background noise).
2. Type the text you want spoken.
3. Choose a language.
4. Click **Generate MP3** — the first run downloads the voice model
   (a few GB), so it will take longer than later runs.
5. The generated MP3 plays in the browser and downloads automatically.

Uploaded samples and intermediate audio files are deleted from the server
right after each generation; only the MP3 is returned to you.

## Prompt generator

A second page at http://localhost:5000/prompt turns a handful of words into a
structured prompt, then refines it as you add more.

Type loose keywords — `cat, cyberpunk, neon, close-up, 16:9, no people` — and it
sorts them into subject, style, lighting, camera, tone, format and so on,
detects what kind of prompt you are after (image, video, audio, code, writing,
analysis), and assembles a matching template:

```
A cyberpunk image of cat.

Composition: close-up
Lighting: neon
Output: 16:9, high detail and sharp focus on the subject
Avoid: people
```

It also lists what the prompt is still missing (an art style, a target length,
who the reader is), so you can type those words in and press **Refine** to fold
them into the same prompt instead of starting over. Refining a prompt with no
new words leaves it unchanged, so you can iterate as many times as you like.

Two conventions worth knowing:

- Words prefixed with `no`, `without` or `-` go into the avoid list.
- A comma-separated chunk of five or more words is kept verbatim as the subject,
  so `why is my deploy slow` survives intact; shorter chunks get their filler
  words stripped.

This runs offline like the rest of the app — no model, no API key. It is a
parser and a set of templates, not a language model.

### From the command line

```bash
python prompt_generator.py "drone shot over a desert highway, golden hour, cinematic, 9:16"
python prompt_generator.py "500 word linkedin post on hiring, witty" --json
python prompt_generator.py "cat, watercolor" --intent video
```

### Tests

```bash
python -m unittest
```
