# Voice Cloning Web UI (HTMX)

Lightweight web interface for Qwen3-TTS voice cloning, built with
HTMX + FastAPI + DaisyUI.

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTMX 2.x (14KB, no build step) |
| Styling | DaisyUI + Tailwind CSS (CDN) |
| Templates | Jinja2 |
| Backend | FastAPI + Uvicorn |
| Audio Recording | Vanilla JS (getUserMedia) |
| Session | Server-side (Starlette SessionMiddleware) |

## Prerequisites

- Python 3.10+
- CUDA GPU (for TTS model inference)
- Models will be downloaded automatically from Hugging Face:
  - `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` (speaker encoder, ~12M params)
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (TTS model, loaded on first generation)

## Quick Start

```bash
# Install web dependencies
uv sync --extra web

# Run the server
uv run python examples/online_serving/qwen3_tts/web/app.py

# Custom model / device / port
uv run python examples/online_serving/qwen3_tts/web/app.py \
    --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --device cuda:0 \
    --port 8001
```

Open http://localhost:8001 in your browser.

## Tabs

### Tab 1: Extract Embedding
Upload a reference audio file (or record from microphone) to extract a
2048-dimensional speaker embedding using the ECAPA-TDNN encoder.
The embedding is stored in the server session and automatically available
in Tab 2. You can also download it as JSON.

### Tab 2: Voice Cloning TTS
Generate speech with a cloned voice. Voice source priority:
1. Uploaded Embedding JSON (highest)
2. Tab 1 session embedding (automatic)
3. Reference audio (direct x-vector cloning)

### Tab 3: Voice Interpolation
Blend two voices using SLERP (Spherical Linear Interpolation).
Upload two reference audio files, adjust the blend ratio (0.0 = Voice A,
1.0 = Voice B), and generate speech with the blended voice.

## API Endpoints

| Method | Path | Returns | Purpose |
|---|---|---|---|
| GET | `/` | HTML | Main page |
| GET | `/tab/extract` | HTML fragment | Tab 1 content |
| GET | `/tab/cloning` | HTML fragment | Tab 2 content |
| GET | `/tab/interpolation` | HTML fragment | Tab 3 content |
| POST | `/api/extract-embedding` | HTML fragment | Extract embedding |
| POST | `/api/generate-speech` | HTML fragment | Generate TTS |
| POST | `/api/interpolate` | HTML fragment | SLERP interpolation |
| GET | `/api/download-embedding` | JSON file | Download embedding |
| POST | `/api/upload-recording` | HTML fragment | Microphone recording |

## Comparison with Gradio Version

| Feature | Gradio (port 7860) | HTMX (port 8001) |
|---|---|---|
| Tabs 1-3 (Extract, Clone, Interpolate) | Yes | Yes |
| Tabs 4-7 (Library, Arithmetic, etc.) | Yes | No (future) |
| Build step | None | None |
| JS bundle size | ~500KB (Gradio) | ~14KB (HTMX) |
| Dark theme | Gradio Soft theme | DaisyUI night theme |

Both versions can run simultaneously on different ports.
