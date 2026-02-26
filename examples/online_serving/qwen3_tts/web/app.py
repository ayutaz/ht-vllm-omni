"""HTMX + FastAPI Voice Cloning Web UI for Qwen3-TTS.

Provides three tabs:
  1. Voice Embedding extraction (ECAPA-TDNN speaker encoder)
  2. Voice Cloning TTS (text-to-speech with ref audio or saved embedding)
  3. Voice Interpolation (SLERP blend of two voices)

Usage:
    uv run python examples/online_serving/qwen3_tts/web/app.py

    # Custom model / device
    uv run python examples/online_serving/qwen3_tts/web/app.py \
        --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import traceback
import uuid
from typing import Any

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.online_serving.qwen3_tts.speaker_embedding_interpolation import slerp

# ---------------------------------------------------------------------------
# Shim: inject stub modules (reused from gradio_voice_cloning.py)
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.join(
    _REPO_ROOT, "vllm_omni", "model_executor", "models", "qwen3_tts"
)


def _install_stubs():
    import importlib
    import types

    def _download_weights_stub(
        model_name_or_path: str,
        cache_dir: str | None = None,
        allow_patterns: list[str] | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> str:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_name_or_path,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            revision=revision,
        )

    for mod_path in [
        "vllm_omni",
        "vllm_omni.model_executor",
        "vllm_omni.model_executor.model_loader",
        "vllm_omni.model_executor.model_loader.weight_utils",
        "vllm_omni.model_executor.models",
    ]:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)

    sys.modules[
        "vllm_omni.model_executor.model_loader.weight_utils"
    ].download_weights_from_hf_specific = _download_weights_stub

    pkg_name = "vllm_omni.model_executor.models.qwen3_tts"
    if _MODELS_DIR not in sys.path:
        sys.path.insert(0, _MODELS_DIR)

    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(_MODELS_DIR, "__init__.py"),
            submodule_search_locations=[_MODELS_DIR],
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg


_install_stubs()

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_tts.processing_qwen3_tts import (
    Qwen3TTSProcessor,
)

# ---------------------------------------------------------------------------
# Speaker Embedding Encoder (lightweight, trust_remote_code model)
# ---------------------------------------------------------------------------

_emb_encoder: torch.nn.Module | None = None
_emb_feature_extractor = None
_emb_model_id: str = ""


def get_embedding_encoder():
    global _emb_encoder, _emb_feature_extractor
    if _emb_encoder is None:
        from transformers import AutoFeatureExtractor, AutoModel

        print(f"[init] Loading embedding encoder from {_emb_model_id} ...")
        _emb_encoder = AutoModel.from_pretrained(
            _emb_model_id, trust_remote_code=True
        ).eval()
        _emb_feature_extractor = AutoFeatureExtractor.from_pretrained(
            _emb_model_id, trust_remote_code=True
        )
        print("[init] Embedding encoder ready.")
    return _emb_encoder, _emb_feature_extractor


@torch.inference_mode()
def _extract_embedding(audio_path: str) -> np.ndarray:
    import librosa

    encoder, fe = get_embedding_encoder()
    audio, sr = librosa.load(audio_path, sr=fe.sampling_rate, mono=True)
    inputs = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt")
    device = next(encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = encoder(**inputs)
    return output.last_hidden_state[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# TTS Wrapper (reused from gradio_voice_cloning.py)
# ---------------------------------------------------------------------------


class _TTSWrapper:
    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        processor: Qwen3TTSProcessor,
        generate_defaults: dict[str, Any] | None = None,
    ):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}
        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "_TTSWrapper":
        from transformers import AutoConfig, AutoModel, AutoProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(model_path, **kwargs)
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        gen_defaults = getattr(model, "generate_config", {}) or {}
        return cls(model=model, processor=processor, generate_defaults=gen_defaults)

    def _tokenize(self, text: str) -> torch.Tensor:
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        ids = inp["input_ids"].to(self.device)
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    @staticmethod
    def _wrap_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _gen_kwargs(self, **overrides) -> dict[str, Any]:
        defaults = dict(
            non_streaming_mode=False,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )
        merged = {}
        for k, v in defaults.items():
            merged[k] = overrides.get(k) or self.generate_defaults.get(k, v)
        return merged

    @torch.no_grad()
    def generate_with_embedding(
        self,
        text: str,
        language: str | None,
        speaker_embedding: list[float],
    ) -> tuple[np.ndarray, int]:
        spk = torch.tensor(speaker_embedding, dtype=torch.float32).to(self.device)
        prompt_dict = dict(
            ref_code=[None],
            ref_spk_embedding=[spk],
            x_vector_only_mode=[True],
            icl_mode=[False],
        )
        input_ids = [self._tokenize(self._wrap_text(text))]
        lang = language if language and language != "Auto" else "Auto"

        codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=None,
            voice_clone_prompt=prompt_dict,
            languages=[lang],
            **self._gen_kwargs(),
        )

        wavs, sr = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_list]
        )
        return wavs[0], int(sr)

    @torch.no_grad()
    def generate_with_ref_audio(
        self,
        text: str,
        language: str | None,
        ref_audio_path: str,
    ) -> tuple[np.ndarray, int]:
        import librosa as _lr

        wav, sr = _lr.load(ref_audio_path, sr=None, mono=True)
        wav = wav.astype(np.float32)

        spk_sr = int(self.model.speaker_encoder_sample_rate)
        if sr != spk_sr:
            wav_resample = _lr.resample(wav, orig_sr=sr, target_sr=spk_sr)
        else:
            wav_resample = wav

        spk_emb = self.model.extract_speaker_embedding(
            audio=wav_resample, sr=spk_sr
        )

        prompt_dict = dict(
            ref_code=[None],
            ref_spk_embedding=[spk_emb],
            x_vector_only_mode=[True],
            icl_mode=[False],
        )

        input_ids = [self._tokenize(self._wrap_text(text))]
        lang = language if language and language != "Auto" else "Auto"

        codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=None,
            voice_clone_prompt=prompt_dict,
            languages=[lang],
            **self._gen_kwargs(),
        )

        wavs, sr_out = self.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_list]
        )
        return wavs[0], int(sr_out)


# ---------------------------------------------------------------------------
# Global TTS model handle (lazy-loaded)
# ---------------------------------------------------------------------------
_tts_model: _TTSWrapper | None = None
_tts_model_path: str = ""
_device: str = "cuda:0"


def get_tts_model() -> _TTSWrapper:
    global _tts_model
    if _tts_model is None:
        print(f"[init] Loading TTS model from {_tts_model_path} ...")
        _tts_model = _TTSWrapper.from_pretrained(
            _tts_model_path,
            torch_dtype=torch.bfloat16,
            device_map=_device,
        )
        print("[init] TTS model ready.")
    return _tts_model


# ---------------------------------------------------------------------------
# Supported languages (same as Gradio version)
# ---------------------------------------------------------------------------
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish",
]

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Qwen3 TTS Voice Cloning")
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", uuid.uuid4().hex))
app.mount("/static", StaticFiles(directory=os.path.join(_WEB_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(_WEB_DIR, "templates"))

# Temp directory for generated audio files
_TEMP_DIR = tempfile.mkdtemp(prefix="voice_cloning_")


def _save_upload(upload: UploadFile) -> str:
    """Save an UploadFile to a temp path and return the path."""
    suffix = os.path.splitext(upload.filename or "audio.wav")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=_TEMP_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _wav_to_data_uri(wav: np.ndarray, sr: int) -> str:
    """Encode WAV audio as a base64 data URI for inline <audio> playback."""
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), sr, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _save_wav(wav: np.ndarray, sr: int) -> str:
    """Save WAV to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".wav", dir=_TEMP_DIR)
    with os.fdopen(fd, "wb") as f:
        sf.write(f, wav.astype(np.float32), sr, format="WAV")
    return path


# ---------------------------------------------------------------------------
# Routes: pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tab/extract", response_class=HTMLResponse)
async def tab_extract(request: Request):
    return templates.TemplateResponse("tabs/extract.html", {"request": request})


@app.get("/tab/cloning", response_class=HTMLResponse)
async def tab_cloning(request: Request):
    return templates.TemplateResponse(
        "tabs/cloning.html",
        {"request": request, "languages": LANGUAGES},
    )


@app.get("/tab/interpolation", response_class=HTMLResponse)
async def tab_interpolation(request: Request):
    return templates.TemplateResponse(
        "tabs/interpolation.html",
        {"request": request, "languages": LANGUAGES},
    )


# ---------------------------------------------------------------------------
# Routes: API endpoints
# ---------------------------------------------------------------------------


@app.post("/api/extract-embedding", response_class=HTMLResponse)
async def api_extract_embedding(
    request: Request,
    audio: UploadFile = File(...),
):
    try:
        audio_path = _save_upload(audio)
        emb = _extract_embedding(audio_path)

        norm = float(np.linalg.norm(emb))
        info = {
            "dimension": emb.shape[0],
            "l2_norm": f"{norm:.4f}",
            "min": f"{emb.min():.4f}",
            "max": f"{emb.max():.4f}",
            "mean": f"{emb.mean():.4f}",
        }

        # Store in session
        request.session["embedding_state"] = emb.tolist()

        # Save JSON for download
        json_path = os.path.join(_TEMP_DIR, f"embedding_{uuid.uuid4().hex[:8]}.json")
        with open(json_path, "w") as f:
            json.dump(emb.tolist(), f)
        request.session["embedding_json_path"] = json_path

        return templates.TemplateResponse(
            "partials/embedding_info.html",
            {"request": request, "info": info, "success": True},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/status.html",
            {"request": request, "message": f"Error: {e}", "error": True},
        )


@app.get("/api/download-embedding")
async def api_download_embedding(request: Request):
    json_path = request.session.get("embedding_json_path")
    if not json_path or not os.path.exists(json_path):
        return JSONResponse({"error": "No embedding available"}, status_code=404)
    return FileResponse(
        json_path,
        media_type="application/json",
        filename="speaker_embedding.json",
    )


@app.post("/api/generate-speech", response_class=HTMLResponse)
async def api_generate_speech(
    request: Request,
    text: str = Form(...),
    language: str = Form("Auto"),
    ref_audio: UploadFile | None = File(None),
    embedding_json: UploadFile | None = File(None),
):
    try:
        if not text.strip():
            return templates.TemplateResponse(
                "partials/status.html",
                {"request": request, "message": "Please enter text to synthesize.", "error": True},
            )

        model = get_tts_model()
        status_msg = ""

        # Priority: uploaded JSON > session state > ref audio
        if embedding_json is not None and embedding_json.filename:
            content = await embedding_json.read()
            emb = json.loads(content.decode("utf-8"))
            wav, sr = model.generate_with_embedding(text, language, emb)
            status_msg = f"Generated with uploaded embedding ({len(emb)} dims)"

        elif request.session.get("embedding_state"):
            emb = request.session["embedding_state"]
            wav, sr = model.generate_with_embedding(text, language, emb)
            status_msg = f"Generated with Tab 1 embedding ({len(emb)} dims)"

        elif ref_audio is not None and ref_audio.filename:
            audio_path = _save_upload(ref_audio)
            wav, sr = model.generate_with_ref_audio(text, language, audio_path)
            status_msg = "Generated with reference audio (x-vector mode)"

        else:
            return templates.TemplateResponse(
                "partials/status.html",
                {"request": request, "message": "Please provide a reference audio or embedding.", "error": True},
            )

        audio_uri = _wav_to_data_uri(wav, sr)

        return templates.TemplateResponse(
            "partials/audio_player.html",
            {
                "request": request,
                "audio_uri": audio_uri,
                "status": status_msg,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/status.html",
            {"request": request, "message": f"Error: {e}\n{traceback.format_exc()}", "error": True},
        )


@app.post("/api/interpolate", response_class=HTMLResponse)
async def api_interpolate(
    request: Request,
    audio_a: UploadFile = File(...),
    audio_b: UploadFile = File(...),
    ratio: float = Form(0.5),
    text: str = Form(...),
    language: str = Form("Auto"),
):
    try:
        if not text.strip():
            return templates.TemplateResponse(
                "partials/status.html",
                {"request": request, "message": "Please enter text to synthesize.", "error": True},
            )

        path_a = _save_upload(audio_a)
        path_b = _save_upload(audio_b)

        emb_a = _extract_embedding(path_a)
        emb_b = _extract_embedding(path_b)
        blended = slerp(emb_a, emb_b, ratio)

        model = get_tts_model()
        wav, sr = model.generate_with_embedding(text, language, blended.tolist())

        audio_uri = _wav_to_data_uri(wav, sr)

        cos_sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))

        info = {
            "norm_a": f"{np.linalg.norm(emb_a):.4f}",
            "norm_b": f"{np.linalg.norm(emb_b):.4f}",
            "ratio": f"{ratio:.2f}",
            "norm_blended": f"{np.linalg.norm(blended):.4f}",
            "cosine_similarity": f"{cos_sim:.4f}",
        }

        return templates.TemplateResponse(
            "partials/audio_player.html",
            {
                "request": request,
                "audio_uri": audio_uri,
                "status": f"SLERP ratio: {ratio:.2f} (0=A, 1=B)",
                "interpolation_info": info,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/status.html",
            {"request": request, "message": f"Error: {e}\n{traceback.format_exc()}", "error": True},
        )


@app.post("/api/upload-recording", response_class=HTMLResponse)
async def api_upload_recording(
    request: Request,
    audio: UploadFile = File(...),
):
    try:
        audio_path = _save_upload(audio)
        emb = _extract_embedding(audio_path)

        norm = float(np.linalg.norm(emb))
        info = {
            "dimension": emb.shape[0],
            "l2_norm": f"{norm:.4f}",
            "min": f"{emb.min():.4f}",
            "max": f"{emb.max():.4f}",
            "mean": f"{emb.mean():.4f}",
        }

        request.session["embedding_state"] = emb.tolist()

        json_path = os.path.join(_TEMP_DIR, f"embedding_{uuid.uuid4().hex[:8]}.json")
        with open(json_path, "w") as f:
            json.dump(emb.tolist(), f)
        request.session["embedding_json_path"] = json_path

        return templates.TemplateResponse(
            "partials/embedding_info.html",
            {"request": request, "info": info, "success": True, "from_recording": True},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "partials/status.html",
            {"request": request, "message": f"Recording error: {e}", "error": True},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="HTMX Voice Cloning Web UI for Qwen3-TTS"
    )
    parser.add_argument(
        "--encoder-model",
        default="marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
        help="HF model ID for the ECAPA-TDNN speaker encoder",
    )
    parser.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HF model ID or local path for the TTS model",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for TTS model (default: cuda:0)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Server port (default: 8001)"
    )
    return parser.parse_args()


def main():
    global _emb_model_id, _tts_model_path, _device

    args = parse_args()
    _emb_model_id = args.encoder_model
    _tts_model_path = args.tts_model
    _device = args.device

    # Eagerly load the speaker encoder (lightweight, ~12M params)
    get_embedding_encoder()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
