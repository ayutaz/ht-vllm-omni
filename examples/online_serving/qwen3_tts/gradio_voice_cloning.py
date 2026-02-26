"""Gradio GUI for Qwen3-TTS voice cloning with speaker embeddings.

Provides three tabs:
  1. Voice Embedding extraction (ECAPA-TDNN speaker encoder)
  2. Voice Cloning TTS (text-to-speech with ref audio or saved embedding)
  3. Voice Interpolation (SLERP blend of two voices)

Requirements:
    pip install gradio torch librosa soundfile numpy

Usage:
    python examples/online_serving/qwen3_tts/gradio_voice_cloning.py

    # Use a different TTS model or device
    python examples/online_serving/qwen3_tts/gradio_voice_cloning.py \
        --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from typing import Any

import gradio as gr
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the repo root is importable so we can use the example module
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.online_serving.qwen3_tts.speaker_embedding_interpolation import (
    compute_mel_spectrogram,
    load_speaker_encoder,
    slerp,
)

# ---------------------------------------------------------------------------
# Shim: inject stub modules so that the local modeling files can be imported
# without pulling in the full vllm / vllm_omni dependency tree.
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.join(
    _REPO_ROOT, "vllm_omni", "model_executor", "models", "qwen3_tts"
)


def _install_stubs():
    """Create minimal stub modules so that ``modeling_qwen3_tts`` can be
    imported without ``vllm`` or ``vllm_omni`` being installed."""
    import importlib
    import types

    # Stub for vllm_omni.model_executor.model_loader.weight_utils
    # The only symbol used is download_weights_from_hf_specific.
    def _download_weights_from_hf_specific(
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
    ]:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)

    sys.modules[
        "vllm_omni.model_executor.model_loader.weight_utils"
    ].download_weights_from_hf_specific = _download_weights_from_hf_specific

    # Make the qwen3_tts package importable as a real package
    pkg_name = "vllm_omni.model_executor.models.qwen3_tts"
    parent_names = [
        "vllm_omni.model_executor.models",
    ]
    for pn in parent_names:
        if pn not in sys.modules:
            sys.modules[pn] = types.ModuleType(pn)

    # Import configuration and modeling from the local files
    if _MODELS_DIR not in sys.path:
        sys.path.insert(0, _MODELS_DIR)

    # The local files use relative imports (from .configuration_qwen3_tts import ...)
    # so we register them as a proper package.
    if pkg_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(_MODELS_DIR, "__init__.py"),
            submodule_search_locations=[_MODELS_DIR],
        )
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg

    return pkg_name


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
# Lightweight TTS wrapper (subset of Qwen3TTSModel from qwen3_tts.py,
# without the vllm dependency).
# ---------------------------------------------------------------------------


class _TTSWrapper:
    """Minimal wrapper around Qwen3TTSForConditionalGeneration for voice cloning."""

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

    # -- text helpers --
    def _tokenize(self, text: str) -> torch.Tensor:
        inp = self.processor(text=text, return_tensors="pt", padding=True)
        ids = inp["input_ids"].to(self.device)
        return ids.unsqueeze(0) if ids.dim() == 1 else ids

    @staticmethod
    def _wrap_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    # -- generation kwargs --
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

    # -- public API --
    @torch.no_grad()
    def generate_with_embedding(
        self,
        text: str,
        language: str | None,
        speaker_embedding: list[float],
    ) -> tuple[np.ndarray, int]:
        """Generate speech using a pre-computed speaker embedding (x-vector only mode)."""
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
        """Generate speech using reference audio (x-vector only, no ICL)."""
        import librosa as _lr

        wav, sr = _lr.load(ref_audio_path, sr=None, mono=True)
        wav = wav.astype(np.float32)

        # Encode reference for speech codes (not used in x-vector-only, but
        # we still need the speaker embedding from the model's own encoder).
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
# Global model handles (lazy-loaded)
# ---------------------------------------------------------------------------
_speaker_encoder: torch.nn.Module | None = None
_tts_model: _TTSWrapper | None = None
_encoder_model_path: str = ""
_tts_model_path: str = ""
_device: str = "cuda:0"


def get_speaker_encoder():
    global _speaker_encoder
    if _speaker_encoder is None:
        print(f"[init] Loading speaker encoder from {_encoder_model_path} ...")
        _speaker_encoder = load_speaker_encoder(_encoder_model_path, device="cpu")
        print("[init] Speaker encoder ready.")
    return _speaker_encoder


def get_tts_model() -> _TTSWrapper:
    global _tts_model
    if _tts_model is None:
        print(f"[init] Loading TTS model from {_tts_model_path} (this may take a while) ...")
        _tts_model = _TTSWrapper.from_pretrained(
            _tts_model_path,
            torch_dtype=torch.bfloat16,
            device_map=_device,
        )
        print("[init] TTS model ready.")
    return _tts_model


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _extract_embedding(audio_path: str) -> np.ndarray:
    import librosa

    encoder = get_speaker_encoder()
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    dev = next(encoder.parameters()).device
    mel = compute_mel_spectrogram(audio, sr).to(dev)
    embedding = encoder(mel.to(next(encoder.parameters()).dtype))[0]
    return embedding.float().cpu().numpy()


# ---------------------------------------------------------------------------
# Tab 1: Voice Embedding Extraction
# ---------------------------------------------------------------------------

def tab1_extract(audio_file):
    if audio_file is None:
        return "Please upload an audio file.", None, None
    try:
        emb = _extract_embedding(audio_file)
        norm = float(np.linalg.norm(emb))
        info = (
            f"Dimensions: {emb.shape[0]}\n"
            f"L2 Norm: {norm:.4f}\n"
            f"Min: {emb.min():.4f}\n"
            f"Max: {emb.max():.4f}"
        )
        tmp = tempfile.NamedTemporaryFile(
            suffix="_embedding.json", delete=False, mode="w"
        )
        json.dump(emb.tolist(), tmp)
        tmp.close()
        return info, tmp.name, emb.tolist()
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}", None, None


# ---------------------------------------------------------------------------
# Tab 2: Voice Cloning TTS
# ---------------------------------------------------------------------------

def tab2_generate(text: str, language: str, ref_audio, embedding_json, embedding_state):
    if not text.strip():
        return None, "Please enter text to synthesize."
    try:
        model = get_tts_model()

        # Priority: uploaded JSON > Tab1 state > ref audio
        if embedding_json is not None:
            with open(embedding_json, "r") as f:
                emb = json.load(f)
            wav, sr = model.generate_with_embedding(text, language, emb)
            return (sr, wav.astype(np.float32)), f"Generated using uploaded embedding ({len(emb)} dims)"

        if embedding_state is not None and len(embedding_state) > 0:
            wav, sr = model.generate_with_embedding(text, language, embedding_state)
            return (sr, wav.astype(np.float32)), f"Generated using Tab 1 embedding ({len(embedding_state)} dims)"

        if ref_audio is not None:
            wav, sr = model.generate_with_ref_audio(text, language, ref_audio)
            return (sr, wav.astype(np.float32)), "Generated using reference audio (x-vector mode)"

        return None, "Please provide reference audio OR an embedding JSON file."
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tab 3: Voice Interpolation
# ---------------------------------------------------------------------------

def tab3_interpolate(audio_a, audio_b, ratio: float, text: str, language: str):
    if audio_a is None or audio_b is None:
        return None, "Please upload both Voice A and Voice B audio files."
    if not text.strip():
        return None, "Please enter text to synthesize."
    try:
        emb_a = _extract_embedding(audio_a)
        emb_b = _extract_embedding(audio_b)
        blended = slerp(emb_a, emb_b, ratio)

        model = get_tts_model()
        wav, sr = model.generate_with_embedding(text, language, blended.tolist())

        info = (
            f"Voice A norm: {np.linalg.norm(emb_a):.4f}\n"
            f"Voice B norm: {np.linalg.norm(emb_b):.4f}\n"
            f"SLERP ratio: {ratio:.2f} (0=A, 1=B)\n"
            f"Blended norm: {np.linalg.norm(blended):.4f}"
        )
        return (sr, wav.astype(np.float32)), info
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Qwen3 TTS Voice Cloning", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            "# Qwen3 TTS Voice Cloning\n"
            "Extract speaker embeddings, clone voices, and interpolate between speakers."
        )

        embedding_state = gr.State(value=None)

        # -- Tab 1 ---------------------------------------------------------
        with gr.Tab("1. Extract Embedding"):
            gr.Markdown(
                "Upload a reference audio file to extract a speaker embedding "
                "vector (ECAPA-TDNN)."
            )
            with gr.Row():
                with gr.Column():
                    t1_audio = gr.Audio(label="Reference Audio", type="filepath")
                    t1_btn = gr.Button("Extract Embedding", variant="primary")
                with gr.Column():
                    t1_info = gr.Textbox(
                        label="Embedding Info", lines=5, interactive=False
                    )
                    t1_download = gr.File(label="Download Embedding JSON")

            t1_btn.click(
                fn=tab1_extract,
                inputs=[t1_audio],
                outputs=[t1_info, t1_download, embedding_state],
            )

        # -- Tab 2 ---------------------------------------------------------
        with gr.Tab("2. Voice Cloning TTS"):
            gr.Markdown(
                "Generate speech using a cloned voice.\n\n"
                "Voice identity source (highest priority first):\n"
                "1. **Embedding JSON** uploaded below\n"
                "2. **Embedding from Tab 1** (auto-forwarded)\n"
                "3. **Reference audio** (direct x-vector cloning)"
            )
            with gr.Row():
                with gr.Column():
                    t2_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=3,
                        placeholder="Enter text here...",
                    )
                    t2_lang = gr.Dropdown(
                        choices=LANGUAGES, value="Auto", label="Language"
                    )
                    t2_ref_audio = gr.Audio(
                        label="Reference Audio (optional)", type="filepath"
                    )
                    t2_emb_json = gr.File(
                        label="Embedding JSON (optional)",
                        file_types=[".json"],
                    )
                    t2_btn = gr.Button("Generate Speech", variant="primary")
                with gr.Column():
                    t2_output = gr.Audio(label="Generated Speech", type="numpy")
                    t2_status = gr.Textbox(label="Status", interactive=False)

            t2_btn.click(
                fn=tab2_generate,
                inputs=[t2_text, t2_lang, t2_ref_audio, t2_emb_json, embedding_state],
                outputs=[t2_output, t2_status],
            )

        # -- Tab 3 ---------------------------------------------------------
        with gr.Tab("3. Voice Interpolation"):
            gr.Markdown(
                "Blend two voices using SLERP (Spherical Linear Interpolation).\n\n"
                "Upload two reference audio files, adjust the blend ratio, "
                "and generate speech."
            )
            with gr.Row():
                with gr.Column():
                    t3_audio_a = gr.Audio(label="Voice A", type="filepath")
                    t3_audio_b = gr.Audio(label="Voice B", type="filepath")
                    t3_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="SLERP Ratio (0=A, 1=B)",
                    )
                    t3_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=3,
                        placeholder="Enter text here...",
                    )
                    t3_lang = gr.Dropdown(
                        choices=LANGUAGES, value="Auto", label="Language"
                    )
                    t3_btn = gr.Button(
                        "Generate Blended Voice", variant="primary"
                    )
                with gr.Column():
                    t3_output = gr.Audio(label="Generated Speech", type="numpy")
                    t3_info = gr.Textbox(
                        label="Interpolation Info", lines=5, interactive=False
                    )

            t3_btn.click(
                fn=tab3_interpolate,
                inputs=[t3_audio_a, t3_audio_b, t3_ratio, t3_text, t3_lang],
                outputs=[t3_output, t3_info],
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gradio GUI for Qwen3-TTS voice cloning"
    )
    parser.add_argument(
        "--encoder-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="HF model ID or local path for the speaker encoder",
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
        "--port", type=int, default=7860, help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    return parser.parse_args()


def main():
    global _encoder_model_path, _tts_model_path, _device

    args = parse_args()
    _encoder_model_path = args.encoder_model
    _tts_model_path = args.tts_model
    _device = args.device

    # Eagerly load the speaker encoder (lightweight, ~12M params on CPU)
    get_speaker_encoder()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
