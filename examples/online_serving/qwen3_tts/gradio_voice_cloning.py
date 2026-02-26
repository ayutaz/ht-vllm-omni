"""Gradio GUI for Qwen3-TTS voice cloning with speaker embeddings.

Provides seven tabs:
  1. Voice Embedding extraction (ECAPA-TDNN speaker encoder)
  2. Voice Cloning TTS (text-to-speech with ref audio or saved embedding)
  3. Voice Interpolation (SLERP blend of two voices)
  4. Voice Library (persistent collection of named embeddings)
  5. Vector Arithmetic (average, add, subtract, scale, custom formula)
  6. Direction Vectors (gender, pitch, emotion transforms)
  7. Embedding Space Visualization (t-SNE / PCA scatter plot)

Requirements:
    pip install gradio torch librosa soundfile numpy transformers scikit-learn pandas

Usage:
    python examples/online_serving/qwen3_tts/gradio_voice_cloning.py

    # Use a different TTS model or device
    python examples/online_serving/qwen3_tts/gradio_voice_cloning.py \
        --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# SLERP from the existing utility
from examples.online_serving.qwen3_tts.speaker_embedding_interpolation import slerp

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
# Uses marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B (~12M params)
# ---------------------------------------------------------------------------

_emb_encoder: torch.nn.Module | None = None
_emb_feature_extractor = None
_emb_model_id: str = ""


def get_embedding_encoder():
    """Return (or lazy-load) the ECAPA-TDNN speaker encoder + feature extractor."""
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
    """Extract a 2048-dim speaker embedding from an audio file."""
    import librosa

    encoder, fe = get_embedding_encoder()
    audio, sr = librosa.load(audio_path, sr=fe.sampling_rate, mono=True)
    inputs = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt")
    device = next(encoder.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = encoder(**inputs)
    return output.last_hidden_state[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# TTS Wrapper (lightweight subset of Qwen3TTSModel, no vllm dependency)
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
        """Generate speech using a pre-computed speaker embedding."""
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
        """Generate speech using reference audio (x-vector only)."""
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
        print(f"[init] Loading TTS model from {_tts_model_path} (this may take a while) ...")
        _tts_model = _TTSWrapper.from_pretrained(
            _tts_model_path,
            torch_dtype=torch.bfloat16,
            device_map=_device,
        )
        print("[init] TTS model ready.")
    return _tts_model


# ---------------------------------------------------------------------------
# VoiceLibrary: persistent collection of named speaker embeddings
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VoiceEntry:
    id: str
    name: str
    tags: list[str]
    source: str  # "audio", "json", "tab1", "arithmetic", "direction"
    source_detail: str
    created_at: str
    embedding: list[float]


class VoiceLibrary:
    """Manages a persistent JSON file of named speaker embeddings."""

    def __init__(self, path: str):
        self.path = path
        self._entries: dict[str, VoiceEntry] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                entry = VoiceEntry(**item)
                self._entries[entry.id] = entry

    def _save(self):
        data = [dataclasses.asdict(e) for e in self._entries.values()]
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(
        self,
        name: str,
        embedding: list[float],
        tags: list[str] | None = None,
        source: str = "unknown",
        source_detail: str = "",
    ) -> VoiceEntry:
        entry = VoiceEntry(
            id=uuid.uuid4().hex[:12],
            name=name,
            tags=tags or [],
            source=source,
            source_detail=source_detail,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            embedding=embedding,
        )
        self._entries[entry.id] = entry
        self._save()
        return entry

    def remove(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._save()
            return True
        return False

    def get_entry(self, entry_id: str) -> VoiceEntry | None:
        return self._entries.get(entry_id)

    def get_embedding(self, entry_id: str) -> np.ndarray | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        return np.array(entry.embedding, dtype=np.float32)

    def list_entries(self) -> list[VoiceEntry]:
        return list(self._entries.values())

    def find_similar(
        self, query: np.ndarray, top_k: int = 5
    ) -> list[tuple[VoiceEntry, float]]:
        """Find entries most similar to *query* by cosine similarity."""
        q_norm = query / (np.linalg.norm(query) + 1e-9)
        results: list[tuple[VoiceEntry, float]] = []
        for entry in self._entries.values():
            emb = np.array(entry.embedding, dtype=np.float32)
            e_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sim = float(np.dot(q_norm, e_norm))
            results.append((entry, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(columns=["ID", "名前", "タグ", "ソース", "次元数", "作成日時"])
        rows = []
        for e in self._entries.values():
            rows.append({
                "ID": e.id,
                "名前": e.name,
                "タグ": ", ".join(e.tags),
                "ソース": e.source,
                "次元数": len(e.embedding),
                "作成日時": e.created_at,
            })
        return pd.DataFrame(rows)

    def import_json(self, json_path: str) -> int:
        """Import entries from an exported JSON file. Returns count imported."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data:
            entry = VoiceEntry(**item)
            if entry.id not in self._entries:
                self._entries[entry.id] = entry
                count += 1
        self._save()
        return count

    def export_json(self, output_path: str) -> str:
        data = [dataclasses.asdict(e) for e in self._entries.values()]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return output_path

    def __len__(self) -> int:
        return len(self._entries)


# Global voice library instance
_voice_library: VoiceLibrary | None = None


def get_voice_library() -> VoiceLibrary:
    assert _voice_library is not None, "Voice library not initialized"
    return _voice_library


def get_library_choices() -> list[tuple[str, str]]:
    """Return (label, entry_id) pairs for dropdown choices."""
    lib = get_voice_library()
    return [(f"{e.name} [{e.id}]", e.id) for e in lib.list_entries()]


def get_direction_vector_choices() -> list[tuple[str, str]]:
    """Return choices for entries tagged with 'direction'."""
    lib = get_voice_library()
    return [
        (f"{e.name} [{e.id}]", e.id)
        for e in lib.list_entries()
        if "direction" in e.tags
    ]


# ---------------------------------------------------------------------------
# Tab 1: Voice Embedding Extraction
# ---------------------------------------------------------------------------

def tab1_extract(audio_file):
    if audio_file is None:
        return "音声ファイルをアップロードしてください。", None, None
    try:
        emb = _extract_embedding(audio_file)
        norm = float(np.linalg.norm(emb))
        info = (
            f"次元数: {emb.shape[0]}\n"
            f"L2 ノルム: {norm:.4f}\n"
            f"最小値: {emb.min():.4f}\n"
            f"最大値: {emb.max():.4f}"
        )
        tmp = tempfile.NamedTemporaryFile(
            suffix="_embedding.json", delete=False, mode="w"
        )
        json.dump(emb.tolist(), tmp)
        tmp.close()
        return info, tmp.name, emb.tolist()
    except Exception as e:
        return f"エラー: {e}\n{traceback.format_exc()}", None, None


# ---------------------------------------------------------------------------
# Tab 2: Voice Cloning TTS
# ---------------------------------------------------------------------------

def tab2_generate(text: str, language: str, ref_audio, embedding_json, embedding_state):
    if not text.strip():
        return None, "合成するテキストを入力してください。"
    try:
        model = get_tts_model()

        # 優先順位: アップロードJSON > タブ1のstate > 参照音声
        if embedding_json is not None:
            with open(embedding_json, "r") as f:
                emb = json.load(f)
            wav, sr = model.generate_with_embedding(text, language, emb)
            return (sr, wav.astype(np.float32)), f"アップロードされたEmbeddingで生成 ({len(emb)}次元)"

        if embedding_state is not None and len(embedding_state) > 0:
            wav, sr = model.generate_with_embedding(text, language, embedding_state)
            return (sr, wav.astype(np.float32)), f"タブ1のEmbeddingで生成 ({len(embedding_state)}次元)"

        if ref_audio is not None:
            wav, sr = model.generate_with_ref_audio(text, language, ref_audio)
            return (sr, wav.astype(np.float32)), "参照音声で生成 (x-vectorモード)"

        return None, "参照音声またはEmbedding JSONを指定してください。"
    except Exception as e:
        return None, f"エラー: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tab 3: Voice Interpolation
# ---------------------------------------------------------------------------

def tab3_interpolate(audio_a, audio_b, ratio: float, text: str, language: str):
    if audio_a is None or audio_b is None:
        return None, "声A と 声B の両方の音声ファイルをアップロードしてください。"
    if not text.strip():
        return None, "合成するテキストを入力してください。"
    try:
        emb_a = _extract_embedding(audio_a)
        emb_b = _extract_embedding(audio_b)
        blended = slerp(emb_a, emb_b, ratio)

        model = get_tts_model()
        wav, sr = model.generate_with_embedding(text, language, blended.tolist())

        info = (
            f"声A ノルム: {np.linalg.norm(emb_a):.4f}\n"
            f"声B ノルム: {np.linalg.norm(emb_b):.4f}\n"
            f"SLERP 比率: {ratio:.2f} (0=A, 1=B)\n"
            f"ブレンド後ノルム: {np.linalg.norm(blended):.4f}"
        )
        return (sr, wav.astype(np.float32)), info
    except Exception as e:
        return None, f"エラー: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tab 4: Voice Library callbacks
# ---------------------------------------------------------------------------

def tab4_register_from_audio(audio_file, name: str, tags_str: str):
    """Register a voice from an audio file."""
    if audio_file is None:
        return "音声ファイルをアップロードしてください。", _lib_df(), _lib_choices()
    if not name.strip():
        return "名前を入力してください。", _lib_df(), _lib_choices()
    try:
        emb = _extract_embedding(audio_file)
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        lib = get_voice_library()
        entry = lib.add(name.strip(), emb.tolist(), tags=tags, source="audio", source_detail=os.path.basename(audio_file))
        return f"登録完了: {entry.name} ({entry.id})", _lib_df(), _lib_choices()
    except Exception as e:
        return f"エラー: {e}", _lib_df(), _lib_choices()


def tab4_register_from_json(json_file, name: str, tags_str: str):
    """Register a voice from a JSON embedding file."""
    if json_file is None:
        return "JSONファイルをアップロードしてください。", _lib_df(), _lib_choices()
    if not name.strip():
        return "名前を入力してください。", _lib_df(), _lib_choices()
    try:
        with open(json_file, "r") as f:
            emb = json.load(f)
        if not isinstance(emb, list):
            return "JSONはfloatの配列である必要があります。", _lib_df(), _lib_choices()
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        lib = get_voice_library()
        entry = lib.add(name.strip(), emb, tags=tags, source="json", source_detail=os.path.basename(json_file))
        return f"登録完了: {entry.name} ({entry.id})", _lib_df(), _lib_choices()
    except Exception as e:
        return f"エラー: {e}", _lib_df(), _lib_choices()


def tab4_register_from_state(name: str, tags_str: str, embedding_state):
    """Register a voice from Tab 1's embedding state."""
    if embedding_state is None or len(embedding_state) == 0:
        return "タブ1でEmbeddingを抽出してください。", _lib_df(), _lib_choices()
    if not name.strip():
        return "名前を入力してください。", _lib_df(), _lib_choices()
    try:
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        lib = get_voice_library()
        entry = lib.add(name.strip(), list(embedding_state), tags=tags, source="tab1", source_detail="Tab1抽出")
        return f"登録完了: {entry.name} ({entry.id})", _lib_df(), _lib_choices()
    except Exception as e:
        return f"エラー: {e}", _lib_df(), _lib_choices()


def tab4_delete(entry_id: str):
    if not entry_id:
        return "IDを選択してください。", _lib_df(), _lib_choices()
    lib = get_voice_library()
    if lib.remove(entry_id):
        return f"削除完了: {entry_id}", _lib_df(), _lib_choices()
    return f"ID {entry_id} が見つかりません。", _lib_df(), _lib_choices()


def tab4_export():
    lib = get_voice_library()
    if len(lib) == 0:
        return "ライブラリが空です。", None
    tmp = tempfile.NamedTemporaryFile(suffix="_voice_library.json", delete=False, mode="w")
    tmp.close()
    lib.export_json(tmp.name)
    return f"{len(lib)}件エクスポート完了", tmp.name


def tab4_import(json_file):
    if json_file is None:
        return "JSONファイルをアップロードしてください。", _lib_df(), _lib_choices()
    try:
        lib = get_voice_library()
        count = lib.import_json(json_file)
        return f"{count}件インポート完了", _lib_df(), _lib_choices()
    except Exception as e:
        return f"エラー: {e}", _lib_df(), _lib_choices()


def tab4_search(query_audio, top_k: int):
    if query_audio is None:
        return "クエリ音声をアップロードしてください。"
    try:
        lib = get_voice_library()
        if len(lib) == 0:
            return "ライブラリが空です。"
        query_emb = _extract_embedding(query_audio)
        results = lib.find_similar(query_emb, top_k=int(top_k))
        lines = ["## 類似検索結果\n"]
        for i, (entry, sim) in enumerate(results, 1):
            lines.append(f"{i}. **{entry.name}** (ID: {entry.id}) — 類似度: {sim:.4f}  タグ: {', '.join(entry.tags) or '-'}")
        return "\n".join(lines)
    except Exception as e:
        return f"エラー: {e}"


def tab4_use_in_tts(entry_id: str):
    """Load a library embedding into the shared state for TTS."""
    if not entry_id:
        return None, "IDを選択してください。"
    lib = get_voice_library()
    emb = lib.get_embedding(entry_id)
    if emb is None:
        return None, f"ID {entry_id} が見つかりません。"
    entry = lib.get_entry(entry_id)
    return emb.tolist(), f"「{entry.name}」をTTS用にロードしました。タブ2で使用できます。"


def _lib_df() -> pd.DataFrame:
    return get_voice_library().to_dataframe()


def _lib_choices():
    return gr.update(choices=get_library_choices())


def _refresh_all_dropdowns():
    """Return gr.update for all library-based dropdowns (14 total)."""
    choices = get_library_choices()
    dir_choices = get_direction_vector_choices()
    upd = gr.update(choices=choices)
    return (
        upd,  # tab4 delete dropdown
        upd,  # tab4 use_in_tts dropdown
        upd,  # tab5 add_a
        upd,  # tab5 add_b
        upd,  # tab5 sub_a
        upd,  # tab5 sub_b
        upd,  # tab5 scale_a
        upd,  # tab5 cust_a
        upd,  # tab5 cust_b
        upd,  # tab5 cust_c
        upd,  # tab5 avg_voices
        upd,  # tab6 base voice
        upd,  # tab6 group_a
        upd,  # tab6 group_b
        gr.update(choices=dir_choices),  # tab6 saved direction
    )


# ---------------------------------------------------------------------------
# Tab 5: Vector Arithmetic callbacks
# ---------------------------------------------------------------------------

def _resolve_emb(entry_id: str) -> np.ndarray | None:
    lib = get_voice_library()
    return lib.get_embedding(entry_id)


def tab5_average(selected_ids: list[str], normalize: bool):
    if not selected_ids or len(selected_ids) < 2:
        return "2つ以上の声を選択してください。", None
    embeddings = []
    for eid in selected_ids:
        emb = _resolve_emb(eid)
        if emb is None:
            return f"ID {eid} が見つかりません。", None
        embeddings.append(emb)
    result = np.mean(embeddings, axis=0)
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    info = f"平均: {len(embeddings)}声 → {result.shape[0]}次元, ノルム: {np.linalg.norm(result):.4f}"
    return info, result.tolist()


def tab5_add(id_a: str, id_b: str, normalize: bool):
    a, b = _resolve_emb(id_a), _resolve_emb(id_b)
    if a is None or b is None:
        return "声A, 声B を選択してください。", None
    result = a + b
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    return f"A + B → ノルム: {np.linalg.norm(result):.4f}", result.tolist()


def tab5_subtract(id_a: str, id_b: str, normalize: bool):
    a, b = _resolve_emb(id_a), _resolve_emb(id_b)
    if a is None or b is None:
        return "声A, 声B を選択してください。", None
    result = a - b
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    return f"A - B → ノルム: {np.linalg.norm(result):.4f}", result.tolist()


def tab5_scale(id_a: str, alpha: float, normalize: bool):
    a = _resolve_emb(id_a)
    if a is None:
        return "声A を選択してください。", None
    result = a * alpha
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    return f"A * {alpha:.2f} → ノルム: {np.linalg.norm(result):.4f}", result.tolist()


def tab5_custom(id_a: str, id_b: str, id_c: str, alpha: float, normalize: bool):
    a, b, c = _resolve_emb(id_a), _resolve_emb(id_b), _resolve_emb(id_c)
    if a is None or b is None or c is None:
        return "声A, 声B, 声C をすべて選択してください。", None
    result = a + alpha * (b - c)
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    return f"A + {alpha:.2f}*(B - C) → ノルム: {np.linalg.norm(result):.4f}", result.tolist()


def tab5_save_result(name: str, tags_str: str, result_state):
    if result_state is None:
        return "先に演算を実行してください。"
    if not name.strip():
        return "名前を入力してください。"
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    lib = get_voice_library()
    entry = lib.add(name.strip(), result_state, tags=tags, source="arithmetic", source_detail="Tab5演算結果")
    return f"保存完了: {entry.name} ({entry.id})"


def tab5_tts_preview(text: str, language: str, result_state):
    if result_state is None:
        return None, "先に演算を実行してください。"
    if not text.strip():
        return None, "テキストを入力してください。"
    try:
        model = get_tts_model()
        wav, sr = model.generate_with_embedding(text, language, result_state)
        return (sr, wav.astype(np.float32)), "演算結果のEmbeddingで生成完了"
    except Exception as e:
        return None, f"エラー: {e}"


# ---------------------------------------------------------------------------
# Tab 6: Direction Vector callbacks
# ---------------------------------------------------------------------------

def tab6_compute_direction(group_a_ids: list[str], group_b_ids: list[str], dir_name: str, save_to_lib: bool):
    if not group_a_ids or not group_b_ids:
        return "グループA/Bにそれぞれ1つ以上の声を選択してください。", None
    embs_a = [_resolve_emb(eid) for eid in group_a_ids]
    embs_b = [_resolve_emb(eid) for eid in group_b_ids]
    if any(e is None for e in embs_a) or any(e is None for e in embs_b):
        return "一部のIDが見つかりません。", None
    mean_a = np.mean(embs_a, axis=0)
    mean_b = np.mean(embs_b, axis=0)
    direction = mean_b - mean_a
    info = (
        f"方向ベクトル計算完了\n"
        f"  グループA: {len(embs_a)}声 → mean ノルム: {np.linalg.norm(mean_a):.4f}\n"
        f"  グループB: {len(embs_b)}声 → mean ノルム: {np.linalg.norm(mean_b):.4f}\n"
        f"  方向ベクトル ノルム: {np.linalg.norm(direction):.4f}"
    )
    if save_to_lib and dir_name.strip():
        lib = get_voice_library()
        lib.add(
            dir_name.strip(), direction.tolist(),
            tags=["direction"],
            source="direction",
            source_detail=f"mean(B:{len(embs_b)}) - mean(A:{len(embs_a)})",
        )
        info += f"\nライブラリに「{dir_name.strip()}」として保存しました。"
    return info, direction.tolist()


def tab6_apply_direction(base_id: str, direction_state, direction_saved_id: str, intensity: float, normalize: bool):
    base = _resolve_emb(base_id) if base_id else None
    if base is None:
        return "ベース声を選択してください。", None

    # Prefer computed direction from state, fallback to saved
    direction = None
    if direction_state is not None:
        direction = np.array(direction_state, dtype=np.float32)
    elif direction_saved_id:
        direction = _resolve_emb(direction_saved_id)

    if direction is None:
        return "方向ベクトルを計算するか、保存済みの方向を選択してください。", None

    result = base + intensity * direction
    if normalize:
        result = result / (np.linalg.norm(result) + 1e-9)
    info = (
        f"ベース ノルム: {np.linalg.norm(base):.4f}\n"
        f"方向ノルム: {np.linalg.norm(direction):.4f}\n"
        f"強度: {intensity:.2f}\n"
        f"結果ノルム: {np.linalg.norm(result):.4f}"
    )
    return info, result.tolist()


def tab6_save_result(name: str, tags_str: str, result_state):
    if result_state is None:
        return "先に方向ベクトルを適用してください。"
    if not name.strip():
        return "名前を入力してください。"
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    lib = get_voice_library()
    entry = lib.add(name.strip(), result_state, tags=tags, source="direction", source_detail="Tab6変換結果")
    return f"保存完了: {entry.name} ({entry.id})"


def tab6_tts_preview(text: str, language: str, result_state):
    if result_state is None:
        return None, "先に方向ベクトルを適用してください。"
    if not text.strip():
        return None, "テキストを入力してください。"
    try:
        model = get_tts_model()
        wav, sr = model.generate_with_embedding(text, language, result_state)
        return (sr, wav.astype(np.float32)), "方向ベクトル適用後のEmbeddingで生成完了"
    except Exception as e:
        return None, f"エラー: {e}"


# ---------------------------------------------------------------------------
# Tab 7: Visualization callbacks
# ---------------------------------------------------------------------------

def tab7_visualize(method: str, perplexity: int, random_seed: int):
    lib = get_voice_library()
    entries = lib.list_entries()
    if len(entries) < 2:
        return None, "可視化には2つ以上のライブラリエントリが必要です。"

    embeddings = np.array([e.embedding for e in entries], dtype=np.float32)
    names = [e.name for e in entries]
    tags = [", ".join(e.tags) if e.tags else "none" for e in entries]
    sources = [e.source for e in entries]

    if method == "t-SNE":
        from sklearn.manifold import TSNE
        n_samples = len(entries)
        perp = min(perplexity, max(1, n_samples - 1))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=random_seed)
        coords = reducer.fit_transform(embeddings)
    else:  # PCA
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_seed)
        coords = reducer.fit_transform(embeddings)

    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "名前": names,
        "タグ": tags,
        "ソース": sources,
    })

    info = (
        f"手法: {method}\n"
        f"エントリ数: {len(entries)}\n"
        f"次元: {embeddings.shape[1]}D → 2D"
    )
    if method == "t-SNE":
        info += f"\nPerplexity: {perp}, Seed: {random_seed}"
    else:
        info += f"\nSeed: {random_seed}"

    return df, info


# ---------------------------------------------------------------------------
# Gradio UI builder functions (one per tab)
# ---------------------------------------------------------------------------

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish",
]


def _build_tab1(embedding_state: gr.State):
    """Tab 1: Embedding Extraction."""
    with gr.Tab("1. Embedding抽出"):
        gr.Markdown(
            "参照音声ファイルをアップロードして、話者Embeddingベクトル"
            "（ECAPA-TDNN）を抽出します。"
        )
        with gr.Row():
            with gr.Column():
                t1_audio = gr.Audio(label="参照音声", type="filepath")
                t1_btn = gr.Button("Embedding抽出", variant="primary")
            with gr.Column():
                t1_info = gr.Textbox(
                    label="Embedding情報", lines=5, interactive=False
                )
                t1_download = gr.File(label="Embedding JSONダウンロード")

        t1_btn.click(
            fn=tab1_extract,
            inputs=[t1_audio],
            outputs=[t1_info, t1_download, embedding_state],
        )


def _build_tab2(embedding_state: gr.State):
    """Tab 2: Voice Cloning TTS."""
    with gr.Tab("2. ボイスクローンTTS"):
        gr.Markdown(
            "クローンした声で音声を生成します。\n\n"
            "声の指定方法（優先順位順）:\n"
            "1. **Embedding JSON** を下からアップロード\n"
            "2. **タブ1のEmbedding**（自動連携）\n"
            "3. **参照音声**（直接x-vectorクローニング）"
        )
        with gr.Row():
            with gr.Column():
                t2_text = gr.Textbox(
                    label="合成テキスト",
                    lines=3,
                    placeholder="テキストを入力してください...",
                )
                t2_lang = gr.Dropdown(
                    choices=LANGUAGES, value="Auto", label="言語"
                )
                t2_ref_audio = gr.Audio(
                    label="参照音声（任意）", type="filepath"
                )
                t2_emb_json = gr.File(
                    label="Embedding JSON（任意）",
                    file_types=[".json"],
                )
                t2_btn = gr.Button("音声生成", variant="primary")
            with gr.Column():
                t2_output = gr.Audio(label="生成音声", type="numpy")
                t2_status = gr.Textbox(label="ステータス", interactive=False)

        t2_btn.click(
            fn=tab2_generate,
            inputs=[t2_text, t2_lang, t2_ref_audio, t2_emb_json, embedding_state],
            outputs=[t2_output, t2_status],
        )


def _build_tab3():
    """Tab 3: Voice Interpolation."""
    with gr.Tab("3. 声の補間"):
        gr.Markdown(
            "SLERP（球面線形補間）で2つの声をブレンドします。\n\n"
            "2つの参照音声をアップロードし、ブレンド比率を調整して"
            "音声を生成します。"
        )
        with gr.Row():
            with gr.Column():
                t3_audio_a = gr.Audio(label="声A", type="filepath")
                t3_audio_b = gr.Audio(label="声B", type="filepath")
                t3_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="SLERP比率 (0=A, 1=B)",
                )
                t3_text = gr.Textbox(
                    label="合成テキスト",
                    lines=3,
                    placeholder="テキストを入力してください...",
                )
                t3_lang = gr.Dropdown(
                    choices=LANGUAGES, value="Auto", label="言語"
                )
                t3_btn = gr.Button(
                    "ブレンド音声生成", variant="primary"
                )
            with gr.Column():
                t3_output = gr.Audio(label="生成音声", type="numpy")
                t3_info = gr.Textbox(
                    label="補間情報", lines=5, interactive=False
                )

        t3_btn.click(
            fn=tab3_interpolate,
            inputs=[t3_audio_a, t3_audio_b, t3_ratio, t3_text, t3_lang],
            outputs=[t3_output, t3_info],
        )


def _build_tab4(embedding_state: gr.State):
    """Tab 4: Voice Library. Returns dropdown components for cross-tab updates."""
    with gr.Tab("4. ボイスライブラリ"):
        gr.Markdown(
            "話者Embeddingをライブラリに保存・管理します。\n"
            "登録した声はタブ2のTTS、タブ5の演算、タブ6の方向ベクトルで使用できます。"
        )

        t4_status = gr.Textbox(label="ステータス", interactive=False)
        t4_table = gr.Dataframe(
            value=_lib_df,
            label="ライブラリ一覧",
            interactive=False,
        )

        # --- Registration ---
        with gr.Accordion("声の登録", open=True):
            with gr.Tab("音声から"):
                t4_reg_audio = gr.Audio(label="音声ファイル", type="filepath")
                t4_reg_audio_name = gr.Textbox(label="名前", placeholder="例: 女性A")
                t4_reg_audio_tags = gr.Textbox(label="タグ（カンマ区切り）", placeholder="例: female, soft")
                t4_reg_audio_btn = gr.Button("音声から登録")
            with gr.Tab("JSONから"):
                t4_reg_json_file = gr.File(label="Embedding JSON", file_types=[".json"])
                t4_reg_json_name = gr.Textbox(label="名前")
                t4_reg_json_tags = gr.Textbox(label="タグ（カンマ区切り）")
                t4_reg_json_btn = gr.Button("JSONから登録")
            with gr.Tab("タブ1から"):
                t4_reg_state_name = gr.Textbox(label="名前")
                t4_reg_state_tags = gr.Textbox(label="タグ（カンマ区切り）")
                t4_reg_state_btn = gr.Button("タブ1の結果を登録")

        # --- Management ---
        with gr.Row():
            with gr.Column():
                t4_del_dropdown = gr.Dropdown(
                    label="削除するエントリ", choices=get_library_choices(), interactive=True
                )
                t4_del_btn = gr.Button("削除", variant="stop")
            with gr.Column():
                t4_use_dropdown = gr.Dropdown(
                    label="TTS用にロードするエントリ", choices=get_library_choices(), interactive=True
                )
                t4_use_btn = gr.Button("タブ2にロード")
                t4_use_status = gr.Textbox(label="ロード結果", interactive=False)

        # --- Import / Export ---
        with gr.Row():
            with gr.Column():
                t4_export_btn = gr.Button("ライブラリをエクスポート")
                t4_export_file = gr.File(label="エクスポートファイル")
            with gr.Column():
                t4_import_file = gr.File(label="インポートJSON", file_types=[".json"])
                t4_import_btn = gr.Button("インポート")

        # --- Semantic Search ---
        with gr.Accordion("セマンティック類似検索", open=False):
            t4_search_audio = gr.Audio(label="クエリ音声", type="filepath")
            t4_search_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top-K")
            t4_search_btn = gr.Button("類似検索")
            t4_search_result = gr.Markdown(label="検索結果")

        # --- Wiring ---
        # We need to collect all dropdown components that depend on library state
        # These will be returned so build_ui can wire cross-tab refresh
        dropdowns = {
            "t4_del": t4_del_dropdown,
            "t4_use": t4_use_dropdown,
        }

        def _register_audio_and_refresh(audio, name, tags):
            status, df, _ = tab4_register_from_audio(audio, name, tags)
            return (status, df) + _refresh_all_dropdowns()

        def _register_json_and_refresh(jf, name, tags):
            status, df, _ = tab4_register_from_json(jf, name, tags)
            return (status, df) + _refresh_all_dropdowns()

        def _register_state_and_refresh(name, tags, emb_state):
            status, df, _ = tab4_register_from_state(name, tags, emb_state)
            return (status, df) + _refresh_all_dropdowns()

        def _delete_and_refresh(entry_id):
            status, df, _ = tab4_delete(entry_id)
            return (status, df) + _refresh_all_dropdowns()

        def _import_and_refresh(jf):
            status, df, _ = tab4_import(jf)
            return (status, df) + _refresh_all_dropdowns()

        # Store wiring functions and outputs as attributes on tab4 components
        # so build_ui can connect them once all tabs' dropdowns are available
        dropdowns["_reg_audio_btn"] = t4_reg_audio_btn
        dropdowns["_reg_audio_inputs"] = [t4_reg_audio, t4_reg_audio_name, t4_reg_audio_tags]
        dropdowns["_reg_audio_fn"] = _register_audio_and_refresh

        dropdowns["_reg_json_btn"] = t4_reg_json_btn
        dropdowns["_reg_json_inputs"] = [t4_reg_json_file, t4_reg_json_name, t4_reg_json_tags]
        dropdowns["_reg_json_fn"] = _register_json_and_refresh

        dropdowns["_reg_state_btn"] = t4_reg_state_btn
        dropdowns["_reg_state_inputs"] = [t4_reg_state_name, t4_reg_state_tags, embedding_state]
        dropdowns["_reg_state_fn"] = _register_state_and_refresh

        dropdowns["_del_btn"] = t4_del_btn
        dropdowns["_del_input"] = t4_del_dropdown
        dropdowns["_del_fn"] = _delete_and_refresh

        dropdowns["_import_btn"] = t4_import_btn
        dropdowns["_import_input"] = t4_import_file
        dropdowns["_import_fn"] = _import_and_refresh

        # Non-refresh wiring (export, search, use)
        t4_export_btn.click(fn=tab4_export, inputs=[], outputs=[t4_status, t4_export_file])

        t4_search_btn.click(
            fn=tab4_search, inputs=[t4_search_audio, t4_search_k], outputs=[t4_search_result]
        )

        t4_use_btn.click(
            fn=tab4_use_in_tts,
            inputs=[t4_use_dropdown],
            outputs=[embedding_state, t4_use_status],
        )

        return t4_status, t4_table, dropdowns


def _build_tab5(embedding_state: gr.State):
    """Tab 5: Vector Arithmetic. Returns dropdown components."""
    with gr.Tab("5. ベクトル演算"):
        gr.Markdown(
            "話者Embeddingの算術演算を行います。\n"
            "ライブラリの声を使って平均・加算・減算・スケーリング・カスタム式が使えます。"
        )

        t5_result_state = gr.State(value=None)
        t5_status = gr.Textbox(label="演算結果", interactive=False)
        t5_normalize = gr.Checkbox(label="結果を正規化（L2ノルム=1）", value=False)

        lib_choices = get_library_choices()

        with gr.Tab("平均 (N声)"):
            t5_avg_voices = gr.Dropdown(
                label="平均する声（複数選択）", choices=lib_choices,
                multiselect=True, interactive=True,
            )
            t5_avg_btn = gr.Button("平均を計算")
            t5_avg_btn.click(
                fn=tab5_average,
                inputs=[t5_avg_voices, t5_normalize],
                outputs=[t5_status, t5_result_state],
            )

        with gr.Tab("加算 (A + B)"):
            t5_add_a = gr.Dropdown(label="声A", choices=lib_choices, interactive=True)
            t5_add_b = gr.Dropdown(label="声B", choices=lib_choices, interactive=True)
            t5_add_btn = gr.Button("A + B")
            t5_add_btn.click(
                fn=tab5_add,
                inputs=[t5_add_a, t5_add_b, t5_normalize],
                outputs=[t5_status, t5_result_state],
            )

        with gr.Tab("減算 (A - B)"):
            t5_sub_a = gr.Dropdown(label="声A", choices=lib_choices, interactive=True)
            t5_sub_b = gr.Dropdown(label="声B", choices=lib_choices, interactive=True)
            t5_sub_btn = gr.Button("A - B")
            t5_sub_btn.click(
                fn=tab5_subtract,
                inputs=[t5_sub_a, t5_sub_b, t5_normalize],
                outputs=[t5_status, t5_result_state],
            )

        with gr.Tab("スケーリング (A * α)"):
            t5_scale_a = gr.Dropdown(label="声A", choices=lib_choices, interactive=True)
            t5_scale_alpha = gr.Slider(minimum=-3.0, maximum=3.0, value=1.0, step=0.1, label="α (係数)")
            t5_scale_btn = gr.Button("A * α")
            t5_scale_btn.click(
                fn=tab5_scale,
                inputs=[t5_scale_a, t5_scale_alpha, t5_normalize],
                outputs=[t5_status, t5_result_state],
            )

        with gr.Tab("カスタム (A + α*(B - C))"):
            t5_cust_a = gr.Dropdown(label="声A", choices=lib_choices, interactive=True)
            t5_cust_b = gr.Dropdown(label="声B", choices=lib_choices, interactive=True)
            t5_cust_c = gr.Dropdown(label="声C", choices=lib_choices, interactive=True)
            t5_cust_alpha = gr.Slider(minimum=-3.0, maximum=3.0, value=1.0, step=0.1, label="α (係数)")
            t5_cust_btn = gr.Button("A + α*(B - C)")
            t5_cust_btn.click(
                fn=tab5_custom,
                inputs=[t5_cust_a, t5_cust_b, t5_cust_c, t5_cust_alpha, t5_normalize],
                outputs=[t5_status, t5_result_state],
            )

        gr.Markdown("---")

        # Save / Preview
        with gr.Row():
            with gr.Column():
                t5_save_name = gr.Textbox(label="保存名", placeholder="例: A+Bブレンド")
                t5_save_tags = gr.Textbox(label="タグ（カンマ区切り）")
                t5_save_btn = gr.Button("ライブラリに保存")
                t5_save_status = gr.Textbox(label="保存結果", interactive=False)
            with gr.Column():
                t5_tts_text = gr.Textbox(label="試聴テキスト", lines=2, placeholder="テキストを入力...")
                t5_tts_lang = gr.Dropdown(choices=LANGUAGES, value="Auto", label="言語")
                t5_tts_btn = gr.Button("TTS試聴", variant="primary")
                t5_tts_audio = gr.Audio(label="生成音声", type="numpy")
                t5_tts_status = gr.Textbox(label="TTS結果", interactive=False)

        t5_save_btn.click(
            fn=tab5_save_result,
            inputs=[t5_save_name, t5_save_tags, t5_result_state],
            outputs=[t5_save_status],
        )
        t5_tts_btn.click(
            fn=tab5_tts_preview,
            inputs=[t5_tts_text, t5_tts_lang, t5_result_state],
            outputs=[t5_tts_audio, t5_tts_status],
        )

        # Dropdowns dict for refresh
        dropdowns = {
            "t5_add_a": t5_add_a,
            "t5_add_b": t5_add_b,
            "t5_sub_a": t5_sub_a,
            "t5_sub_b": t5_sub_b,
            "t5_scale_a": t5_scale_a,
            "t5_cust_a": t5_cust_a,
            "t5_cust_b": t5_cust_b,
            "t5_cust_c": t5_cust_c,
            "t5_avg_voices": t5_avg_voices,
        }
        return dropdowns


def _build_tab6(embedding_state: gr.State):
    """Tab 6: Direction Vectors. Returns dropdown components."""
    with gr.Tab("6. 方向ベクトル"):
        gr.Markdown(
            "グループAとBの平均の差から方向ベクトルを計算し、任意の声に適用します。\n\n"
            "**例**: 男性グループ(A) → 女性グループ(B) で「女性化」方向ベクトルを作成"
        )

        t6_direction_state = gr.State(value=None)
        t6_result_state = gr.State(value=None)

        lib_choices = get_library_choices()
        dir_choices = get_direction_vector_choices()

        # --- Direction computation ---
        with gr.Accordion("方向ベクトルの計算", open=True):
            with gr.Row():
                t6_group_a = gr.Dropdown(
                    label="グループA（開始点、複数選択）", choices=lib_choices,
                    multiselect=True, interactive=True,
                )
                t6_group_b = gr.Dropdown(
                    label="グループB（目標点、複数選択）", choices=lib_choices,
                    multiselect=True, interactive=True,
                )
            t6_dir_name = gr.Textbox(label="方向ベクトル名", placeholder="例: male→female")
            t6_save_dir = gr.Checkbox(label="ライブラリに保存", value=True)
            t6_compute_btn = gr.Button("方向ベクトルを計算", variant="primary")
            t6_compute_status = gr.Textbox(label="計算結果", lines=5, interactive=False)

        # --- Application ---
        with gr.Accordion("方向ベクトルの適用", open=True):
            with gr.Row():
                t6_base_voice = gr.Dropdown(
                    label="ベース声", choices=lib_choices, interactive=True,
                )
                t6_saved_dir = gr.Dropdown(
                    label="保存済み方向ベクトル（任意）", choices=dir_choices, interactive=True,
                )
            t6_intensity = gr.Slider(
                minimum=-2.0, maximum=2.0, value=1.0, step=0.1,
                label="強度（-2.0〜+2.0）",
            )
            t6_normalize = gr.Checkbox(label="結果を正規化", value=False)
            t6_apply_btn = gr.Button("方向ベクトルを適用")
            t6_apply_status = gr.Textbox(label="適用結果", lines=4, interactive=False)

        gr.Markdown("---")

        # Save / Preview
        with gr.Row():
            with gr.Column():
                t6_save_name = gr.Textbox(label="保存名", placeholder="例: 女性化声A")
                t6_save_tags = gr.Textbox(label="タグ（カンマ区切り）")
                t6_save_btn = gr.Button("ライブラリに保存")
                t6_save_status = gr.Textbox(label="保存結果", interactive=False)
            with gr.Column():
                t6_tts_text = gr.Textbox(label="試聴テキスト", lines=2, placeholder="テキストを入力...")
                t6_tts_lang = gr.Dropdown(choices=LANGUAGES, value="Auto", label="言語")
                t6_tts_btn = gr.Button("TTS試聴", variant="primary")
                t6_tts_audio = gr.Audio(label="生成音声", type="numpy")
                t6_tts_status = gr.Textbox(label="TTS結果", interactive=False)

        t6_compute_btn.click(
            fn=tab6_compute_direction,
            inputs=[t6_group_a, t6_group_b, t6_dir_name, t6_save_dir],
            outputs=[t6_compute_status, t6_direction_state],
        )
        t6_apply_btn.click(
            fn=tab6_apply_direction,
            inputs=[t6_base_voice, t6_direction_state, t6_saved_dir, t6_intensity, t6_normalize],
            outputs=[t6_apply_status, t6_result_state],
        )
        t6_save_btn.click(
            fn=tab6_save_result,
            inputs=[t6_save_name, t6_save_tags, t6_result_state],
            outputs=[t6_save_status],
        )
        t6_tts_btn.click(
            fn=tab6_tts_preview,
            inputs=[t6_tts_text, t6_tts_lang, t6_result_state],
            outputs=[t6_tts_audio, t6_tts_status],
        )

        dropdowns = {
            "t6_base": t6_base_voice,
            "t6_group_a": t6_group_a,
            "t6_group_b": t6_group_b,
            "t6_saved_dir": t6_saved_dir,
        }
        return dropdowns


def _build_tab7():
    """Tab 7: Embedding Space Visualization."""
    with gr.Tab("7. 埋め込み空間の可視化"):
        gr.Markdown(
            "ライブラリに登録されたEmbeddingをt-SNEまたはPCAで2Dに投影し、散布図で表示します。"
        )

        with gr.Row():
            t7_method = gr.Dropdown(
                choices=["t-SNE", "PCA"], value="t-SNE", label="次元削減手法"
            )
            t7_perplexity = gr.Slider(
                minimum=2, maximum=50, value=10, step=1,
                label="Perplexity (t-SNEのみ)",
            )
            t7_seed = gr.Number(value=42, label="Random Seed", precision=0)

        t7_btn = gr.Button("可視化を更新", variant="primary")
        t7_info = gr.Textbox(label="情報", interactive=False)
        t7_plot = gr.ScatterPlot(
            x="x", y="y", color="ソース", tooltip=["名前", "タグ", "ソース"],
            label="Embedding空間",
            height=500,
        )

        t7_btn.click(
            fn=tab7_visualize,
            inputs=[t7_method, t7_perplexity, t7_seed],
            outputs=[t7_plot, t7_info],
        )


# ---------------------------------------------------------------------------
# Main UI assembly
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen3 TTS ボイスクローニング") as demo:
        gr.Markdown(
            "# Qwen3 TTS ボイスクローニング\n"
            "話者Embeddingの抽出、声のクローン、補間、ライブラリ管理、"
            "ベクトル演算、方向変換、可視化を行います。"
        )

        embedding_state = gr.State(value=None)

        # Build each tab
        _build_tab1(embedding_state)
        _build_tab2(embedding_state)
        _build_tab3()
        t4_status, t4_table, t4_dd = _build_tab4(embedding_state)
        t5_dd = _build_tab5(embedding_state)
        t6_dd = _build_tab6(embedding_state)
        _build_tab7()

        # Collect all library-dependent dropdown components for cross-tab refresh
        all_refresh_outputs = [
            t4_dd["t4_del"],            # tab4 delete dropdown
            t4_dd["t4_use"],            # tab4 use_in_tts dropdown
            t5_dd["t5_add_a"],          # tab5 add A
            t5_dd["t5_add_b"],          # tab5 add B
            t5_dd["t5_sub_a"],          # tab5 sub A
            t5_dd["t5_sub_b"],          # tab5 sub B
            t5_dd["t5_scale_a"],        # tab5 scale A
            t5_dd["t5_cust_a"],         # tab5 custom A
            t5_dd["t5_cust_b"],         # tab5 custom B
            t5_dd["t5_cust_c"],         # tab5 custom C
            t5_dd["t5_avg_voices"],     # tab5 avg voices
            t6_dd["t6_base"],           # tab6 base voice
            t6_dd["t6_group_a"],        # tab6 group A
            t6_dd["t6_group_b"],        # tab6 group B
            t6_dd["t6_saved_dir"],      # tab6 saved direction
        ]

        # Wire tab4 registration/delete/import buttons with cross-tab refresh
        # Registration from audio
        t4_dd["_reg_audio_btn"].click(
            fn=t4_dd["_reg_audio_fn"],
            inputs=t4_dd["_reg_audio_inputs"],
            outputs=[t4_status, t4_table] + all_refresh_outputs,
        )
        # Registration from JSON
        t4_dd["_reg_json_btn"].click(
            fn=t4_dd["_reg_json_fn"],
            inputs=t4_dd["_reg_json_inputs"],
            outputs=[t4_status, t4_table] + all_refresh_outputs,
        )
        # Registration from state
        t4_dd["_reg_state_btn"].click(
            fn=t4_dd["_reg_state_fn"],
            inputs=t4_dd["_reg_state_inputs"],
            outputs=[t4_status, t4_table] + all_refresh_outputs,
        )
        # Delete
        t4_dd["_del_btn"].click(
            fn=t4_dd["_del_fn"],
            inputs=[t4_dd["_del_input"]],
            outputs=[t4_status, t4_table] + all_refresh_outputs,
        )
        # Import
        t4_dd["_import_btn"].click(
            fn=t4_dd["_import_fn"],
            inputs=[t4_dd["_import_input"]],
            outputs=[t4_status, t4_table] + all_refresh_outputs,
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
        default="marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B",
        help="HF model ID for the ECAPA-TDNN speaker encoder "
             "(default: marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B)",
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
        "--library-path",
        default=os.path.expanduser("~/.qwen3_tts_voice_library.json"),
        help="Path to the voice library JSON file "
             "(default: ~/.qwen3_tts_voice_library.json)",
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
    global _emb_model_id, _tts_model_path, _device, _voice_library

    args = parse_args()
    _emb_model_id = args.encoder_model
    _tts_model_path = args.tts_model
    _device = args.device
    _voice_library = VoiceLibrary(args.library_path)
    print(f"[init] Voice library: {args.library_path} ({len(_voice_library)} entries)")

    # Eagerly load the speaker encoder (lightweight, ~12M params on CPU)
    get_embedding_encoder()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
