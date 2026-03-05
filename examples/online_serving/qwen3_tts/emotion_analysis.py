"""Emotion embedding analysis and direction vector construction for Qwen3-TTS.

Extracts speaker embeddings from emotion-labeled audio datasets, visualizes
clusters via t-SNE/PCA, builds emotion direction vectors, and applies them
to TTS synthesis.

Subcommands:
    extract    — Batch-extract embeddings from emotion-labelled audio folders
    visualize  — t-SNE / PCA scatter plots with per-emotion coloring
    direction  — Compute emotion direction vectors (angry/sad/smile − normal)
    apply      — Apply direction vectors to normal embedding and synthesize speech

Examples:
    python emotion_analysis.py extract \
        --data-dir "C:/Users/yuta/Desktop/はるなさん音声/Raw" \
        --output embeddings.npz

    python emotion_analysis.py visualize --input embeddings.npz

    python emotion_analysis.py direction \
        --input embeddings.npz --output emotion_directions.json

    python emotion_analysis.py apply \
        --directions emotion_directions.json \
        --text "こんにちは、今日はいい天気ですね。" \
        --output-dir output/
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "EMPTY"

EMOTIONS = ["angry", "normal", "sad", "smile"]
EMOTION_COLORS = {
    "angry": "#e74c3c",
    "normal": "#95a5a6",
    "sad": "#3498db",
    "smile": "#f39c12",
}

DEFAULT_MODEL_ID = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"

# ──────────────────────────────────────────────
# Embedding extraction (AutoModel + AutoFeatureExtractor)
# ──────────────────────────────────────────────

_encoder = None
_feature_extractor = None


def load_embedding_model(model_id: str, device: str = "cpu"):
    """Load the embedding encoder and feature extractor."""
    global _encoder, _feature_extractor
    if _encoder is not None:
        return _encoder, _feature_extractor

    from transformers import AutoFeatureExtractor, AutoModel

    print(f"Loading embedding model: {model_id} ...")
    _encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    _feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
    print("Embedding model ready.")
    return _encoder, _feature_extractor


@torch.inference_mode()
def extract_single_embedding(
    audio_path: str,
    encoder: torch.nn.Module,
    feature_extractor,
    device: str = "cpu",
) -> np.ndarray:
    """Extract a speaker embedding from a single audio file."""
    import librosa

    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate, mono=True)
    inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = encoder(**inputs)
    return output.last_hidden_state[0].float().cpu().numpy()


# ──────────────────────────────────────────────
# Subcommand: extract
# ──────────────────────────────────────────────


def cmd_extract(args):
    """Batch-extract embeddings from emotion-labelled audio folders."""
    from tqdm import tqdm

    encoder, fe = load_embedding_model(args.model_id, device=args.device)

    all_embeddings = []
    all_labels = []
    all_files = []

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(args.data_dir, emotion, "emotion")
        if not os.path.isdir(emotion_dir):
            print(f"Warning: directory not found, skipping: {emotion_dir}")
            continue

        wav_files = sorted(f for f in os.listdir(emotion_dir) if f.lower().endswith(".wav"))
        print(f"{emotion}: {len(wav_files)} files in {emotion_dir}")

        for fname in tqdm(wav_files, desc=emotion):
            fpath = os.path.join(emotion_dir, fname)
            emb = extract_single_embedding(fpath, encoder, fe, device=args.device)
            all_embeddings.append(emb)
            all_labels.append(emotion)
            all_files.append(os.path.join(emotion, "emotion", fname))

    embeddings = np.stack(all_embeddings)
    labels = np.array(all_labels)
    files = np.array(all_files)

    np.savez(args.output, embeddings=embeddings, labels=labels, files=files)
    print(f"\nSaved {len(embeddings)} embeddings ({embeddings.shape[1]}D) to {args.output}")


# ──────────────────────────────────────────────
# Subcommand: visualize
# ──────────────────────────────────────────────


def cmd_visualize(args):
    """Create t-SNE and PCA scatter plots of emotion embeddings."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    data = np.load(args.input, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    print(f"Loaded {len(embeddings)} embeddings ({embeddings.shape[1]}D)")

    # --- t-SNE ---
    print("Computing t-SNE ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords_tsne = tsne.fit_transform(embeddings)
    _plot_scatter(coords_tsne, labels, "t-SNE", args.output_tsne)

    # --- PCA ---
    print("Computing PCA ...")
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    _plot_scatter(
        coords_pca,
        labels,
        f"PCA (var: {var[0]:.1%}, {var[1]:.1%})",
        args.output_pca,
    )

    print(f"Saved: {args.output_tsne}, {args.output_pca}")


def _plot_scatter(coords: np.ndarray, labels: np.ndarray, title: str, output_path: str):
    """Plot a 2D scatter with per-emotion coloring and centroids."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    for emotion in EMOTIONS:
        mask = labels == emotion
        if not mask.any():
            continue
        pts = coords[mask]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion,
            alpha=0.6,
            s=30,
        )
        centroid = pts.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            c=EMOTION_COLORS[emotion],
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            f"  {emotion}",
            centroid,
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────
# Subcommand: direction
# ──────────────────────────────────────────────


def cmd_direction(args):
    """Compute emotion direction vectors relative to normal."""
    data = np.load(args.input, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    means = {}
    for emotion in EMOTIONS:
        mask = labels == emotion
        if not mask.any():
            print(f"Warning: no samples for {emotion}")
            continue
        means[emotion] = embeddings[mask].mean(axis=0)
        print(f"{emotion}: {mask.sum()} samples, mean norm = {np.linalg.norm(means[emotion]):.4f}")

    if "normal" not in means:
        print("Error: 'normal' emotion is required as baseline.")
        sys.exit(1)

    # Direction vectors
    directions = {}
    for emotion in ["angry", "sad", "smile"]:
        if emotion not in means:
            continue
        directions[emotion] = means[emotion] - means["normal"]
        norm = np.linalg.norm(directions[emotion])
        print(f"{emotion}_direction norm = {norm:.4f}")

    # Cosine similarity matrix
    emotion_keys = [e for e in EMOTIONS if e in means]
    cos_matrix = {}
    for e1 in emotion_keys:
        cos_matrix[e1] = {}
        for e2 in emotion_keys:
            cos_sim = float(
                np.dot(means[e1], means[e2])
                / (np.linalg.norm(means[e1]) * np.linalg.norm(means[e2]) + 1e-8)
            )
            cos_matrix[e1][e2] = round(cos_sim, 6)

    result = {
        "emotion_means": {k: v.tolist() for k, v in means.items()},
        "directions": {k: v.tolist() for k, v in directions.items()},
        "cosine_similarity_matrix": cos_matrix,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved direction vectors to {args.output}")

    print("\nCosine similarity matrix:")
    header = "".join(f"{e:>10}" for e in emotion_keys)
    print(f"{'':>10}{header}")
    for e1 in emotion_keys:
        row = "".join(f"{cos_matrix[e1][e2]:>10.4f}" for e2 in emotion_keys)
        print(f"{e1:>10}{row}")


# ──────────────────────────────────────────────
# Subcommand: apply (direct TTS model loading)
# ──────────────────────────────────────────────

_tts_model = None


def _load_tts_model(model_path: str, device: str = "cuda:0"):
    """Load the Qwen3-TTS model directly for synthesis."""
    global _tts_model
    if _tts_model is not None:
        return _tts_model

    import importlib
    import types

    # Shim: stub out vllm_omni modules so the model code can import
    _REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    _MODELS_DIR = os.path.join(
        _REPO_ROOT, "vllm_omni", "model_executor", "models", "qwen3_tts"
    )

    def _download_weights_stub(
        model_name_or_path, cache_dir=None, allow_patterns=None, revision=None, **kw
    ):
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model_name_or_path, cache_dir=cache_dir,
            allow_patterns=allow_patterns, revision=revision,
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

    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
        Qwen3TTSConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from vllm_omni.model_executor.models.qwen3_tts.processing_qwen3_tts import (
        Qwen3TTSProcessor,
    )
    from transformers import AutoConfig, AutoModel, AutoProcessor

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    print(f"Loading TTS model: {model_path} on {device} ...")
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    gen_defaults = getattr(model, "generate_config", {}) or {}

    _tts_model = (model, processor, gen_defaults)
    print("TTS model ready.")
    return _tts_model


@torch.no_grad()
def _synthesize_direct(
    text: str,
    speaker_embedding: list[float],
    output_path: str,
    model_tuple: tuple,
) -> None:
    """Generate speech directly using the loaded TTS model."""
    import soundfile as sf
    import time

    model, processor, gen_defaults = model_tuple
    device = next(model.parameters()).device

    spk = torch.tensor(speaker_embedding, dtype=torch.float32).to(device)
    prompt_dict = dict(
        ref_code=[None],
        ref_spk_embedding=[spk],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )

    wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    inp = processor(text=wrapped, return_tensors="pt", padding=True)
    input_ids = inp["input_ids"].to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    gen_kwargs = dict(
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
        max_new_tokens=256,
    )
    # Apply model defaults but preserve our max_new_tokens override
    saved_max = gen_kwargs["max_new_tokens"]
    for k, v in gen_defaults.items():
        if k in gen_kwargs:
            gen_kwargs[k] = v
    gen_kwargs["max_new_tokens"] = saved_max

    start = time.time()
    codes_list, _ = model.generate(
        input_ids=[input_ids],
        ref_ids=None,
        voice_clone_prompt=prompt_dict,
        languages=["Auto"],
        **gen_kwargs,
    )
    gen_time = time.time() - start
    print(f"  generation: {gen_time:.1f}s")

    wavs, sr = model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in codes_list]
    )
    wav = wavs[0].astype(np.float32)

    sf.write(output_path, wav, int(sr))
    duration = len(wav) / int(sr)
    print(f"  saved: {output_path} ({duration:.1f}s audio, {os.path.getsize(output_path)} bytes)")


def cmd_apply(args):
    """Apply direction vectors to normal embedding and generate TTS audio."""
    with open(args.directions) as f:
        data = json.load(f)

    normal_mean = np.array(data["emotion_means"]["normal"], dtype=np.float32)
    directions = {k: np.array(v, dtype=np.float32) for k, v in data["directions"].items()}

    os.makedirs(args.output_dir, exist_ok=True)

    # Load TTS model directly
    model_tuple = _load_tts_model(args.tts_model, device=args.device)

    alphas = [float(a) for a in args.alphas]

    for emotion, direction in directions.items():
        for alpha in alphas:
            modified = normal_mean + alpha * direction
            out_path = os.path.join(args.output_dir, f"emotion_{emotion}_alpha_{alpha:.1f}.wav")
            print(f"\n--- {emotion} α={alpha:.1f} ---")
            print(f"  embedding norm: {np.linalg.norm(modified):.4f}")
            _synthesize_direct(
                text=args.text,
                speaker_embedding=modified.tolist(),
                output_path=out_path,
                model_tuple=model_tuple,
            )

    print(f"\nAll outputs saved to {args.output_dir}/")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Emotion embedding analysis and direction vector construction for Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID for embedding extraction (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu/cuda)")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Batch-extract embeddings from emotion folders")
    p_ext.add_argument("--data-dir", required=True, help="Root data dir containing emotion subfolders")
    p_ext.add_argument("--output", "-o", default="embeddings.npz", help="Output .npz file")

    # --- visualize ---
    p_vis = sub.add_parser("visualize", help="t-SNE / PCA visualization")
    p_vis.add_argument("--input", "-i", default="embeddings.npz", help="Input .npz file")
    p_vis.add_argument("--output-tsne", default="emotion_tsne.png", help="t-SNE output image")
    p_vis.add_argument("--output-pca", default="emotion_pca.png", help="PCA output image")

    # --- direction ---
    p_dir = sub.add_parser("direction", help="Compute emotion direction vectors")
    p_dir.add_argument("--input", "-i", default="embeddings.npz", help="Input .npz file")
    p_dir.add_argument("--output", "-o", default="emotion_directions.json", help="Output JSON file")

    # --- apply ---
    p_app = sub.add_parser("apply", help="Apply direction vectors and generate TTS audio")
    p_app.add_argument("--directions", required=True, help="Direction vectors JSON file")
    p_app.add_argument("--text", required=True, help="Text to synthesize")
    p_app.add_argument(
        "--tts-model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="TTS model ID or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p_app.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5],
        help="Alpha multipliers for direction vectors (default: 0.0 0.5 1.0 1.5)",
    )
    p_app.add_argument("--output-dir", default="output", help="Output directory for wav files")

    args = parser.parse_args()

    # Eagerly load the embedding model for commands that need it
    if args.command == "extract":
        load_embedding_model(args.model_id, device=args.device)

    {
        "extract": cmd_extract,
        "visualize": cmd_visualize,
        "direction": cmd_direction,
        "apply": cmd_apply,
    }[args.command](args)


if __name__ == "__main__":
    main()
