# Voice Embedding テクニックガイド

Qwen3-TTS の speaker embedding を活用した声の操作テクニック集。
音声を2048次元のベクトルに変換し、数学的な操作で声のクローニング・変換・検索・感情制御を行う。

> **参考**: [Reddit — Qwen3 TTS Voice Embedding](https://www.reddit.com/r/LocalLLaMA/) by marksverdhei

---

## 目次

### 基礎編
1. [Voice Cloning（声のクローニング）](#1-voice-cloning声のクローニング)
2. [ベクトル演算（Math to Modify Voices）](#2-ベクトル演算math-to-modify-voices)
3. [声の平均化（Average Voices）](#3-声の平均化average-voices)
4. [性別変換（Gender Swap）](#4-性別変換gender-swap)
5. [ピッチ変更（Pitch Modification）](#5-ピッチ変更pitch-modification)

### 応用編
6. [声のブレンド（SLERP 補間）](#6-声のブレンドmix-and-match--slerp-補間)
7. [感情空間の構築（Emotion Space）](#7-感情空間の構築emotion-space)
8. [セマンティック声検索（Semantic Voice Search）](#8-セマンティック声検索semantic-voice-search)
9. [スタンドアロン Embedding モデル](#9-スタンドアロン-embedding-モデル)
10. [ONNX モデル（Web/フロントエンド推論）](#10-onnx-モデルwebフロントエンド推論)

---

## 前提条件

- **モデル**: `Qwen3-TTS-*-Base` モデルが必要（CustomVoice / VoiceDesign では speaker embedding は使用不可）
- **Python**: 3.10+
- **依存パッケージ**: `transformers>=4.57.0,<5.0.0`, `torch`, `librosa`, `numpy`, `soundfile`
- **Embedding モデル**: `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B`（スタンドアロン抽出用）

## セットアップ

```bash
# Gradio UI 起動
uv run python examples/online_serving/qwen3_tts/gradio_voice_cloning.py \
    --encoder-model marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B \
    --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --device cuda:0
```

## クイックリファレンス

| # | テクニック | Gradio | CLI | Python API |
|---|----------|--------|-----|------------|
| 1 | Voice Cloning | Tab 1→2 | `extract` | AutoModel |
| 2 | ベクトル演算 | Tab 5 | - | numpy |
| 3 | 声の平均化 | Tab 5 | - | numpy |
| 4 | 性別変換 | Tab 6 | - | Direction Vector |
| 5 | ピッチ変更 | Tab 6 | - | Direction Vector |
| 6 | 声のブレンド | Tab 3 | `interpolate` / `pipeline` | slerp() |
| 7 | 感情空間 | - | `emotion_analysis.py` | 4サブコマンド |
| 8 | セマンティック検索 | Tab 4 | - | find_similar() |
| 9 | スタンドアロンモデル | - | - | AutoModel |
| 10 | ONNX | - | - | onnxruntime |

---

## 1. Voice Cloning（声のクローニング）

### 概念

Voice Cloning は参照音声から**話者の声の特徴**を2048次元のベクトル（speaker embedding）として抽出し、そのベクトルを使って任意のテキストをその声で読み上げる技術である。

処理の流れ:

```
参照音声 (.wav) → ECAPA-TDNN エンコーダ → 2048次元ベクトル → TTS生成
```

ECAPA-TDNN は約12Mパラメータの軽量な話者エンコーダで、音声のメルスペクトログラム（128bin, 24kHz, n_fft=1024, hop=256）から話者特徴量を抽出する。

### クローニングモード

| モード | 説明 | 話者類似度 | 用途 |
|--------|------|-----------|------|
| **x_vector モード** | embedding のみ使用 | ~0.75 | embedding を事前保存して再利用する場合 |
| **ICL モード** | embedding + 参照音声 + 参照テキスト | ~0.89 | 最高品質のクローンが必要な場合 |

ICL（In-Context Learning）モードでは参照音声の音声トークンも入力に含めるため、より高い話者類似度が得られる。

### Gradio UI での操作

1. **Tab 1（Embedding 抽出）**: 音声をアップロードして「Embedding抽出」をクリック。結果はセッション state に保持され、Tab 2 から参照できる。JSON ダウンロードも可能。
2. **Tab 2（ボイスクローンTTS）**: embedding の読み込み優先順位は、アップロード JSON > Tab 1 の state > 参照音声。参照テキストを入力すると ICL モードになる。

### CLI での操作

```bash
# embedding の抽出
python examples/online_serving/qwen3_tts/speaker_embedding_interpolation.py extract \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --audio voice_sample.wav \
    --output voice_embedding.json \
    --device cpu

# 抽出した embedding で TTS 生成（vLLM サーバー経由）
python examples/online_serving/qwen3_tts/speaker_embedding_interpolation.py interpolate \
    --embedding-a voice_embedding.json \
    --embedding-b voice_embedding.json \
    --ratio 0.0 \
    --text "こんにちは、テスト音声です。" \
    --output output.wav \
    --api-base http://localhost:8000
```

### Python コード例

```python
import torch
import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
import librosa

# --- Embedding 抽出 ---
encoder_model_id = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
encoder = AutoModel.from_pretrained(encoder_model_id, trust_remote_code=True).eval()
fe = AutoFeatureExtractor.from_pretrained(encoder_model_id, trust_remote_code=True)

audio, sr = librosa.load("voice_sample.wav", sr=fe.sampling_rate, mono=True)
inputs = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt")

with torch.inference_mode():
    output = encoder(**inputs)
    embedding = output.last_hidden_state[0].float().cpu().numpy()

print(f"Embedding shape: {embedding.shape}")  # (2048,)

# --- JSON として保存 ---
import json
with open("voice_embedding.json", "w") as f:
    json.dump(embedding.tolist(), f)
```

### vLLM API 経由での利用

```python
import httpx

payload = {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "こんにちは、これはクローン音声です。",
    "task_type": "Base",
    "speaker_embedding": embedding.tolist(),  # 2048次元のリスト
    "response_format": "wav",
}

resp = httpx.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    headers={"Authorization": "Bearer EMPTY"},
    timeout=300.0,
)

with open("cloned_voice.wav", "wb") as f:
    f.write(resp.content)
```

### 注意事項

- **Base モデル専用**: `speaker_embedding` パラメータは `task_type: "Base"` のみ対応。
- **サンプルレート**: 入力音声は内部で24kHzにリサンプリングされる。
- **参照音声の長さ**: 5-15秒を推奨（最低3秒）。
- **モデル間の互換性**: 0.6B モデル（1024次元）と 1.7B モデル（2048次元）の embedding は互換性がない。
- **speaker_embedding の次元数制限**: API では 64〜8192次元を受け付ける。

---

## 2. ベクトル演算（Math to Modify Voices）

### 概念

Speaker embedding は高次元ベクトル空間上の点であるため、通常のベクトル演算を適用して声の特徴を操作できる。word2vec の `king - man + woman = queen` と同様の発想。

### 演算の種類

| 演算 | 式 | 用途 |
|------|-----|------|
| **加算** | `A + B` | 2つの声の特徴を合成 |
| **減算** | `A - B` | A から B の特徴を除去 |
| **スケーリング** | `A * α` | 声の特徴を増幅/減衰 |
| **カスタム** | `A + α * (B - C)` | B と C の差分（方向）を A に適用 |

### Gradio Tab 5 での操作

1. サブタブ（平均/加算/減算/スケーリング/カスタム）を選択
2. ドロップダウンからライブラリの声を選択
3. 「結果を正規化（L2ノルム=1）」チェックボックスで正規化を制御
4. 演算を実行 → 「TTS試聴」で確認

### Python コード例

```python
import numpy as np

emb_a = load_emb("voice_a.json")
emb_b = load_emb("voice_b.json")
emb_c = load_emb("voice_c.json")

result_add = emb_a + emb_b                       # 加算
result_sub = emb_a - emb_b                       # 減算
result_scale = emb_a * 1.5                       # スケーリング
result_custom = emb_a + 0.8 * (emb_b - emb_c)   # カスタム

# L2 正規化（任意）
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-9)
```

### 注意事項

- 演算結果のノルムが元の embedding と大きく異なると音声品質が低下する場合がある。
- 極端なスケーリング係数は不自然な音声になりやすい。

---

## 3. 声の平均化（Average Voices）

### 概念

複数の speaker embedding の平均を取ることで、各話者の個性を平滑化した「中間的な声」を作成できる。

```python
average = np.mean([emb_1, emb_2, ..., emb_N], axis=0)
```

### Gradio Tab 5 での操作

「5. ベクトル演算」タブ → 「平均 (N声)」サブタブで2つ以上の声を選択して計算。

### 活用例

- **男性/女性の平均声**: 性別変換の基準点として使用
- **ニュートラル声**: 多様な話者の平均 → 偏りのないベースライン
- 3人以上の平均が推奨。2人の場合は SLERP 補間（`ratio=0.5`）の方が品質が良いことがある。

---

## 4. 性別変換（Gender Swap）

### 概念

**Direction Vector**（方向ベクトル）を使って声の性別特性を変換する:

```
female_direction = mean(女性群) - mean(男性群)
変換後 = base_embedding + intensity * female_direction
```

### Gradio Tab 6 での操作

1. 「グループA（開始点）」に男性の声を複数選択
2. 「グループB（目標点）」に女性の声を複数選択
3. 「方向ベクトルを計算」をクリック
4. 「ベース声」を選択 → 「強度」スライダーで調整 → 「方向ベクトルを適用」

### intensity パラメータの推奨範囲

| 値 | 効果 |
|----|------|
| **0.0** | 変換なし |
| **0.3〜0.7** | 控えめな変換。自然さを保ちやすい |
| **1.0** | 標準的な変換 |
| **1.5〜2.0** | 強い変換。不自然になる可能性あり |
| **負の値** | 逆方向の変換 |

### 注意事項

- グループごとに最低3人以上の話者推奨。
- **高 intensity（1.0以上）では EOS トークンが出ない場合がある** → `max_new_tokens` で制限必須。

---

## 5. ピッチ変更（Pitch Modification）

### 概念

Direction Vector を使った間接的なアプローチ。embedding 空間ではピッチが他の声質特徴と絡み合っているため、直接的なピッチ制御はできない。

```
pitch_up_direction = mean(高ピッチ群) - mean(低ピッチ群)
result = base + intensity * pitch_up_direction
```

### 音声処理的ピッチシフトとの違い

| 特性 | Direction Vector | 音声処理（リサンプリング等） |
|------|-----------------|---------------------------|
| **自然さ** | 高い（モデルが自然な音声を生成） | 低い（アーティファクトが出やすい） |
| **制御精度** | 低い（ピッチ以外も変わる） | 高い（Hz 単位で指定可能） |
| **声質への影響** | ピッチと連動して声質も変化 | 基本的に声質は保持 |

精密なピッチ制御が必要な場合は、TTS 生成後に `librosa.effects.pitch_shift()` を使う方が適切。

---

## 6. 声のブレンド（Mix and Match / SLERP 補間）

### 概念

2つの speaker embedding を **球面線形補間（SLERP）** でブレンドし、声A と声B の中間的な声質を生成する。

### SLERP vs LERP

| 手法 | 数式 | 特徴 |
|------|------|------|
| LERP | `(1-t) * v0 + t * v1` | 直線的。ノルムが中間で小さくなる |
| SLERP | `sin((1-t)*Ω)/sin(Ω) * v0 + sin(t*Ω)/sin(Ω) * v1` | 球面上を等速移動。ノルム保存 |

### ratio パラメータ

| ratio | 結果 |
|-------|------|
| `0.0` | 完全に声A |
| `0.5` | 声A と声B の中間 |
| `1.0` | 完全に声B |

### Gradio Tab 3 での操作

声A と声B の音声をアップロード → SLERP 比率スライダーで調整 → 合成テキストを入力して生成。

### CLI

```bash
# 2つの embedding をブレンドして音声生成
python speaker_embedding_interpolation.py interpolate \
    --embedding-a voice_a.json --embedding-b voice_b.json \
    --ratio 0.5 --text "ブレンドされた声です。" --output blended.wav

# 一括処理: 音声ファイルから抽出 → 複数比率で補間
python speaker_embedding_interpolation.py pipeline \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --audio-a voice_a.wav --audio-b voice_b.wav \
    --ratios 0.0 0.25 0.5 0.75 1.0 \
    --text "こんにちは" --output-dir ./interpolated/
```

### Python コード例

```python
def slerp(v0, v1, t):
    v0_n = v0 / (np.linalg.norm(v0) + 1e-8)
    v1_n = v1 / (np.linalg.norm(v1) + 1e-8)
    dot = np.clip(np.dot(v0_n, v1_n), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-6:
        return (1.0 - t) * v0 + t * v1  # LERP fallback
    so = np.sin(omega)
    return (np.sin((1-t)*omega)/so) * v0 + (np.sin(t*omega)/so) * v1
```

---

## 7. 感情空間の構築（Emotion Space）

### 概念

感情ラベル付き音声データセットから embedding を抽出し、感情クラスタの差分ベクトル（Direction Vector）を計算。任意の声に感情を付与する。

### 4ステップのワークフロー

#### Step 1: extract（バッチ embedding 抽出）

```bash
python emotion_analysis.py extract \
    --data-dir ./emotion_dataset/ --output embeddings.npz \
    --model-id marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B --device cuda
```

フォルダ構成:
```
emotion_dataset/
  angry/emotion/    ← 怒り音声 .wav
  normal/emotion/   ← 通常音声 .wav
  sad/emotion/      ← 悲しみ音声 .wav
  smile/emotion/    ← 笑顔音声 .wav
```

#### Step 2: visualize（t-SNE / PCA 可視化）

```bash
python emotion_analysis.py visualize --input embeddings.npz
```

感情ごとに色分けされた散布図（angry=赤, normal=灰, sad=青, smile=オレンジ）を生成。

#### Step 3: direction（Direction Vector 計算）

```bash
python emotion_analysis.py direction --input embeddings.npz --output emotion_directions.json
```

計算式: `direction = mean(emotion) - mean(normal)`

#### Step 4: apply（TTS に適用）

```bash
python emotion_analysis.py apply \
    --directions emotion_directions.json \
    --text "こんにちは、今日はいい天気ですね。" \
    --alphas 0.0 0.5 1.0 1.5 --output-dir output/ --device cuda:0
```

計算式: `modified = normal_mean + alpha * direction`

### alpha 値の推奨範囲

| alpha | 効果 |
|-------|------|
| `0.0` | normal そのまま |
| `0.3-0.5` | 微妙な感情のニュアンス |
| `0.8-1.0` | はっきりした感情表現 |
| `1.5+` | 強調された感情（不自然になる可能性、EOS問題注意） |

### 感情 Direction Vector の利用

`emotion_directions.json` を生成後、`apply` コマンドで任意のテキストに感情を適用して音声生成できる。

---

## 8. セマンティック声検索（Semantic Voice Search）

### 概念

**コサイン類似度**で embedding 空間内の最近傍を検索し、最も類似する声をライブラリから発見する。

```
similarity = dot(query_norm, entry_norm)
```

### Gradio Tab 4 での操作

1. クエリ音声をアップロード
2. Top-K スライダーで返す結果数を指定（1-20）
3. 「類似検索」をクリック → 類似度スコア付きのランキング表示

### Python コード例

```python
def find_similar(query, library_embeddings, top_k=5):
    q_norm = query / (np.linalg.norm(query) + 1e-9)
    results = []
    for name, emb in library_embeddings.items():
        e_norm = emb / (np.linalg.norm(emb) + 1e-9)
        sim = float(np.dot(q_norm, e_norm))
        results.append((name, sim))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
```

### 活用例

- 目標の声に近い声をライブラリから探す
- 生成した embedding の品質チェック
- 補間に適した声のペア選択

---

## 9. スタンドアロン Embedding モデル

### 概念

フルTTSモデル（数GB）をロードせず、軽量な ECAPA-TDNN encoder（数十MB）だけで embedding を抽出する。

### 利用可能なモデル

| モデル ID | 出力次元 | パラメータ数 | 元モデル |
|-----------|---------|------------|---------|
| `marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B` | 2048 | ~12.2M | 1.7B-Base |
| `marksverdhei/Qwen3-Voice-Embedding-25Hz-0.6B` | 1024 | ~6.3M | 0.6B-Base |

### Python コード例

```python
from transformers import AutoModel, AutoFeatureExtractor
import librosa, torch

model_id = "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B"
encoder = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
fe = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)

audio, sr = librosa.load("reference.wav", sr=fe.sampling_rate, mono=True)
inputs = fe(audio, sampling_rate=fe.sampling_rate, return_tensors="pt")

with torch.inference_mode():
    embedding = encoder(**inputs).last_hidden_state[0].float().cpu().numpy()
# → (2048,)
```

### フル TTS モデルとの比較

| 比較項目 | スタンドアロン | フル TTS |
|---------|--------------|---------|
| モデルサイズ | ~50MB | ~7GB |
| ロード時間 | 数秒 | 数十秒〜数分 |
| GPU 必要性 | CPU で十分 | GPU 推奨 |
| 機能 | 抽出のみ | 抽出 + TTS 生成 |

### 注意

- **0.6B と 1.7B の embedding は互換性なし**（次元数が異なる）
- `*-Base` 以外のモデルは speaker encoder を含まない

---

## 10. ONNX モデル（Web/フロントエンド推論）

### 概念

ONNX 形式にエクスポートされた speaker encoder をブラウザや Edge デバイスで実行。サーバーに音声を送らずクライアント側で embedding を計算可能。

### Python（onnxruntime）

```python
import onnxruntime as ort
import librosa, numpy as np

session = ort.InferenceSession("qwen3_voice_embedding_12hz_1.7b.onnx")
audio, sr = librosa.load("reference.wav", sr=24000, mono=True)

input_name = session.get_inputs()[0].name
result = session.run(None, {input_name: audio.reshape(1, -1).astype(np.float32)})
embedding = result[0][0]  # (2048,)
```

### JavaScript（onnxruntime-web）

```javascript
import * as ort from 'onnxruntime-web';

const session = await ort.InferenceSession.create('qwen3_voice_embedding.onnx');
const audioContext = new AudioContext({ sampleRate: 24000 });
const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
const audioData = audioBuffer.getChannelData(0);

const inputTensor = new ort.Tensor('float32', audioData, [1, audioData.length]);
const results = await session.run({ input: inputTensor });
const embedding = results.output.data;  // Float32Array(2048)

// embedding をサーバーの TTS API に送信
await fetch('/api/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        text: 'こんにちは',
        speaker_embedding: Array.from(embedding),
        language: 'Japanese',
    }),
});
```

> **注意:** 上記は概念的なコード例です。入力テンソル名やフォーマットは ONNX モデルの仕様を確認してください。

### 現在のプロジェクトでの対応状況

ht-vllm-omni プロジェクトでは ONNX モデルを直接サポートする機能は未実装。ただし vLLM の `/v1/audio/speech` エンドポイントは `speaker_embedding` パラメータを受け付けるため、クライアント側で ONNX 推論した embedding をそのまま送信できる:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer EMPTY" \
    -d '{"model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "input": "こんにちは", "speaker_embedding": [0.123, -0.456, ...], "task_type": "Base", "response_format": "wav"}' \
    --output output.wav
```

### 将来の可能性

- ブラウザ内 embedding 計算（音声アップロード不要）
- Edge デバイス対応（スマホ、ラズベリーパイ）
- プライバシー保護（embedding のみ送信）
- リアルタイム声変換（低レイテンシ embedding + WebSocket TTS）
