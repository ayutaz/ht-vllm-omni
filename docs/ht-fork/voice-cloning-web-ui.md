# Voice Cloning Web UI Specification

HTMX + FastAPI + DaisyUI ベースの Voice Cloning Web UI の仕様ドキュメント。

## 1. UI概要

- **タイトル**: "Qwen3 TTS Voice Cloning"
- **サブタイトル**: "Extract speaker embeddings, clone voices, and interpolate between speakers."
- **テーマ**: ダークテーマ（暗い背景、紫/青アクセントカラー）
- **構成**: 3タブ構成のタブナビゲーション

### 技術スタック

| 層 | 技術 | 理由 |
|---|---|---|
| フロントエンド | HTMX 2.x | 軽量(14KB)、HTMLファースト、ビルド不要 |
| スタイリング | DaisyUI + Tailwind CSS | ダークテーマ内蔵、HTMXと相性良好 |
| テンプレート | Jinja2 | FastAPIネイティブ対応 |
| バックエンド | FastAPI | Python ML stackと直接統合 |
| 音声録音 | vanilla JS (getUserMedia) | HTMXだけでは不可、最小限のJS |
| セッション | サーバーサイドセッション | embedding_stateの共有 |

---

## 2. Tab 1: Extract Embedding

**目的**: 参照音声からECAPA-TDNN話者Embeddingを抽出する。

### UI要素

- **説明文**: "Upload a reference audio file to extract a speaker embedding vector (ECAPA-TDNN)."
- **左カラム** (広め):
  - 「Reference Audio」ラベル付き音声アップロードエリア（ドラッグ&ドロップ対応）
  - アップロードボタン（`<input type="file" accept="audio/*">`）
  - マイクボタン（録音対応 — `getUserMedia` API）
  - 「Extract Embedding」ボタン（紫色、`btn-primary`）
- **右カラム**:
  - 「Embedding Info」テキスト表示エリア（読み取り専用）
    - 次元数、L2ノルム、最小値、最大値
  - 「Download Embedding JSON」ファイルダウンロードリンク

### データフロー

```
ユーザー → 音声ファイルアップロード
         → POST /api/extract-embedding (multipart/form-data)
         → ECAPA-TDNN encoder推論 (marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B)
         → 2048次元 embedding ベクトル
         → セッションに保存 (session["embedding_state"])
         → HTML fragment返却 (embedding情報 + ダウンロードリンク)
```

---

## 3. Tab 2: Voice Cloning TTS

**目的**: 抽出したembeddingまたは参照音声を使ってテキストを音声に変換する。

### UI要素

- **合成テキスト入力** (`textarea`, 複数行)
- **言語選択ドロップダウン**: Auto, Chinese, English, Japanese, Korean, French, German, Spanish
- **参照音声アップロード** (任意, `file-input`)
- **Embedding JSONアップロード** (任意, `file-input`)
- **「音声生成」ボタン** (`btn-primary`)
- **生成音声プレーヤー** (`<audio>` タグ)
- **ステータス表示** (使用したembeddingソースを表示)

### 優先順位

声の指定方法（優先順位順）:
1. **Embedding JSON** アップロード
2. **Tab 1のセッションState** (自動連携)
3. **参照音声** (直接x-vectorクローニング)

### データフロー

```
ユーザー → テキスト + 言語 + (参照音声 or JSON or セッション)
         → POST /api/generate-speech (multipart/form-data)
         → TTSモデル推論 (Qwen3-TTS-12Hz-1.7B-Base)
         → WAV音声データ
         → HTML fragment返却 (audio playerを含む)
```

---

## 4. Tab 3: Voice Interpolation

**目的**: 2つの声をSLERP（球面線形補間）でブレンドして音声を生成する。

### UI要素

- **声A** の音声アップロード (`file-input`, `accept="audio/*"`)
- **声B** の音声アップロード (`file-input`, `accept="audio/*"`)
- **SLERPスライダー** (`range`, 0.0〜1.0, ステップ0.05, デフォルト0.5)
  - 0.0 = 完全に声A、1.0 = 完全に声B
  - スライダー横に現在の値を表示
- **テキスト入力** (`textarea`)
- **言語選択ドロップダウン** (Tab 2と同じ選択肢)
- **「ブレンド音声生成」ボタン** (`btn-primary`)
- **生成音声プレーヤー** (`<audio>` タグ)
- **補間情報表示**:
  - 声A L2ノルム
  - 声B L2ノルム
  - SLERP比率
  - ブレンド後ノルム

### データフロー

```
ユーザー → 声A + 声B + SLERP比率 + テキスト + 言語
         → POST /api/interpolate (multipart/form-data)
         → ECAPA-TDNN で声A・声Bのembedding抽出
         → SLERP補間 (t = 比率)
         → TTSモデル推論
         → WAV音声データ + 補間統計
         → HTML fragment返却 (audio player + 補間情報)
```

---

## 5. バックエンドAPI仕様

### Embedding抽出エンジン

- **モデル**: marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B (ECAPA-TDNN)
- **出力次元**: 2048次元
- **入力**: 音声ファイル (任意フォーマット、内部で24kHzにリサンプリング)
- **パラメータ**: サンプルレート 24kHz、Mel 128bin、n_fft=1024、hop=256

### TTSモデル

- **モデル**: Qwen/Qwen3-TTS-12Hz-1.7B-Base
- **生成モード**: `x_vector_only_mode` (speaker embeddingベース)
- **出力**: WAVフォーマット音声

### SLERP補間

```python
def slerp(v0, v1, t):
    """球面線形補間: t=0でv0、t=1でv1を返す"""
```

### 状態管理

- サーバーサイドセッション（`SessionMiddleware`）
- `session["embedding_state"]`: Tab 1で抽出したembeddingをTab 2で自動共有
- Cookie/UUIDベースのセッション識別

### FastAPI エンドポイント

| メソッド | パス | 返却 | 機能 |
|---|---|---|---|
| GET | `/` | HTML | メインページ |
| GET | `/tab/extract` | HTML fragment | Tab 1コンテンツ |
| GET | `/tab/cloning` | HTML fragment | Tab 2コンテンツ |
| GET | `/tab/interpolation` | HTML fragment | Tab 3コンテンツ |
| POST | `/api/extract-embedding` | HTML fragment | embedding抽出→結果表示 |
| POST | `/api/generate-speech` | HTML fragment | TTS生成→音声プレーヤー |
| POST | `/api/interpolate` | HTML fragment | SLERP補間→音声プレーヤー |
| GET | `/api/download-embedding` | JSON file | embedding JSONダウンロード |
| POST | `/api/upload-recording` | HTML fragment | マイク録音の受信 |

---

## 6. ファイル構造

```
examples/online_serving/qwen3_tts/web/
├── app.py                    # FastAPIアプリ（メイン）
├── templates/
│   ├── base.html             # ベーステンプレート（HTMX/DaisyUI読み込み）
│   ├── index.html            # メインページ（タブ構造）
│   ├── tabs/
│   │   ├── extract.html      # Tab 1: Embedding抽出
│   │   ├── cloning.html      # Tab 2: Voice Cloning TTS
│   │   └── interpolation.html # Tab 3: 声の補間
│   └── partials/
│       ├── embedding_info.html    # embedding情報フラグメント
│       ├── audio_player.html      # 音声プレーヤーフラグメント
│       └── status.html            # ステータス表示フラグメント
├── static/
│   ├── js/
│   │   └── audio-recorder.js # マイク録音（vanilla JS）
│   └── css/
│       └── custom.css         # カスタムスタイル（最小限）
└── README.md
```
