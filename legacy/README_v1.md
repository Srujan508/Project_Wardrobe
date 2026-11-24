# 🧥 AI Wardrobe Stylist — Project Wardrobe

An AI-powered outfit recommendation system that uses deep learning (ResNet50 + Siamese Network with Triplet Loss) to score the compatibility between clothing items. Users upload tops and bottoms, and the AI suggests the best-matching outfit from their digital closet.

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    USER (Browser)                        │
│          static/index.html  (TailwindCSS UI)             │
│    Upload clothes │ View wardrobe │ Get AI suggestions    │
└────────┬─────────┴───────────────┴───────────┬───────────┘
         │              REST API               │
         ▼                                     ▼
┌──────────────────────────────────────────────────────────┐
│              FastAPI Backend  (main.py)                   │
│                                                          │
│  POST /upload-item      ─ Save image to uploads/         │
│  GET  /get-wardrobe     ─ List all uploaded images        │
│  GET  /get-suggestion   ─ AI picks the best outfit        │
│  GET  /                 ─ Serve the frontend              │
└────────┬─────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│                   ML Pipeline                            │
│                                                          │
│  StyleExtractor (ResNet50)  → 2048-dim feature vector    │
│          ▼                                               │
│  CompatibilityEngine (FC)   → 128-dim embedding          │
│          ▼                                               │
│  Cosine Similarity          → Compatibility Score 0-100  │
└──────────────────────────────────────────────────────────┘
```

---

## 📁 File-by-File Breakdown

### 🔵 Core Application (Ready to Run — after setup)

| File | Purpose |
|---|---|
| `main.py` | **FastAPI web server**. Serves the frontend, handles uploads, runs AI inference to find the best outfit. |
| `models.py` | **ML model definitions**. `StyleExtractor` (ResNet50 feature extractor) and `CompatibilityEngine` (Siamese network for compatibility scoring). |
| `static/index.html` | **Frontend UI**. Dark-themed TailwindCSS interface with 3 tabs: Get Styled, My Wardrobe, Add Clothes. |

### 🟢 Data Preparation Pipeline (Run once to prepare training data)

| File | Purpose |
|---|---|
| `database_setup.py` | Creates a SQLite `wardrobe.db` with a `Clothing_Items` table. |
| `data_pipeline.py` | Populates the DB with mock clothing data + travel/weather tags. |
| `prepare_deepfashion.py` | Parses DeepFashion annotation files (`train_ann_file.txt`, etc.) into a cleaned CSV with category and weather tags. |
| `make_deepfashion_triplets.py` | Generates 10,000 (anchor, positive, negative) triplets from DeepFashion for training. |
| `mixing_images.py` | Audits the DeepFashion triplet CSV to remove rows with missing images → `deepfashion_triplets_fixed.csv`. |
| `dow_poly_img.py` | Downloads Polyvore outfit images from URLs in the JSON files (multi-threaded). |
| `make_polyvore_triplets.py` | Generates triplets from Polyvore outfit JSON data. |
| `audict_polyvore.py` | Audits Polyvore triplets to remove entries with missing images → `polyvore_triplets_fixed.csv`. |
| `make_reploy.py` | Alternative triplet generator using Re-PolyVore folder structure (tops/pants/shoes). |
| `triplet_maker.py` | Early prototype triplet generator using the SQLite database (not used in final pipeline). |

### 🟡 Training & Evaluation

| File | Purpose |
|---|---|
| `train_model.py` | **Main training script**. Trains the CompatibilityEngine using triplet loss. Two phases: DeepFashion (base) → Polyvore (final). |
| `test_model.py` | Tests the trained model on a single pair of images (CLI-based). |
| `test_2.py` | Evaluates model accuracy on 500 random triplets from the Polyvore dataset (accuracy benchmark). |
| `check_model.py` | Quick sanity check for the DeepFashion base model (`wardrobe_model_base.pth`). |

### 📦 Data Files (in repo)

| File | Purpose |
|---|---|
| `polyvore/*.json` | Polyvore outfit data (train/test/valid splits + fill-in-blank test). |
| `*_ann_file.txt` | DeepFashion annotation files listing image paths + attributes. |
| `allfiles.txt` | Large index file (~16 MB) listing all available image paths. |

---

## ❌ What's NOT Working / Missing

### 🔴 Critical — App Won't Start

| Issue | Details |
|---|---|
| **No `uploads/` directory** | `main.py` mounts `uploads/` as a static folder on line 14, but this directory doesn't exist. The app will **crash on startup**. |
| **No `uploads/tops/` and `uploads/bottoms/` dirs** | The upload and suggestion endpoints expect these subdirectories. They don't exist. |
| **No trained model file (`wardrobe_model_final.pth`)** | The AI model weights are missing (gitignored by `*.pth`). The app prints a warning but won't crash — however, the AI suggestion feature **will not work** without it. |
| **Missing Python dependencies** | Only `requests` is installed. You need: `torch`, `torchvision`, `fastapi`, `uvicorn`, `pandas`, `pillow`, `tqdm`. |

### 🟠 Critical — Training Pipeline Won't Work

| Issue | Details |
|---|---|
| **No `polyvore_images/` directory** | Downloaded Polyvore images are missing. You need to run `dow_poly_img.py` first (downloads ~100K+ images from URLs). |
| **No `test_images/` directory** | DeepFashion images are missing. You need to download the DeepFashion dataset separately. |
| **No generated CSV files** | `deepfashion_triplets_fixed.csv` and `polyvore_triplets_fixed.csv` haven't been generated yet. |
| **Hardcoded Linux paths in `test_model.py`** | Lines 60-61 have `/home/smirkmachine/Downloads/...` paths that won't work on Windows. |

### 🟡 Minor Issues

| Issue | Details |
|---|---|
| **No `requirements.txt`** | No dependency file exists for easy installation. |
| **`train_model.py` uses `torch.no_grad()` for feature extraction during training** | Lines 69-72 wrap the extractor in `no_grad()`, which means the unfrozen `layer4` of ResNet **won't actually learn** during training — contradicting the comment in `models.py` about unfreezing layer4. |
| **No error handling in upload** | If `uploads/tops/` doesn't exist, the upload will silently fail with a 500 error. |

---

## ✅ What IS Working (Structurally Sound)

| Component | Status |
|---|---|
| `models.py` — Model architecture | ✅ Clean, correct Siamese network design |
| `static/index.html` — Frontend UI | ✅ Functional dark-mode UI with upload, wardrobe view, and suggestion tabs |
| `main.py` — API design | ✅ Well-structured FastAPI app with proper endpoints |
| `polyvore/*.json` — Training data (metadata) | ✅ Present and ready to use |
| `*_ann_file.txt` — DeepFashion annotations | ✅ Present and ready to use |
| Data preparation scripts | ✅ Logic is sound; just need the raw image data |
| `.gitignore` | ✅ Properly excludes large files |

---

## 🚀 What You Need To Do (Step by Step)

### Step 1: Install Python Dependencies

```bash
pip install torch torchvision fastapi uvicorn pandas pillow tqdm requests python-multipart
```

> **Note:** For GPU-accelerated training, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

### Step 2: Create Required Directories

```bash
mkdir uploads\tops
mkdir uploads\bottoms
```

### Step 3: (Optional) Train the Model

If you want to train the model yourself:

#### Phase A — DeepFashion Training
1. Download the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) and place images in `test_images/`
2. Run `python prepare_deepfashion.py` — cleans annotations into `deepfashion_cleaned.csv`
3. Run `python make_deepfashion_triplets.py` — generates triplets
4. Run `python mixing_images.py` — audits and fixes the triplet CSV
5. Edit `train_model.py`: set `TRAINING_PHASE = 'deepfashion'`
6. Run `python train_model.py` → produces `wardrobe_model_base.pth`

#### Phase B — Polyvore Training (Fine-tuning)
1. Run `python dow_poly_img.py` — downloads Polyvore images (takes a while!)
2. Run `python make_polyvore_triplets.py` — generates triplets from outfit JSON
3. Run `python audict_polyvore.py` — audits and cleans the triplet CSV  
   *(OR use `python make_reploy.py` if using the Re-PolyVore folder structure)*
4. Edit `train_model.py`: set `TRAINING_PHASE = 'polyvore'`
5. Run `python train_model.py` → produces `wardrobe_model_final.pth`

#### Phase C — Evaluate
1. Run `python test_2.py` — runs accuracy benchmark on 500 triplets

### Step 4: Run the Web App

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

### Step 5: Use the App
1. Go to **Add Clothes** → upload top and bottom images
2. Go to **Get Styled** → click "Generate Perfect Outfit"
3. View your **My Wardrobe** to see all uploaded items

---

## 🛠️ Recommended Fixes to Apply

1. **Create a `requirements.txt`** — for reproducible installs
2. **Auto-create `uploads/` directories on startup** — add `os.makedirs()` to `main.py`
3. **Fix the training bug** — remove the `torch.no_grad()` block around the extractor in `train_model.py` if you want layer4 to actually fine-tune
4. **Fix hardcoded paths** — update Linux paths in `test_model.py` and `check_model.py` to use relative or Windows paths

---

## 📊 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML + TailwindCSS (CDN) |
| Backend | Python, FastAPI |
| ML Framework | PyTorch |
| Feature Extractor | ResNet50 (pretrained on ImageNet) |
| Compatibility Model | Siamese Network with Triplet Margin Loss |
| Datasets | DeepFashion + Polyvore |
| Database | SQLite (for data pipeline, not used in web app) |

---

## 📄 License

Not specified.
