# 🔬 Image Forensics & Tampering Detection System

A production-ready, modular image forensics pipeline that detects digital manipulation using multiple independent forensic techniques fused into an ensemble confidence score.

---

## 📁 Project Structure

```
image_forensics/
├── modules/
│   ├── __init__.py
│   ├── ela_detector.py        # Error Level Analysis
│   ├── noise_analyzer.py      # PRNU noise inconsistency detection
│   ├── copy_move_detector.py  # ORB keypoint copy-move forgery detection
│   ├── metadata_analyzer.py   # EXIF metadata forensics
│   ├── frequency_analyzer.py  # FFT/DCT frequency domain analysis
│   └── report_generator.py    # PDF forensic report generator
├── tests/
│   └── test_modules.py        # Unit + integration tests
├── pipeline.py                # Orchestrator — runs all modules, fuses scores
├── server.py                  # FastAPI REST API server
├── cli.py                     # Command-line interface
├── requirements.txt
└── README.md

# React Dashboard (separate)
ForensicsDashboard.jsx          # Upload UI + results visualization
```

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
cd image_forensics
pip install -r requirements.txt
```

> Python 3.9+ recommended. GPU optional (PyTorch is listed but CPU-only is fine for all included modules).

### 2. Install System ExifTool (optional but recommended)

```bash
# macOS
brew install exiftool

# Ubuntu/Debian
sudo apt-get install libimage-exiftool-perl
```

---

## 🚀 Usage

### Option A — Command Line

```bash
# Full analysis with PDF report
python cli.py --image path/to/photo.jpg

# Specify output directory
python cli.py --image photo.jpg --output ./results

# JSON output only (no PDF, great for pipelines)
python cli.py --image photo.jpg --json-only

# Skip PDF generation
python cli.py --image photo.jpg --no-report
```

**CLI output example:**
```
============================================================
  FORENSIC ANALYSIS COMPLETE
============================================================
  File         : suspect_photo.jpg
  Size         : 1920×1080px
  Overall Score: 74.3%
  Verdict      : LIKELY MANIPULATED
  Manipulation : Copy-Move Forgery, Regional Pixel Editing

  Module Scores:
    Metadata Analysis            [░░░░░░░░░░░░░░░░░░░░]   8.0%
    Error Level Analysis         [████████████████░░░░]  81.2%
    Noise Inconsistency          [████████░░░░░░░░░░░░]  42.5%
    Copy-Move Detection          [█████████████░░░░░░░]  67.8%
    Frequency Analysis           [████░░░░░░░░░░░░░░░░]  22.1%

  PDF Report   : output/forensic_report_suspect_photo_20250101_120000.pdf
============================================================
```

---

### Option B — REST API

```bash
# Start the server
python server.py
# Server runs at http://localhost:8000
```

#### Analyze an Image
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@suspect_photo.jpg"
```

#### Download PDF Report
```bash
curl http://localhost:8000/report/{analysis_id} --output report.pdf
```

#### API Docs
Visit `http://localhost:8000/docs` for interactive Swagger UI.

---

### Option C — Python API

```python
from pipeline import ForensicPipeline

pipeline = ForensicPipeline(output_dir="./output")
results = pipeline.analyze("suspect_photo.jpg", generate_report=True)

print(f"Score: {results['overall_score']}%")
print(f"Verdict: {results['verdict']}")
print(f"Manipulation: {results['manipulation_type']}")
print(f"Report: {results['report_path']}")

# Access per-module details
for module_name, info in results['module_scores'].items():
    print(f"  {module_name}: {info['score']}% — {info['key_finding']}")
```

---

### Option D — React Dashboard

Point the React app at `http://localhost:8000` and open `ForensicsDashboard.jsx`.

The dashboard provides:
- Drag-and-drop image upload
- Live confidence ring gauge
- Per-module score cards with expandable flag lists
- Multi-tab visualization viewer (ELA map, noise heatmap, FFT spectrum, etc.)
- One-click PDF report download

---

## 🔬 Detection Modules

### 1. Error Level Analysis (ELA)
- Re-saves image at known JPEG quality and computes pixel-level difference
- Authentic images compress uniformly; tampered regions show anomalous error levels
- **Best at**: Detecting region replacements, AI-generated patches, filter application

### 2. Noise Inconsistency (PRNU)
- Extracts sensor noise residual via wavelet denoising
- Analyzes per-block noise variance; flags outlier blocks
- **Best at**: Splice detection (foreign image regions have different noise)

### 3. Copy-Move Forgery Detection
- ORB keypoint extraction + BFMatcher within same image
- DBSCAN clustering of matched point pairs to localize regions
- **Best at**: Object duplication, region cloning to hide/reveal content

### 4. Metadata Analysis
- EXIF parsing for: software tags, timestamp anomalies, thumbnail mismatches
- Known editing software signature database (Photoshop, GIMP, AI tools, etc.)
- **Best at**: Chain-of-custody verification, identifying editing tools used

### 5. Frequency Domain Analysis (FFT/DCT)
- Detects periodic DCT histogram dips → double JPEG compression
- High-frequency energy ratio analysis → AI generation detection
- Spectral peak detection → grid noise from splicing/stitching
- **Best at**: Double compression, resizing artifacts, AI-generated content

---

## 📊 Score Fusion

Scores are fused via **weighted ensemble**:

| Module | Weight |
|--------|--------|
| Error Level Analysis | 25% |
| Noise Inconsistency | 20% |
| Copy-Move Detection | 20% |
| Metadata Analysis | 20% |
| Frequency Analysis | 15% |

A **corroboration boost** (+8–15%) is applied when 2+ modules independently flag high scores, as independent convergence increases confidence.

### Verdict Thresholds
| Score | Verdict |
|-------|---------|
| ≥ 70% | LIKELY MANIPULATED |
| 40–69% | POSSIBLY MANIPULATED |
| < 40% | LIKELY AUTHENTIC |

---

## 🧪 Running Tests

```bash
pip install pytest
cd image_forensics
python -m pytest tests/ -v
```

Tests use **synthetic images** — no external datasets required.

---

## 📄 Report Output

The PDF report includes:
- Executive summary with verdict and confidence
- Visual confidence gauge
- Module-by-module results table
- Detailed flags and findings per module
- Annotated image with suspicious region bounding boxes
- ELA map, noise heatmap, FFT spectrum, DCT histogram
- Full EXIF metadata inventory

---

## 🔧 Configuration

Tune module parameters in `pipeline.py`:

```python
self.modules = {
    "Error Level Analysis": ELADetector(quality=90, amplification=10),
    "Noise Inconsistency":  NoiseAnalyzer(block_size=64, overlap=32),
    "Copy-Move Detection":  CopyMoveDetector(min_match_distance=30, num_keypoints=2000),
    ...
}
```

---

## ⚠️ Limitations

- **Social media re-uploads** destroy most forensic traces (JPEG compression, EXIF strip)
- **ELA** is only reliable on JPEG; PNG/TIFF use pixel-domain analysis
- **Copy-move** ORB approach may miss regions with low texture
- No module replaces expert human forensic review for legal use
- AI-generated image detection is an active research area — false negatives expected

---

## 🗺️ Roadmap / Extensions

- [ ] Integrate ManTraNet / MVSS-Net deep learning localization
- [ ] GAN fingerprint detector (CNN trained on StyleGAN/Stable Diffusion artifacts)
- [ ] Batch processing mode for dataset-scale analysis
- [ ] REST API authentication + rate limiting
- [ ] Redis-backed result persistence
- [ ] Docker containerization
