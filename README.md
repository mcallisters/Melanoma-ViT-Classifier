# Melanoma Classification with Vision Transformers

An end-to-end **AI-powered medical imaging pipeline** that classifies dermoscopic skin lesion images as melanoma or nevus (benign mole) using a fine-tuned Vision Transformer. The project reproduces and extends the methodology of Garcia et al. 2025, *"Clinical Application of Vision Transformers for Melanoma Classification: A Multi-Dataset Evaluation Study"* (Cancers 17, 3447).

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Project Overview

This project implements a machine learning pipeline for melanoma classification:

1. **Data Preprocessing**: ISIC 2019 filtering (melanoma vs. nevus), stratified train/val split, aspect-ratio-preserving resize
2. **Model Training**: Transfer learning with ViT-L/16 (frozen backbone) on ISIC 2019, trained on Google Colab (T4 GPU)
3. **Class Imbalance Handling**: Weighted loss (`pos_weight`) to counteract ISIC's ~74%/26% nevus/melanoma imbalance
4. **Model Evaluation**: ROC-AUC, accuracy, sensitivity/specificity across multiple decision thresholds
5. **Web Deployment**: Interactive Streamlit application for real-time predictions

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It is NOT intended for clinical diagnosis or medical decision-making.

---

## 📊 Dataset

- **Source**: [ISIC 2019 Challenge dataset](https://challenge.isic-archive.com/data/#2019) (International Skin Imaging Collaboration)
- **Classes**:
  - `melanoma` — malignant lesion
  - `nevus` — benign mole
- **Filtering**: Full ISIC 2019 archive (25,331 images, 9 diagnosis categories) filtered down to melanoma/nevus only (17,397 images), since this project follows the paper's binary classification design
- **Split**: Stratified 80/20 train/val (not the paper's original split, which isn't publicly documented) — 13,917 train / 3,480 val
  - Train: 3,617 melanoma / 10,300 nevus
  - Val: 905 melanoma / 2,575 nevus
- **Image Size**: 224×224 pixels, resized with aspect-ratio-preserving black padding (not naive stretch)
- **External test set**: A held-out set of unique samples (`data/raw/unique_samples/`), analogous to the paper's MN187 biopsy-confirmed external validation set — evaluation on this set is planned but not yet complete as of this writing

---

## 🏗️ Project Structure

```
Melanoma-ViT-Classifier/
├── data/
│   ├── raw/
│   │   ├── isic_2019/
│   │   │   ├── isic_2019_training_groundtruth.csv
│   │   │   └── isic_2019_training_input/
│   │   └── unique_samples/            # Held-out external test set (not yet evaluated)
│   └── preprocessed/
│       ├── train/{melanoma,nevus}/
│       └── val/{melanoma,nevus}/
├── models/
│   ├── vit_l16_best.pt                # Unweighted baseline (matches paper methodology)
│   ├── vit_l16_posweight_best.pt      # Class-weighted version (better melanoma sensitivity)
│   ├── training_history.json
│   ├── vit_l16_posweight_history.json
│   └── roc_curve_val.png
├── scripts/
│   ├── preprocessing.py               # CSV filtering, stratified split, resize+pad
│   ├── check_preprocessing.py         # One-off sanity check on preprocessing output
│   ├── train_vit.py                   # ViT-L/16 fine-tuning (pos_weight + resume support)
│   ├── train_baseline.py              # Generalized trainer for CNN/ViT baseline comparisons
│   ├── evaluate.py                    # ROC-AUC + multi-threshold profile evaluation
│   └── threshold_sweep.py             # Full threshold sweep for sensitivity/specificity tradeoff
├── notebooks/
│   └── colab_training.ipynb           # T4-adapted training notebook
├── streamlit_app.py                   # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 🚀 Features

### Data Preprocessing (`scripts/preprocessing.py`)
- **CSV-based labeling**: ISIC 2019 images are unlabeled by filename; labels come from a separate ground-truth CSV, filtered here to melanoma (`MEL`) and nevus (`NV`) only
- **Stratified splitting**: 80/20 train/val split preserving class ratio in both sets
- **Aspect-ratio-preserving resize**: Images are scaled to fit within 224×224 and padded with black (not stretched), avoiding lesion distortion

### Model Training (`scripts/train_vit.py`)
- **Architecture**: ViT-L/16 (`timm`'s `vit_large_patch16_224`, ImageNet-1k pretrained), frozen backbone + 4-layer classification head (256→128→64→1), matching the paper's Figure 1 design
- **Class Balancing**: `BCEWithLogitsLoss(pos_weight=...)`, computed automatically from the actual training class counts, to counteract ISIC's imbalance
- **Resume support**: `--resume` flag reloads the last checkpoint and carries forward the best validation accuracy, so multi-session Colab training (across free-tier disconnects) doesn't lose progress
- **Per-epoch history logging**: Training history saves after every epoch, not just at the end, so a disconnect doesn't lose the record of progress made so far

### Model Evaluation (`scripts/evaluate.py`, `scripts/threshold_sweep.py`)
- **ROC-AUC**: The paper's primary, threshold-independent metric
- **Multi-threshold reporting**: Rather than a single 0.5 cutoff, reports accuracy/sensitivity/specificity at three named operating points (default, balanced, high-sensitivity) — since the class imbalance makes the default threshold a poor choice for melanoma recall
- **Confusion matrix** at the default threshold

### Web Application (`streamlit_app.py`)
- **Model selector**: Switch between the unweighted baseline and the class-weighted model
- **Adjustable decision threshold**: Slider across the three threshold profiles from `evaluate.py`
- **Single Image Upload**: Upload and classify individual dermoscopic images
- **Batch Processing**: Classify multiple images at once, with CSV export
- **Probability Visualization**: Bar chart showing melanoma/nevus probability split relative to the chosen decision threshold

---

## 📦 Installation

### Prerequisites
- Python 3.11 (conda environment recommended)
- macOS with Apple Silicon (MPS) or a CUDA-capable GPU (training was done on Google Colab's T4)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Melanoma-ViT-Classifier.git
cd Melanoma-ViT-Classifier
```

### 2. Create Environment

```bash
conda create -n local_llm python=3.11
conda activate local_llm
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
```txt
torch>=2.13.0
torchvision>=0.28.0
numpy>=2.4.0
timm
streamlit
grad-cam
einops
albumentations
opencv-python
pillow
pandas
scikit-learn
scipy
matplotlib
seaborn
imagehash
```

**Note on macOS + conda:** if you hit an `OMP: Error #15` (duplicate OpenMP runtime) crash, set:
```bash
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n local_llm
```
This is a known, common conflict between conda's `llvm-openmp` and pip-bundled OpenMP copies in `torch`/`opencv`/`albumentations` wheels — see project history for details.

---

## 🎓 Usage Guide

### Step 1: Download ISIC 2019 Data

```
https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
```

Unzip the images into `data/raw/isic_2019/isic_2019_training_input/` and place the CSV alongside it.

### Step 2: Preprocess the Data

```bash
python scripts/preprocessing.py
```

**What it does**:
- Filters the ground-truth CSV to melanoma/nevus rows only
- Performs a stratified 80/20 train/val split
- Resizes and pads each image to 224×224, preserving aspect ratio
- Writes output to `data/preprocessed/{train,val}/{melanoma,nevus}/`

### Step 3: Train the Model

**Local debug run** (M1 Pro / MPS, tiny subset, fast sanity check):
```bash
python scripts/train_vit.py --subset 200 --epochs 2 --batch_size 8
```

**Full training run** (Google Colab, T4 GPU):
```bash
python scripts/train_vit.py --epochs 20 --batch_size 32 --accum_steps 4
```

**Resume an interrupted session**:
```bash
python scripts/train_vit.py --epochs 10 --batch_size 32 --accum_steps 4 --resume
```

A ready-to-run version of this pipeline is also available as `notebooks/colab_training.ipynb`.

### Step 4: Evaluate the Model

```bash
python scripts/evaluate.py
```

**What it does**:
- Loads a trained checkpoint and runs inference on the validation set
- Reports ROC-AUC and a confusion matrix
- Reports accuracy/sensitivity/specificity at three threshold profiles (default, balanced, high-sensitivity)
- Saves an ROC curve plot to `models/roc_curve_val.png`

To see the full sensitivity/specificity tradeoff curve across all thresholds:
```bash
python scripts/threshold_sweep.py
```

### Step 5: Run the Web Application

```bash
streamlit run streamlit_app.py
```

**Usage**:
1. Select which trained model to use (unweighted baseline or class-weighted)
2. Choose a decision threshold profile
3. Upload a dermoscopic image (or select an example from the validation set)
4. View prediction, confidence, and the probability breakdown
5. Optionally batch-process multiple images and export results as CSV

---

## 🧠 Model Architecture

### Base Model: ViT-L/16
- **Pre-trained Weights**: ImageNet-1k (via `timm`)
- **Configuration**: 24 transformer layers, 16 attention heads, hidden size 1024, ~304M total parameters
- **Transfer Learning Approach**: Backbone frozen; only the classification head is trained
- **Classification Head**: 4 fully connected layers (256 → 128 → 64 → 1), ReLU activations, dropout 0.5 after each layer, sigmoid output (via `BCEWithLogitsLoss`)

### Why This Architecture?
- Matches the paper's exact design (Figure 1), enabling direct comparison to their published results
- Frozen backbone keeps training feasible on a single Colab T4 (16GB VRAM) and an M1 Pro, versus the paper's dual A5000 (48GB) setup
- Self-attention captures global lesion structure (asymmetry, border irregularity) that CNNs' local receptive fields can miss

### Training Configuration
```python
optimizer = AdamW(lr=2e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(T_max=NUM_EPOCHS)
criterion = BCEWithLogitsLoss(pos_weight=melanoma_count/nevus_count)  # class-weighted variant
batch_size = 32 (effective 128 via 4-step gradient accumulation)
augmentation = RandomResizedCrop(0.7-1.0 scale) + RandomHorizontalFlip + RandAugment
```

---

## 📈 Model Performance

Two model variants were trained and evaluated on the ISIC 2019 validation set (3,480 images: 905 melanoma, 2,575 nevus):

### Unweighted Baseline (`vit_l16_best.pt`)
Matches the paper's exact loss configuration. Trained 18 epochs (across two Colab sessions).

| Metric | Value |
|---|---|
| ROC-AUC | 0.9171 |
| Accuracy (threshold=0.50) | 0.8759 |
| Melanoma Sensitivity (threshold=0.50) | 0.6055 |
| Nevus Sensitivity (threshold=0.50) | 0.9709 |

This closely matches the paper's own reported frozen-backbone ViT-L/16 result (ROC-AUC 0.901–0.921 across their runs). However, the default threshold badly undersells melanoma detection — a direct consequence of ISIC's ~74%/26% class imbalance.

### Class-Weighted Model (`vit_l16_posweight_best.pt`)
Same architecture, trained 20 epochs from scratch with `pos_weight ≈ 0.351` to counteract the class imbalance.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9158 |
| Accuracy (threshold=0.50) | 0.8557 |
| Melanoma Sensitivity (threshold=0.50) | **0.7657** |
| Nevus Sensitivity (threshold=0.50) | 0.8874 |

**Key finding**: ROC-AUC is essentially unchanged between the two models (0.9171 vs. 0.9158 — within normal run-to-run noise), confirming the underlying discriminative power is the same. What changes is *where the default decision boundary sits* — the class-weighted model catches substantially more true melanomas (76.6% vs. 60.6%) at the standard 0.5 cutoff, at a modest cost to overall accuracy and nevus sensitivity. This is a clinically meaningful improvement: missing a melanoma (false negative) is far costlier than an unnecessary follow-up on a benign nevus (false positive).

### Threshold Profile Comparison (class-weighted model)

| Profile | Threshold | Accuracy | Melanoma Sens. | Nevus Sens. |
|---|---|---|---|---|
| Default | 0.50 | 0.8557 | 0.7657 | 0.8874 |
| Balanced | 0.65 | 0.7957 | 0.8619 | 0.7724 |
| High-sensitivity | 0.80 | 0.6557 | 0.9591 | 0.5491 |

At the high-sensitivity operating point, the model catches 95.9% of true melanomas — at the cost of a much higher false-positive rate on benign nevi. Which operating point is "correct" depends on clinical context; a screening tool intended to minimize missed melanomas would reasonably favor a higher threshold than a tool aiming to minimize unnecessary biopsies.

---

## 🔬 Comparison to Garcia et al. 2025

| | Garcia et al. (ViT-L/16, frozen) | This project (unweighted) | This project (class-weighted) |
|---|---|---|---|
| Training data | ISIC 2019 (17,397 images, original split) | ISIC 2019 (17,397 images, stratified 80/20 split) | Same |
| Hardware | 2× NVIDIA A5000 (48GB) | Google Colab T4 (16GB) | Same |
| Epochs | 30 | 18 (2 sessions) | 20 |
| ROC-AUC (val) | 0.901–0.921 | 0.9171 | 0.9158 |
| Class balancing | GAN-based oversampling (Section 2.4) | None | `pos_weight` reweighting |

The paper's own GAN-based augmentation pipeline (training separate StyleGAN2-ADA models per class, generating 50,000 synthetic images, confidence-filtering them) achieved a further improvement to ROC-AUC 0.915–0.926. This project used a lighter-weight alternative (`pos_weight` loss reweighting) to address the same underlying class imbalance, trading the multi-day GAN training cost for a much faster fix that improved melanoma sensitivity substantially without meaningfully changing ROC-AUC.

---

## 🚧 Limitations

1. **No independent external test set evaluated yet**: Unlike the paper's MN187 biopsy-confirmed external set, this project's held-out `unique_samples/` set has not yet been evaluated — current results are validation-set performance only (same distribution as training data)
2. **Shorter training runs**: 18-20 epochs vs. the paper's 30, due to Colab free-tier session limits
3. **Different train/val split**: A stratified random 80/20 split, not the paper's original (undocumented) split — exact numbers aren't directly comparable, though methodology is equivalent
4. **No GAN-based augmentation**: This project used loss reweighting instead of the paper's synthetic data pipeline
5. **No baseline model comparisons yet**: ResNet-152, DenseNet-201, EfficientNet-B7, ConvNeXt-XL, and ViT-B/16 comparisons (matching the paper's Table 2) are supported by `train_baseline.py` but not yet run
6. **No attention-based interpretability yet**: The paper's attention rollout visualization (Section 2.6) is planned but not yet implemented
7. **No fine-tuning stage**: The paper's unfreezing of the last 6 transformer layers (Section 3.3, which improved their ROC-AUC to 0.926) has not yet been attempted here

---

## 🔮 Future Improvements

- [ ] Evaluate on the held-out `unique_samples/` external test set
- [ ] Train and evaluate baseline models (ResNet-152, ViT-B/16, etc.) via `train_baseline.py`
- [ ] Implement attention rollout visualization for interpretability
- [ ] Attempt the paper's fine-tuning stage (unfreeze last 6 transformer layers)
- [ ] Explore GAN-based data augmentation for direct comparison to the paper's approach
- [ ] DeLong's test for statistical comparison between model variants

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. It is NOT a medical device and is NOT intended for clinical use, medical diagnosis, or treatment decisions.

- ❌ Do NOT use for patient diagnosis
- ❌ Do NOT replace professional medical advice
- ❌ Do NOT use in clinical settings without proper validation
- ✅ Consult qualified healthcare professionals for medical decisions

The developers assume no liability for any medical decisions made using this software.

---

## 📧 Contact

**Project Maintainer**: Sean McAllister
**Email**: sean.david.mcallister@gmail.com
**GitHub**: https://github.com/mcallisters

---

## 🙏 Acknowledgments

- Garcia, A.; Zhou, J.; Pinero-Crespo, G.; Beachkofsky, T.; Huang, X. for the original methodology this project reproduces and extends (Cancers 2025, 17, 3447)
- International Skin Imaging Collaboration (ISIC) for the publicly available dermoscopic image archive
- PyTorch and `timm` maintainers for the deep learning framework and pretrained model zoo
- Streamlit for the web application framework

---

## 📚 Additional Resources

- [Garcia et al. 2025, Cancers 17, 3447](https://doi.org/10.3390/cancers17213447)
- [ISIC 2019 Challenge Data](https://challenge.isic-archive.com/data/#2019)
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929)
- [timm Documentation](https://timm.fast.ai/)

---

**Built for advancing melanoma detection research**
