import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import timm

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Melanoma Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Apply monospace font ONLY to actual text content -- NOT to span/div/label
       broadly, since Streamlit renders built-in icons (upload icon, expander
       arrow, sidebar collapse arrow) as literal text using an icon ligature
       font (e.g. the text "keyboard_double_arrow_left" is meant to render as
       an arrow glyph). Overriding font-family on those elements breaks the
       icon font and shows the raw ligature text instead. */
    body, p, h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stCaption, .stAlert p,
    .stButton button, .stSelectbox div[data-baseweb="select"] span,
    .stTextInput input, .stDataFrame, .stMetric label, .stMetric [data-testid="stMetricValue"] {
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Explicitly restore Streamlit's icon font on icon elements.
       IMPORTANT: use the actual font name here, not `inherit` -- `inherit`
       pulls the parent's current computed font-family, which is now our
       overridden monospace font, not the icon font. That was the bug. */
    [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Courier New', Courier, monospace !important;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
        line-height: 1.5;
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Sidebar metrics -- default Streamlit metric font is too large for
       a dense sidebar with several stacked metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }

    .stAlert {
        margin-top: 1rem;
    }

    /* Prediction result banners */
    .risk-melanoma {
        background-color: #f8d7da;
        border-left: 5px solid #CC0000;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-nevus {
        background-color: #d4edda;
        border-left: 5px solid #228B22;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========== CONFIGURATION ==========
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
VAL_DIR = PROJECT_ROOT / "data" / "preprocessed" / "val"

IMG_SIZE = 224
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Available checkpoints -- maps a friendly label to a filename.
# These are the small "head-only" checkpoints produced by
# scripts/strip_checkpoint.py from the full checkpoints scripts/train_vit.py
# writes. They contain only the trained classifier weights (a few MB) --
# the frozen ViT-L/16 backbone is re-downloaded from timm's ImageNet-pretrained
# weights at load time, since it's identical to the public pretrained model
# and was never actually modified during training. See README for details.
AVAILABLE_MODELS = {
    "ViT-L/16 (pos_weight, recommended)": "vit_l16_posweight_best_head.pt",
    "ViT-L/16 (unweighted, matches paper baseline)": "vit_l16_best_head.pt",
}

# Threshold profiles from scripts/evaluate.py / threshold_sweep.py, tuned on the
# ISIC 2019 validation set. NOTE: these were tuned per-model; the ones below
# correspond to the pos_weight model. See README for the unweighted model's numbers.
THRESHOLD_PROFILES = {
    "Default (0.50)": 0.50,
    "Balanced (0.65)": 0.65,
    "High-sensitivity (0.80)": 0.80,
}


# ========== MODEL DEFINITION ==========
# Must match scripts/train_vit.py exactly.
class MelanomaViT(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        # pretrained=True: we only ship the classifier head (see AVAILABLE_MODELS
        # comment above), so the backbone must actually fetch real ImageNet
        # weights here rather than starting from random initialization.
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=True, num_classes=0
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)


# ========== LOAD MODEL (CACHED) ==========
def _strip_classifier_prefix(state_dict):
    """Head checkpoints save keys as 'classifier.0.weight', etc. (the full
    model's state_dict naming). model.classifier.load_state_dict() expects
    them without that leading 'classifier.' prefix."""
    return {k.removeprefix("classifier."): v for k, v in state_dict.items()}


@st.cache_resource
def load_model(checkpoint_filename: str):
    model = MelanomaViT(freeze_backbone=True)
    checkpoint_path = MODELS_DIR / checkpoint_filename
    classifier_state = torch.load(checkpoint_path, map_location=DEVICE)
    model.classifier.load_state_dict(_strip_classifier_prefix(classifier_state))
    model = model.to(DEVICE)
    model.eval()
    return model


# ========== IMAGE PREPROCESSING ==========
# Matches scripts/preprocessing.py's resize+pad approach as closely as
# practical for a single uploaded image at inference time.
def resize_and_pad(image: Image.Image, target_size: int = IMG_SIZE) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ========== MAKE PREDICTION ==========
def predict(model, image: Image.Image, threshold: float):
    padded = resize_and_pad(image)
    img_tensor = eval_transform(padded).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_tensor)
        prob_nevus = torch.sigmoid(logit).item()  # class 1 = nevus (ImageFolder alphabetical order)

    prob_melanoma = 1 - prob_nevus
    prediction = "NEVUS (benign)" if prob_nevus > threshold else "MELANOMA (suspicious)"
    confidence = max(prob_nevus, prob_melanoma)

    return {
        "prediction": prediction,
        "prob_melanoma": prob_melanoma,
        "prob_nevus": prob_nevus,
        "confidence": confidence,
        "threshold": threshold,
        "padded_image": padded,
    }


# ========== MAIN APP ==========
def main():
    # ---------- Title ----------
    st.markdown('<div class="main-header">🔬 Melanoma Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload a dermoscopic image to be evaluated by a Vision Transformer model<br>'
        'Model was fine-tuned on ISIC 2019 to distinguish melanoma from nevus</div>',
        unsafe_allow_html=True
    )

    st.warning(
        "⚠️ **Research prototype, not a diagnostic tool.** This app is for educational "
        "and portfolio purposes only. It has not been clinically validated and must not "
        "be used to inform real medical decisions. See the Disclaimer in the sidebar."
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("<h3 style='font-family:Arial;'>Model Selection</h3>", unsafe_allow_html=True)

        available_checkpoints = {
            label: fname for label, fname in AVAILABLE_MODELS.items()
            if (MODELS_DIR / fname).exists()
        }

        if not available_checkpoints:
            st.error(
                "No model checkpoints found in `models/`. Run scripts/strip_checkpoint.py "
                "on a full checkpoint first (see scripts/train_vit.py), or copy a head "
                "checkpoint into the models/ folder."
            )
            st.stop()

        selected_label = st.selectbox("Choose a model:", list(available_checkpoints.keys()))
        checkpoint_filename = available_checkpoints[selected_label]
        is_posweight_model = "posweight" in checkpoint_filename

        st.divider()
        st.markdown("<h3 style='font-family:Arial;'>Decision Threshold</h3>", unsafe_allow_html=True)
        threshold_label = st.select_slider(
            "Operating point:",
            options=list(THRESHOLD_PROFILES.keys()),
            value="Default (0.50)"
        )
        threshold = THRESHOLD_PROFILES[threshold_label]

        st.divider()
        st.markdown("<h3 style='font-family:Arial;'>Model Information</h3>", unsafe_allow_html=True)
        st.metric("Architecture", "ViT-L/16 (frozen backbone)")
        st.metric("Training Data", "ISIC 2019 (13,917 images)")

        if is_posweight_model:
            st.metric("ROC-AUC (val)", "0.9158")
            st.caption(
                "Trained with a class-weighted loss (pos_weight ≈ 0.35) to counteract "
                "ISIC's ~74%/26% nevus/melanoma imbalance. Prioritizes catching melanoma "
                "over minimizing false alarms."
            )
        else:
            st.metric("ROC-AUC (val)", "0.9171")
            st.caption(
                "Unweighted baseline, closely matching the frozen-backbone ViT-L/16 "
                "result reported in Garcia et al. 2025 (Cancers 17, 3447)."
            )

        st.write(f"**At the selected threshold ({threshold}):**")
        st.write("- See README.md for the full threshold-profile table")
        st.write("- Higher threshold → catches more melanoma, more false alarms on nevus")
        st.write("- Lower threshold → fewer false alarms, more missed melanoma")

    model = load_model(checkpoint_filename)

    col1, col2 = st.columns(2)

    # ---------- Upload / Example ----------
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image",
            type=["png", "jpg", "jpeg"],
            help="Upload a dermoscopic (close-up, magnified) image of a skin lesion",
            label_visibility="collapsed"
        )

        # Example images, if a val/ split is present locally
        if VAL_DIR.exists():
            st.subheader("Or select an example image")
            example_category = st.radio(
                "Example category:",
                ["melanoma", "nevus"],
                horizontal=True
            )
            example_dir = VAL_DIR / example_category
            example_files = sorted(example_dir.glob("*.jpg"))[:50]  # cap the list for UI speed

            if example_files:
                selected_example = st.selectbox(
                    "Choose an example image:",
                    example_files,
                    format_func=lambda x: x.name
                )
                example_image = Image.open(selected_example).convert("RGB")
                st.image(example_image, caption=f"Example: {selected_example.name}", use_container_width=True)

                if uploaded_file is None:
                    uploaded_file = selected_example
            else:
                st.info(f"No example images found in {example_dir}")
        else:
            st.caption(
                "No local val/ dataset found -- example image picker is unavailable. "
                "Upload your own image above."
            )

    # ---------- Prediction ----------
    with col2:
        st.subheader("Prediction Result")
        result = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            result = predict(model, image, threshold)

            if result["prediction"].startswith("MELANOMA"):
                st.markdown(
                    f'<div class="risk-melanoma"><h3>⚠️ {result["prediction"]}</h3></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="risk-nevus"><h3>✓ {result["prediction"]}</h3></div>',
                    unsafe_allow_html=True
                )

            st.metric(
                "Model Confidence",
                f"{result['confidence']:.2%}",
                help="How confident the model is in this specific prediction"
            )

    # ---------- Display image and probabilities ----------
    if uploaded_file is not None and result is not None:
        col_img, col_details = st.columns([1, 1])
        with col_img:
            st.subheader("Input Image")
            st.image(image, caption="Original upload", use_container_width=True)
            st.image(result["padded_image"], caption="Preprocessed (224×224, resized+padded)", use_container_width=True)

        with col_details:
            st.subheader("Prediction Details")
            st.write("**Probability Scores:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Melanoma", f"{result['prob_melanoma']:.4f}", help="Model's estimated probability of melanoma")
            with col_p2:
                st.metric("Nevus", f"{result['prob_nevus']:.4f}", help="Model's estimated probability of nevus (benign)")

            st.write("**Classification Logic:**")
            st.write(f"- If P(nevus) > {result['threshold']} → predict NEVUS")
            st.write(f"- If P(nevus) ≤ {result['threshold']} → predict MELANOMA")
            st.divider()

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 3))
            categories = ["Melanoma", "Nevus"]
            probabilities = [result["prob_melanoma"], result["prob_nevus"]]
            colors = ["#CC0000", "#228B22"]
            bars = ax.barh(categories, probabilities, color=colors)
            ax.axvline(1 - result["threshold"], color="black", linestyle="--", linewidth=2,
                       label=f"Melanoma decision line")
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probability", fontsize=11, fontweight="bold")
            ax.set_title("Model Output Probabilities", fontsize=12, fontweight="bold")
            ax.legend()
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                ax.text(prob + 0.02, i, f"{prob:.4f}", va="center", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # ---------- Batch Processing ----------
    st.divider()
    st.subheader("Batch Processing")
    uploaded_files = st.file_uploader(
        "Upload multiple images for batch predictions",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_uploader",
        label_visibility="collapsed"
    )

    if uploaded_files and len(uploaded_files) > 0:
        st.write(f"Processing {len(uploaded_files)} images...")
        results_list = []
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            result = predict(model, image, threshold)
            results_list.append({
                "Filename": file.name,
                "Prediction": result["prediction"],
                "P(Melanoma)": f"{result['prob_melanoma']:.4f}",
                "P(Nevus)": f"{result['prob_nevus']:.4f}",
                "Confidence": f"{result['confidence']:.2%}",
            })
            progress_bar.progress((idx + 1) / len(uploaded_files))

        df = pd.DataFrame(results_list)

        def highlight_melanoma(row):
            is_mel = row["Prediction"].startswith("MELANOMA")
            style = "background-color: #FFCCCC; font-family: Arial;" if is_mel else "font-family: Arial;"
            return [style for _ in row]

        st.subheader("Batch Results")
        st.dataframe(df.style.apply(highlight_melanoma, axis=1), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results as CSV", csv, "melanoma_batch_results.csv", "text/csv")


main()
