import streamlit as st
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os
import json
import torch
import timm

st.set_page_config(
    page_title="AgriScan-AI | High-Precision Produce & Edibility Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Creative Botanical Glassmorphism Background & Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 45%, #e0f2f1 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        right: -5%;
        width: 450px;
        height: 450px;
        background: radial-gradient(circle, rgba(46, 125, 50, 0.14) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -10%;
        left: -5%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0, 137, 123, 0.12) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }
    .block-container {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 2.2rem 2.5rem !important;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(27, 94, 32, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.7);
    }
    .stMetric {
        background: linear-gradient(145deg, #ffffff, #f4fbf6);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    h1 { color: #1b5e20 !important; font-weight: 800 !important; }
    h2, h3 { color: #2e7d32 !important; }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2e7d32 0%, #00897b 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 1.8rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px rgba(46, 125, 50, 0.25);
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #1b5e20 0%, #00695c 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

SAFE_EDIBLE_BENCHMARK = 80.0
OPTIMAL_FRESH_AVERAGE = 92.0

# --- Load Ultra-Lightweight MobileNetV3-Small CNN (10 MB, Python 3.14 Native) ---
@st.cache_resource
def load_cnn_model():
    model = timm.create_model("mobilenetv3_small_100", pretrained=True)
    model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    labels_file = "imagenet_class_index.json"
    if not os.path.exists(labels_file):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        urllib.request.urlretrieve(url, labels_file)
    with open(labels_file, "r") as f:
        raw_idx = json.load(f)
    id2label = {int(k): v[1].replace("_", " ").lower() for k, v in raw_idx.items()}
    return model, transforms, id2label

cnn_model, cnn_transforms, id2label = load_cnn_model()

CNN_PRODUCE_MAP = {
    "granny smith": ("Apple", "Fruit"),
    "apple": ("Apple", "Fruit"),
    "pomegranate": ("Apple / Pomegranate", "Fruit"),
    "croquet ball": ("Apple", "Fruit"),
    "banana": ("Banana", "Fruit"),
    "orange": ("Orange", "Fruit"),
    "lemon": ("Lemon", "Fruit"),
    "fig": ("Fig", "Fruit"),
    "pineapple": ("Pineapple", "Fruit"),
    "strawberry": ("Strawberry", "Fruit"),
    "custard apple": ("Guava / Custard Apple", "Fruit"),
    "jackfruit": ("Jackfruit", "Fruit"),
    "bell pepper": ("Bell Pepper", "Vegetable"),
    "cucumber": ("Cucumber", "Vegetable"),
    "zucchini": ("Zucchini", "Vegetable"),
    "spaghetti squash": ("Squash", "Vegetable"),
    "acorn squash": ("Squash", "Vegetable"),
    "butternut squash": ("Butternut Squash", "Vegetable"),
    "broccoli": ("Broccoli", "Vegetable"),
    "cauliflower": ("Cauliflower", "Vegetable"),
    "head cabbage": ("Cabbage", "Vegetable"),
    "artichoke": ("Artichoke", "Vegetable"),
    "mushroom": ("Mushroom", "Vegetable"),
    "corn": ("Corn", "Vegetable"),
    "ear": ("Corn", "Vegetable"),
    "mashed potato": ("Potato", "Vegetable"),
    "hip": ("Tomato", "Vegetable")
}

NON_FOOD_TERMS = {
    "laptop", "notebook", "cellular telephone", "monitor", "screen", "keyboard",
    "mouse", "wall", "suit", "jersey", "car", "wheel", "shoe", "chair", "desk",
    "remote control", "book", "pen", "bottle", "cup", "glasses"
}

def focus_and_clarify_image(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bgr_std = cv2.resize(bgr, (400, 400))

    # GrabCut central subject isolation
    mask = np.zeros((400, 400), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (28, 28, 344, 344)
    try:
        cv2.grabCut(bgr_std, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception:
        fg_mask = np.zeros((400, 400), np.uint8)

    fg_ratio = np.sum(fg_mask) / (400.0 * 400.0)
    if fg_ratio < 0.08 or fg_ratio > 0.92:
        hsv_tmp = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2HSV)
        # Include dark/rotten low-saturation regions inside foreground when produce is heavily decayed
        fg_mask = (cv2.inRange(hsv_tmp, np.array([0, 15, 10]), np.array([180, 255, 245])) > 0).astype("uint8")

    fg_mask_255 = (fg_mask * 255).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_CLOSE, kernel)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_OPEN, kernel)

    blurred_bg = cv2.GaussianBlur(bgr_std, (35, 35), 0)
    blurred_bg = cv2.convertScaleAbs(blurred_bg, alpha=0.65, beta=0)

    gauss = cv2.GaussianBlur(bgr_std, (0, 0), 1.8)
    sharp_true_bgr = cv2.addWeighted(bgr_std, 1.25, gauss, -0.25, 0)

    alpha = (fg_mask_255.astype(np.float32) / 255.0)[:, :, None]
    focused_true_bgr = (sharp_true_bgr.astype(np.float32) * alpha + blurred_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    focused_true_pil = Image.fromarray(cv2.cvtColor(focused_true_bgr, cv2.COLOR_BGR2RGB))

    lab = cv2.cvtColor(focused_true_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clarified_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    clarified_pil = Image.fromarray(cv2.cvtColor(clarified_bgr, cv2.COLOR_BGR2RGB))

    return focused_true_bgr, focused_true_pil, clarified_pil, fg_mask_255

def classify_with_cnn_and_morphology(focused_pil, hsv, mask_clean, circularity, aspect_ratio, category_choice, optional_name):
    if optional_name.strip():
        clean_name = optional_name.strip().title()
        return True, clean_name, 99.0, [(clean_name, 99.0)]

    tensor_img = cnn_transforms(focused_pil).unsqueeze(0)
    with torch.no_grad():
        logits = cnn_model(tensor_img)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k=10)

    raw_preds = [(id2label[int(idx)], round(float(p) * 100, 1)) for p, idx in zip(top_probs, top_idxs)]

    top_raw_name, top_raw_score = raw_preds[0]
    has_produce_match = any(any(k in lbl for k in CNN_PRODUCE_MAP) for lbl, _ in raw_preds[:6])

    if any(nf in top_raw_name for nf in NON_FOOD_TERMS) and top_raw_score > 30.0 and not has_produce_match:
        return False, top_raw_name.title(), top_raw_score, []

    valid_pixels = hsv[mask_clean > 0]
    mean_hue = float(np.mean(valid_pixels[:, 0])) if len(valid_pixels) > 0 else 0.0
    mean_sat = float(np.mean(valid_pixels[:, 1])) if len(valid_pixels) > 0 else 0.0

    if mean_sat < 18 and not has_produce_match and category_choice == "Fruit":
        return False, "Non-Food / Colorless Object", 85.0, []

    produce_candidates = {}
    for lbl, score in raw_preds:
        for key, (clean_produce, prod_cat) in CNN_PRODUCE_MAP.items():
            if key in lbl:
                adj_score = score * (1.45 if prod_cat == category_choice else 0.55)
                produce_candidates[clean_produce] = produce_candidates.get(clean_produce, 0.0) + adj_score

    if "Apple / Pomegranate" in produce_candidates:
        if category_choice == "Vegetable":
            produce_candidates["Tomato"] = produce_candidates.pop("Apple / Pomegranate") * 1.3
        else:
            produce_candidates["Apple"] = produce_candidates.pop("Apple / Pomegranate") * 1.25

    if "Strawberry" in produce_candidates and circularity >= 0.78:
        produce_candidates["Apple"] = produce_candidates.get("Apple", 0.0) + produce_candidates["Strawberry"] * 1.15
        produce_candidates["Strawberry"] *= 0.45

    if not produce_candidates or max(produce_candidates.values()) < 10.0:
        if category_choice == "Fruit":
            if 22 <= mean_hue <= 54 and circularity >= 0.75:
                produce_candidates["Guava"] = 86.0
                produce_candidates["Green Apple"] = 64.0
                produce_candidates["Pear"] = 55.0
            elif mean_hue < 14 or mean_hue > 165:
                if circularity >= 0.78:
                    produce_candidates["Apple"] = 88.0
                    produce_candidates["Pomegranate"] = 64.0
                    produce_candidates["Strawberry"] = 45.0
                else:
                    produce_candidates["Strawberry"] = 85.0
                    produce_candidates["Apple"] = 50.0
            elif 14 <= mean_hue < 22:
                produce_candidates["Orange"] = 86.0
                produce_candidates["Mango"] = 68.0
            elif 54 < mean_hue <= 85:
                produce_candidates["Watermelon"] = 82.0
                produce_candidates["Lime"] = 65.0
            else:
                produce_candidates["Fruit Sample"] = 75.0
        else:
            if mean_hue < 14 or mean_hue > 165:
                produce_candidates["Tomato"] = 88.0
                produce_candidates["Red Bell Pepper"] = 65.0
            elif 35 <= mean_hue <= 82:
                if aspect_ratio < 0.55 or aspect_ratio > 1.8:
                    produce_candidates["Cucumber"] = 87.0
                    produce_candidates["Zucchini"] = 72.0
                else:
                    produce_candidates["Green Bell Pepper"] = 83.0
                    produce_candidates["Broccoli"] = 60.0
            elif 14 <= mean_hue < 34:
                produce_candidates["Potato"] = 82.0
