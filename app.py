import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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

# --- Load Higher-Accuracy CLIP Large Vision-Language Model ---
@st.cache_resource
def load_clip_model():
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor

clip_model, clip_processor = load_clip_model()

FRUITS_LIST = [
    "Guava", "Apple", "Banana", "Orange", "Lemon", "Lime", "Mango", "Strawberry",
    "Pomegranate", "Fig", "Grape", "Watermelon", "Cantaloupe", "Peach", "Pear",
    "Plum", "Kiwi", "Pineapple", "Papaya", "Avocado", "Date", "Apricot",
    "Cherry", "Blueberry", "Raspberry", "Coconut", "Persimmon", "Mandarin"
]

VEGETABLES_LIST = [
    "Tomato", "Cucumber", "Potato", "Sweet Potato", "Onion", "Garlic", "Carrot",
    "Bell Pepper", "Chili Pepper", "Eggplant", "Zucchini", "Broccoli", "Cauliflower",
    "Cabbage", "Lettuce", "Spinach", "Corn", "Mushroom", "Pumpkin", "Radish",
    "Beetroot", "Turnip", "Okra", "Green Beans", "Peas", "Artichoke", "Celery"
]

NON_FOOD_LIST = [
    "plastic object", "electronic device", "metal tool", "paper or book",
    "clothing or fabric", "wooden furniture", "human hand or face", "wall or floor"
]

SAFE_EDIBLE_BENCHMARK = 80.0
OPTIMAL_FRESH_AVERAGE = 92.0

def clarify_and_focus_produce(pil_image, category_choice):
    img_rgb = np.array(pil_image.convert("RGB"))
    h_orig, w_orig, _ = img_rgb.shape

    candidate_boxes = [(0, 0, w_orig, h_orig)]
    cx1, cy1 = int(w_orig * 0.16), int(h_orig * 0.16)
    cx2, cy2 = int(w_orig * 0.84), int(h_orig * 0.84)
    candidate_boxes.append((cx1, cy1, cx2, cy2))

    grid_size = 3
    step_x, step_y = w_orig // grid_size, h_orig // grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            x1 = max(0, c * step_x - int(step_x * 0.25))
            y1 = max(0, r * step_y - int(step_y * 0.25))
            x2 = min(w_orig, (c + 1) * step_x + int(step_x * 0.25))
            y2 = min(h_orig, (r + 1) * step_y + int(step_y * 0.25))
            if (x2 - x1) > 60 and (y2 - y1) > 60:
                candidate_
