import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AgriScan-AI | High-Precision Produce & Edibility Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Botanical Glassmorphism Styling ---
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
        background: rgba(255, 255, 255, 0.90);
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

# High-Precision Multi-Feature Botanical Database:
# (Name, Hue1_Range, Hue2_Range, Sat_Ideal, Circularity_Ideal, Aspect_Ideal, Texture_Max)
BOTANICAL_DB = {
    "Fruit": [
        ("Apple", (0, 12), (168, 180), 165, 0.88, 0.96, 360),
        ("Green Apple", (34, 62), (25, 45), 140, 0.89, 0.97, 300),
        ("Guava", (24, 52), (18, 35), 95, 0.83, 0.90, 380),
        ("Banana", (19, 31), (14, 24), 175, 0.38, 0.40, 320),
        ("Orange", (10, 19), (15, 24), 205, 0.92, 0.98, 520),
        ("Lemon", (22, 33), (18, 28), 195, 0.74, 0.72, 400),
        ("Mango", (13, 30), (30, 50), 165, 0.68, 0.68, 300),
        ("Strawberry", (0, 9), (170, 180), 190, 0.65, 0.75, 950),
        ("Pomegranate", (0, 12), (165, 180), 145, 0.85, 0.94, 420),
        ("Watermelon", (40, 80), (30, 55), 125, 0.85, 0.88, 340),
        ("Pear", (23, 44), (18, 35), 120, 0.67, 0.69, 320),
        ("Peach", (8, 21), (15, 28), 150, 0.88, 0.96, 340),
        ("Grape", (120, 168), (35, 65), 115, 0.62, 0.75, 360)
    ],
    "Vegetable": [
        ("Tomato", (0, 11), (168, 180), 185, 0.90, 0.98, 280),
        ("Cucumber", (38, 80), (30, 55), 135, 0.33, 0.27, 380),
        ("Potato", (13, 27), (10, 22), 85, 0.76, 0.74, 460),
        ("Carrot", (8, 19), (12, 24), 200, 0.35, 0.29, 420),
        ("Green Bell Pepper", (36, 80), (28, 50), 160, 0.76, 0.86, 280),
        ("Red Bell Pepper", (0, 10), (170, 180), 195, 0.76, 0.86, 280),
        ("Yellow Bell Pepper", (19, 32), (15,
