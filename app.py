import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(
    page_title="AgriScan-AI | Precision Produce & Edibility Diagnostics",
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
        background: rgba(255, 255, 255, 0.85);
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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clip_model():
    model_id = "openai/clip-vit-base-patch32"
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

def classify_with_clip(pil_image, category_choice):
    img_rgb = pil_image.convert("RGB")
    
    # Step A: Check edible food vs non-food object
    domain_prompts = [
        f"a photo of a fresh or rotten {category_choice.lower()} fruit or vegetable food",
        "a photo of a non-food object, electronics, plastic, metal, wall, or person"
    ]
    inputs_domain = clip_processor(text=domain_prompts, images=img_rgb, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_domain = clip_model(**inputs_domain).logits_per_image.softmax(dim=1)[0].tolist()
    
    if (probs_domain[0] * 100) < 45.0:
        nf_prompts = [f"a photo of {item}" for item in NON_FOOD_LIST]
        inputs_nf = clip_processor(text=nf_prompts, images=img_rgb, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_nf = clip_model(**inputs_nf).logits_per_image.softmax(dim=1)[0].tolist()
        best_idx = int(np.argmax(probs_nf))
        return False, NON_FOOD_LIST[best_idx].title(), round(probs_nf[best_idx] * 100, 1), 0.0, []

    # Step B: Exact Fruit or Vegetable identification
    target_list = FRUITS_LIST if category_choice == "Fruit" else VEGETABLES_LIST
    species_prompts = [f"a close-up photo of a {item.lower()}" for item in target_list]
    inputs_sp = clip_processor(text=species_prompts, images=img_rgb, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_sp = clip_model(**inputs_sp).logits_per_image.softmax(dim=1)[0].tolist()
    
    ranked_indices = np.argsort(probs_sp)[::-1]
    top_item = target_list[ranked_indices[0]]
    top_conf = round(probs_sp[ranked_indices[0]] * 100, 1)
    top3 = [(target_list[i], round(probs_sp[i] * 100, 1)) for i in ranked_indices[:3]]

    # Step C: Fresh vs Rotten Semantic Assessment
    state_prompts = [
        f"a fresh, healthy, clean, ripe {top_item.lower()} safe to eat",
        f"a rotten, moldy, spoiled, bruised, decayed {top_item.lower()} unsafe to eat"
    ]
    inputs_st = clip_processor(text=state_prompts, images=img_rgb, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_st = clip_model(**inputs_st).logits_per_image.softmax(dim=1)[0].tolist()
    
    ai_fresh_prob = round(probs_st[0] * 100, 1)
    return True, top_item, top_conf, ai_fresh_prob, top3

def analyze_visual_defects(pil_image, ai_fresh_prob):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (400, 400))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Object Segmentation
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask_sat = cv2.inRange(blur, np.array([0, 28, 25]), np.array([180, 255, 255]))
    kernel = np.ones((7, 7), np.uint8)
    mask_clean = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = img_resized.copy()
    circularity, solidity, object_area = 0.0, 1.0, 400 * 400

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 1000:
            object_area = area
            x, y, w, h = cv2.boundingRect(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = round(4 * np.pi * (area / (perimeter * perimeter)), 2)
            hull_area = cv2.contourArea(cv2.convexHull(largest))
            if hull_area > 0:
                solidity = round(float(area) / hull_area, 2)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 2)
            cv2.drawContours(annotated_img, [largest], -1, (52, 152, 219), 2)

    # 2. Bad Marks, Dark Rot & Mold Detection
    rot_dark = cv2.inRange(hsv, np.array([0, 25, 10]), np.array([28, 255, 100]))
    mold_mask = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 38, 225]))
    defect_mask = cv2.bitwise_and(cv2.bitwise_or(rot_dark, mold_mask), mask_clean)
    
    defect_contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_defects = [c for c in defect_contours if cv2.contourArea(c) > 30]
    defect_area = sum(cv2.contourArea(c) for c in valid_defects)
    defect_ratio = min(round((defect_area / max(object_area, 1)) * 100, 2), 100.0)
    cv2.drawContours(annotated_img, valid_defects, -1, (0, 0, 255), 2)

    # 3. Texture, Shape & Color Abnormalities
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    laplacian_var = round(cv2.Laplacian(masked_gray, cv2.CV_64F).var(), 1)
    abnormal_texture = laplacian_var > 1050 or laplacian_var < 30
    texture_status = f"Abnormal ({laplacian_var})" if abnormal_texture else f"Normal ({laplacian_var})"

    abnormal_shape = solidity < 0.84 or circularity < 0.32
    shape_status = f"Abnormal / Deformed (Solidity: {solidity})" if abnormal_shape else f"Normal Shape (Solidity: {solidity})"

    valid_pixels = hsv[mask_clean > 0]
    mean_sat = np.mean(valid_pixels[:, 1]) if len(valid_pixels) > 0 else 0
    mean_val = np.mean(valid_pixels[:, 2]) if len(valid_pixels) > 0 else 0
    abnormal_color = (mean_sat < 50) or (mean_val < 60) or (defect_ratio > 14.0)
    color_status = f"Abnormal Discoloration (Sat: {int(mean_sat)})" if abnormal_color else f"Healthy Natural Color (Sat: {int(mean_sat)})"

    # 4. Hybrid Edibility Percentage vs Safe Benchmark
    cv_score = max(0.0, min(100.0, 100.0 - (defect_ratio * 2.5) - (12 if abnormal_texture else 0) - (10 if abnormal_shape else 0)))
    edibility_pct = round((0.65 * ai_fresh_prob) + (0.35 * cv_score), 1)
    diff_benchmark = round(edibility_pct - SAFE_EDIBLE_BENCHMARK, 1)

    if edibility_pct >= SAFE_EDIBLE_BENCHMARK:
        rot_status = "NOT ROTTEN (Fresh & Healthy)"
        verdict = "EDIBLE — Safe to Consume"
        severity = "success"
        advice = f"Your sample scored {edibility_pct}%, which is +{diff_benchmark}% above the {SAFE_EDIBLE_BENCHMARK}% safe edibility standard."
    elif 55.0 <= edibility_pct < SAFE_EDIBLE_BENCHMARK:
        rot_status = "EARLY BLEMISHES / PARTIAL SPOILAGE"
        verdict = "EDIBLE WITH CAUTION — Trim Bad Marks"
        severity = "warning"
        advice = f"Your sample scored {edibility_pct}% ({abs(diff_benchmark)}% below the {SAFE_EDIBLE_BENCHMARK}% safe standard). Trim bad spots before eating."
    else:
        rot_status = "ROTTEN / SPOILED"
        verdict = "NOT EDIBLE — DO NOT CONSUME"
        severity = "error"
        advice = f"Your sample scored {edibility_pct}% ({abs(diff_benchmark)}% below the {SAFE_EDIBLE_BENCHMARK}% safe standard). Unsafe to eat."

    return {
        "annotated_img": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
        "defect_mask": cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2RGB),
        "edibility_pct": edibility_pct,
        "diff_benchmark": diff_benchmark,
        "rot_status": rot_status,
        "verdict": verdict,
        "severity": severity,
        "advice": advice,
        "defect_count": len(valid_defects),
        "defect_ratio": defect_ratio,
        "abnormal_texture": abnormal_texture,
        "texture_status": texture_status,
        "abnormal_shape": abnormal_shape,
        "shape_status": shape_status,
        "abnormal_color": abnormal_color,
        "color_status": color_status
    }

# --- Sidebar ---
with st.sidebar:
    st.title("🌿 AgriScan-AI")
    st.caption("Vision-Language Produce & Edibility Diagnostics")
    st.divider()
    st.markdown(f"""
    **Edibility Standard Benchmarks:**
    - **Optimal Fresh Average:** `{OPTIMAL_FRESH_AVERAGE}%`
    - **Minimum Safe Edibility:** `{SAFE_EDIBLE_BENCHMARK}%`
    - **Spoilage / Discard Zone:** `< 55.0%`
    """)

# --- Main Interface ---
st.title("🌿 AgriScan-AI: Smart Produce & Edibility Inspector")
st.write("Select your produce category first, then upload or capture a photo for AI species recognition, spoilage detection, and edibility comparison.")

# --- STEP 1: Ask Fruit vs Vegetable BEFORE Upload/Camera ---
st.subheader("Step 1: What are you scanning today?")
category_choice = st.radio(
    "Select Category:",
    ["🍎 Fruit", "🥦 Vegetable"],
    horizontal=True
)
clean_category = "Fruit" if "Fruit" in category_choice else "Vegetable"

st.divider()

# --- STEP 2: Upload or Instant Camera Capture ---
st.subheader(f"Step 2: Provide Your {clean_category} Image")
input_method = st.radio("Choose Input Method:", ["📁 Upload Image", "📸 Instant Camera Capture"], horizontal=True)

image_file = None
if "Upload" in input_method:
    image_file = st.file_uploader(f"Upload a {clean_category.lower()} photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input(f"Take an instant photo of the {clean_category.lower()}")

if image_file is not None:
    pil_img = Image.open(image_file)
    
    with st.spinner(f"Analyzing {clean_category.lower()} species, surface defects, and edibility score..."):
        is_edible_food, detected_name, match_conf, ai_fresh_prob, top3_matches = classify_with_clip(pil_img, clean_category)

    # --- NON-EDIBLE OBJECT ERROR ANNOUNCEMENT ---
    if not is_edible_food:
        st.error(f"🚨 ERROR ANNOUNCEMENT: NOT EDIBLE! This image shows '{detected_name}' ({match_conf}% confidence), which is NOT an edible {clean_category.lower()}.")
        st.warning("Please upload or capture an actual fruit or vegetable.")
        st.image(pil_img, caption=f"Non-Edible Object Detected: {detected_name}", width=380)
    
    # --- PRODUCE DIAGNOSTIC RESULTS & BENCHMARK COMPARISON ---
    else:
        res = analyze_visual_defects(pil_img, ai_fresh_prob)

        if res["severity"] == "error":
            st.error(f"🚨 ERROR ANNOUNCEMENT: {res['verdict']} | {res['rot_status']} ({detected_name})")
        elif res["severity"] == "warning":
            st.warning(f"⚠️ CAUTION: {res['verdict']} | {res['rot_status']} ({detected_name})")
        else:
            st.success(f"✅ VERIFIED: {res['verdict']} | {res['rot_status']} ({detected_name})")

        st.info(f"**Edibility Comparison:** {res['advice']}")

        # Metrics Row with Benchmark Comparison
        st.subheader("Diagnostic & Edibility Benchmark Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Detected {clean_category}", detected_name, f"{match_conf}% AI match")
        c2.metric("Your Sample Edibility", f"{res['edibility_pct']}%", f"{res['diff_benchmark']}% vs Safe Avg")
        c3.metric("Safe Edible Average", f"{SAFE_EDIBLE_BENCHMARK}%", f"Optimal: {OPTIMAL_FRESH_AVERAGE}%")
        c4.metric("Rot & Defect Status", "ROTTEN" if res["severity"] == "error" else ("BLEMISHED" if res["severity"] == "warning" else "NOT ROTTEN"), f"{res['defect_ratio']}% spot area")

        # Visual Comparison Bars
        st.markdown("**Edibility Score Comparison (Your Photo vs. Safe Edible Standard):**")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.caption(f"Your Uploaded {detected_name}: **{res['edibility_pct']}%**")
            st.progress(int(res["edibility_pct"]))
        with comp_col2:
            st.caption(f"Average Safe Edible Standard: **{SAFE_EDIBLE_BENCHMARK}%** (Optimal: {OPTIMAL_FRESH_AVERAGE}%)")
            st.progress(int(SAFE_EDIBLE_BENCHMARK))

        st.divider()

        # Visual Inspection Images
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("**1. Original Photo**")
            st.image(pil_img, use_container_width=True)
        with v2:
            st.markdown("**2. Defect & Contour Detection**")
            st.image(res["annotated_img"], caption="Green: Box | Blue: Shape | Red: Bad Marks/Rot", use_container_width=True)
        with v3:
            st.markdown("**3. Spoilage & Mold Mask**")
            st.image(res["defect_mask"], caption="Isolated dark rot & mold regions", use_container_width=True)

        # Detailed Inspection Table
        st.subheader("Detailed Quality & Abnormality Report")
        st.table({
            "Inspection Parameter": [
                f"{clean_category} Identification (Top Matches)",
                "Color Assessment",
                "Surface Texture",
                "Shape & Morphology",
                "Bad Marks / Spot Coverage",
                "Edibility vs. Average Standard"
            ],
            "Condition": [
                detected_name,
                "ABNORMAL" if res["abnormal_color"] else "NORMAL",
                "ABNORMAL" if res["abnormal_texture"] else "NORMAL",
                "ABNORMAL" if res["abnormal_shape"] else "NORMAL",
                "HIGH" if res["defect_ratio"] >= 15 else ("MODERATE" if res["defect_ratio"] >= 5 else "LOW"),
                res["verdict"]
            ],
            "Measured Details": [
                ", ".join([f"{name} ({conf}%)" for name, conf in top3_matches]),
                res["color_status"],
                res["texture_status"],
                res["shape_status"],
                f"{res['defect_count']} marks covering {res['defect_ratio']}% of surface",
                f"Sample: {res['edibility_pct']}% | Safe Standard: {SAFE_EDIBLE_BENCHMARK}% | Optimal Avg: {OPTIMAL_FRESH_AVERAGE}%"
            ]
        })
else:
    st.info(f"👆 Select **Upload Image** or **Instant Camera Capture** above to scan your {clean_category.lower()}.")
