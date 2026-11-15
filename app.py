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
        background: rgba(255, 255, 255, 0.86);
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

def clarify_and_focus_produce(pil_image, category_choice):
    """
    Isolates the fruit/vegetable from cluttered backgrounds or multiple characters/people,
    crops tightly around the produce, blurs distracting background, and clarifies surface details.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    h_orig, w_orig, _ = img_rgb.shape

    # Step 1: Use CLIP grid-patch scoring to locate the produce region if scene has multiple characters/objects
    grid_size = 3
    best_score = -1.0
    best_box = (0, 0, w_orig, h_orig)

    # Check whole image + overlapping sub-crops (center & 3x3 regions) to lock onto the produce
    candidate_boxes = [(0, 0, w_orig, h_orig)]
    # Center 65% crop (handles handheld captures with people/characters around)
    cx1, cy1 = int(w_orig * 0.17), int(h_orig * 0.17)
    cx2, cy2 = int(w_orig * 0.83), int(h_orig * 0.83)
    candidate_boxes.append((cx1, cy1, cx2, cy2))

    step_x, step_y = w_orig // grid_size, h_orig // grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            x1 = max(0, c * step_x - int(step_x * 0.25))
            y1 = max(0, r * step_y - int(step_y * 0.25))
            x2 = min(w_orig, (c + 1) * step_x + int(step_x * 0.25))
            y2 = min(h_orig, (r + 1) * step_y + int(step_y * 0.25))
            if (x2 - x1) > 60 and (y2 - y1) > 60:
                candidate_boxes.append((x1, y1, x2, y2))

    crop_images = [Image.fromarray(img_rgb[y1:y2, x1:x2]) for (x1, y1, x2, y2) in candidate_boxes]
    target_text = [
        f"a close-up of a {category_choice.lower()} fruit or vegetable",
        "human face, person, hand, background wall, or non-food clutter"
    ]
    inputs = clip_processor(text=target_text, images=crop_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = clip_model(**inputs).logits_per_image
        probs = logits.softmax(dim=1)[:, 0].tolist()

    # Prefer focused sub-crop if it isolates the produce better than the full cluttered frame
    best_idx = int(np.argmax(probs[1:])) + 1 if max(probs[1:]) > (probs[0] - 0.05) else 0
    x1, y1, x2, y2 = candidate_boxes[best_idx]
    cropped_rgb = img_rgb[y1:y2, x1:x2]

    # Step 2: GrabCut / Salient Foreground Isolation to suppress any remaining background characters
    bgr_crop = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    bgr_std = cv2.resize(bgr_crop, (400, 400))

    mask = np.zeros((400, 400), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (25, 25, 350, 350)
    try:
        cv2.grabCut(bgr_std, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception:
        fg_mask = np.ones((400, 400), np.uint8)

    # Fallback if GrabCut removes too much
    if np.sum(fg_mask) < (400 * 400 * 0.12):
        hsv_tmp = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2HSV)
        fg_mask = (cv2.inRange(hsv_tmp, np.array([0, 28, 25]), np.array([180, 255, 255])) > 0).astype("uint8")

    # Smooth mask & blur background characters/clutter while keeping produce crisp
    fg_mask_255 = (fg_mask * 255).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_CLOSE, kernel)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_OPEN, kernel)

    blurred_bg = cv2.GaussianBlur(bgr_std, (35, 35), 0)
    # Darken background slightly so produce stands out clearly
    blurred_bg = cv2.convertScaleAbs(blurred_bg, alpha=0.65, beta=0)

    # Step 3: Image Clarification (CLAHE adaptive contrast + unsharp mask sharpening)
    lab = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_clarified = clahe.apply(l_chan)
    clarified_bgr = cv2.cvtColor(cv2.merge((l_clarified, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    # Gentle unsharp mask to sharpen surface texture & blemishes
    gaussian_blur = cv2.GaussianBlur(clarified_bgr, (0, 0), 2.0)
    clarified_bgr = cv2.addWeighted(clarified_bgr, 1.35, gaussian_blur, -0.35, 0)

    # Composite sharp foreground produce over blurred background
    fg_alpha = (fg_mask_255.astype(np.float32) / 255.0)[:, :, None]
    focused_bgr = (clarified_bgr.astype(np.float32) * fg_alpha + blurred_bg.astype(np.float32) * (1.0 - fg_alpha)).astype(np.uint8)

    focused_pil = Image.fromarray(cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2RGB))
    return focused_pil, focused_bgr, fg_mask_255

def classify_with_clip(focused_pil, category_choice, optional_name=""):
    # Step A: Check edible food vs non-food object
    domain_prompts = [
        f"a photo of a fresh or rotten {category_choice.lower()} fruit or vegetable food",
        "a photo of a non-food object, electronics, plastic, metal, wall, or person"
    ]
    inputs_domain = clip_processor(text=domain_prompts, images=focused_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_domain = clip_model(**inputs_domain).logits_per_image.softmax(dim=1)[0].tolist()
    
    if (probs_domain[0] * 100) < 42.0 and not optional_name.strip():
        nf_prompts = [f"a photo of {item}" for item in NON_FOOD_LIST]
        inputs_nf = clip_processor(text=nf_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_nf = clip_model(**inputs_nf).logits_per_image.softmax(dim=1)[0].tolist()
        best_idx = int(np.argmax(probs_nf))
        return False, NON_FOOD_LIST[best_idx].title(), round(probs_nf[best_idx] * 100, 1), 0.0, []

    # Step B: Use user-provided name if entered, otherwise auto-detect via CLIP
    clean_user_name = optional_name.strip().title()
    if clean_user_name:
        top_item = clean_user_name
        # Verify confidence of user-provided name vs general non-food
        verify_prompts = [
            f"a photo of a {clean_user_name.lower()}",
            "a photo of a non-food object, electronics, plastic, or person"
        ]
        inputs_v = clip_processor(text=verify_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_v = clip_model(**inputs_v).logits_per_image.softmax(dim=1)[0].tolist()
        top_conf = round(probs_v[0] * 100, 1)
        top3 = [(clean_user_name, top_conf)]
    else:
        target_list = FRUITS_LIST if category_choice == "Fruit" else VEGETABLES_LIST
        species_prompts = [f"a close-up photo of a {item.lower()}" for item in target_list]
        inputs_sp = clip_processor(text=species_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_sp = clip_model(**inputs_sp).logits_per_image.softmax(dim=1)[0].tolist()
        
        ranked_indices = np.argsort(probs_sp)[::-1]
        top_item = target_list[ranked_indices[0]]
        top_conf = round(probs_sp[ranked_indices[0]] * 100, 1)
        top3 = [(target_list[i], round(probs_sp[i] * 100, 1)) for i in ranked_indices[:3]]

    # Step C: Fresh vs Rotten Semantic Assessment on clarified produce
    state_prompts = [
        f"a fresh, healthy, clean, ripe {top_item.lower()} safe to eat",
        f"a rotten, moldy, spoiled, bruised, decayed {top_item.lower()} unsafe to eat"
    ]
    inputs_st = clip_processor(text=state_prompts, images=focused_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_st = clip_model(**inputs_st).logits_per_image.softmax(dim=1)[0].tolist()
    
    ai_fresh_prob = round(probs_st[0] * 100, 1)
    return True, top_item, top_conf, ai_fresh_prob, top3

def analyze_visual_defects(focused_bgr, mask_clean, ai_fresh_prob):
    hsv = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = focused_bgr.copy()
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

    # Bad Marks, Dark Rot & Mold Detection
    rot_dark = cv2.inRange(hsv, np.array([0, 25, 10]), np.array([28, 255, 100]))
    mold_mask = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([180, 38, 225]))
    defect_mask = cv2.bitwise_and(cv2.bitwise_or(rot_dark, mold_mask), mask_clean)
    
    defect_contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_defects = [c for c in defect_contours if cv2.contourArea(c) > 30]
    defect_area = sum(cv2.contourArea(c) for c in valid_defects)
    defect_ratio = min(round((defect_area / max(object_area, 1)) * 100, 2), 100.0)
    cv2.drawContours(annotated_img, valid_defects, -1, (0, 0, 255), 2)

    # Texture, Shape & Color Abnormalities
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    laplacian_var = round(cv2.Laplacian(masked_gray, cv2.CV_64F).var(), 1)
    abnormal_texture = laplacian_var > 1250 or laplacian_var < 30
    texture_status = f"Abnormal ({laplacian_var})" if abnormal_texture else f"Normal ({laplacian_var})"

    abnormal_shape = solidity < 0.84 or circularity < 0.32
    shape_status = f"Abnormal / Deformed (Solidity: {solidity})" if abnormal_shape else f"Normal Shape (Solidity: {solidity})"

    valid_pixels = hsv[mask_clean > 0]
    mean_sat = np.mean(valid_pixels[:, 1]) if len(valid_pixels) > 0 else 0
    mean_val = np.mean(valid_pixels[:, 2]) if len(valid_pixels) > 0 else 0
    abnormal_color = (mean_sat < 50) or (mean_val < 60) or (defect_ratio > 14.0)
    color_status = f"Abnormal Discoloration (Sat: {int(mean_sat)})" if abnormal_color else f"Healthy Natural Color (Sat: {int(mean_sat)})"

    # Hybrid Edibility Percentage vs Safe Benchmark
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
        edibility_verdict = "EDIBLE WITH CAUTION — Trim Bad Marks"
        verdict = edibility_verdict
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
    st.caption("Smart Subject-Focused Produce & Edibility Diagnostics")
    st.divider()
    st.markdown(f"""
    **Edibility Standard Benchmarks:**
    - **Optimal Fresh Average:** `{OPTIMAL_FRESH_AVERAGE}%`
    - **Minimum Safe Edibility:** `{SAFE_EDIBLE_BENCHMARK}%`
    - **Spoilage / Discard Zone:** `< 55.0%`
    """)
    st.divider()
    st.markdown("""
    **Smart Focus Pipeline:**
    - Automatically isolates the fruit/vegetable from crowded scenes or people.
    - Clarifies surface texture & lighting (CLAHE + Sharpening) before AI diagnosis.
    """)

# --- Main Interface ---
st.title("🌿 AgriScan-AI: Smart Produce & Edibility Inspector")
st.write("Select your produce category, optionally enter its name, and upload or capture a photo—even in crowded scenes, the AI will isolate and clarify the produce automatically.")

# --- STEP 1: Category + Optional Produce Name ---
st.subheader("Step 1: Produce Details")
col_cat, col_name = st.columns([1, 1.5])
with col_cat:
    category_choice = st.radio(
        "Is it a Fruit or Vegetable?",
        ["🍎 Fruit", "🥦 Vegetable"],
        horizontal=True
    )
clean_category = "Fruit" if "Fruit" in category_choice else "Vegetable"

with col_name:
    optional_produce_name = st.text_input(
        f"What is the {clean_category.lower()}'s name? (Optional — leave empty for AI auto-detection)",
        placeholder="e.g., Guava, Mango, Tomato, Cucumber..."
    )

st.divider()

# --- STEP 2: Upload or Instant Camera Capture ---
st.subheader(f"Step 2: Capture or Upload Your {clean_category}")
input_method = st.radio("Choose Input Method:", ["📁 Upload Image", "📸 Instant Camera Capture"], horizontal=True)

image_file = None
if "Upload" in input_method:
    image_file = st.file_uploader(f"Upload a {clean_category.lower()} photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input(f"Take an instant photo of the {clean_category.lower()}")

if image_file is not None:
    pil_img = Image.open(image_file)
    
    with st.spinner("Isolating produce from background, clarifying image, and running AI diagnostics..."):
        focused_pil, focused_bgr, fg_mask = clarify_and_focus_produce(pil_img, clean_category)
        is_edible_food, detected_name, match_conf, ai_fresh_prob, top3_matches = classify_with_clip(
            focused_pil, clean_category, optional_produce_name
        )

    # --- NON-EDIBLE OBJECT ERROR ANNOUNCEMENT ---
    if not is_edible_food:
        st.error(f"🚨 ERROR ANNOUNCEMENT: NOT EDIBLE! This image shows '{detected_name}' ({match_conf}% confidence), which is NOT an edible {clean_category.lower()}.")
        st.warning("Please upload or capture an actual fruit or vegetable.")
        st.image(pil_img, caption=f"Non-Edible Object Detected: {detected_name}", width=380)
    
    # --- PRODUCE DIAGNOSTIC RESULTS & BENCHMARK COMPARISON ---
    else:
        res = analyze_visual_defects(focused_bgr, fg_mask, ai_fresh_prob)

        if res["severity"] == "error":
            st.error(f"🚨 ERROR ANNOUNCEMENT: {res['verdict']} | {res['rot_status']} ({detected_name})")
        elif res["severity"] == "warning":
            st.warning(f"⚠️ CAUTION: {res['verdict']} | {res['rot_status']} ({detected_name})")
        else:
            st.success(f"✅ VERIFIED: {res['verdict']} | {res['rot_status']} ({detected_name})")

        st.info(f"**Edibility Comparison:** {res['advice']}")

        # Metrics Row
        st.subheader("Diagnostic & Edibility Benchmark Summary")
        c1, c2, c3, c4 = st.columns(4)
        name_source_label = "User Specified" if optional_produce_name.strip() else f"{match_conf}% AI match"
        c1.metric(f"Target {clean_category}", detected_name, name_source_label)
        c2.metric("Your Sample Edibility", f"{res['edibility_pct']}%", f"{res['diff_benchmark']}% vs Safe Avg")
        c3.metric("Safe Edible Average", f"{SAFE_EDIBLE_BENCHMARK}%", f"Optimal: {OPTIMAL_FRESH_AVERAGE}%")
        c4.metric("Rot & Defect Status", "ROTTEN" if res["severity"] == "error" else ("BLEMISHED" if res["severity"] == "warning" else "NOT ROTTEN"), f"{res['defect_ratio']}% spot area")

        # Visual Comparison Bars
        st.markdown("**Edibility Score Comparison (Your Photo vs. Safe Edible Standard):**")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.caption(f"Your Clarified {detected_name}: **{res['edibility_pct']}%**")
            st.progress(int(res["edibility_pct"]))
        with comp_col2:
            st.caption(f"Average Safe Edible Standard: **{SAFE_EDIBLE_BENCHMARK}%** (Optimal: {OPTIMAL_FRESH_AVERAGE}%)")
            st.progress(int(SAFE_EDIBLE_BENCHMARK))

        st.divider()

        # Visual Inspection Images (Shows Original vs Clarified Subject-Focused Image)
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            st.markdown("**1. Original Capture**")
            st.image(pil_img, use_container_width=True)
        with v2:
            st.markdown("**2. Focused & Clarified**")
            st.image(focused_pil, caption="Background isolated & details clarified", use_container_width=True)
        with v3:
            st.markdown("**3. Defect & Shape Scan**")
            st.image(res["annotated_img"], caption="Green: Box | Blue: Shape | Red: Spots", use_container_width=True)
        with v4:
            st.markdown("**4. Spoilage & Mold Mask**")
            st.image(res["defect_mask"], caption="Isolated dark rot & mold regions", use_container_width=True)

        # Detailed Inspection Table
        st.subheader("Detailed Quality & Abnormality Report")
        st.table({
            "Inspection Parameter": [
                f"{clean_category} Identification",
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
                f"{detected_name} ({name_source_label})" if optional_produce_name.strip() else ", ".join([f"{n} ({c}%)" for n, c in top3_matches]),
                res["color_status"],
                res["texture_status"],
                res["shape_status"],
                f"{res['defect_count']} marks covering {res['defect_ratio']}% of surface",
                f"Sample: {res['edibility_pct']}% | Safe Standard: {SAFE_EDIBLE_BENCHMARK}% | Optimal Avg: {OPTIMAL_FRESH_AVERAGE}%"
            ]
        })
else:
    st.info(f"👆 Choose **Upload Image** or **Instant Camera Capture** above to scan your {clean_category.lower()}.")
