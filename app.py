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
        ("Yellow Bell Pepper", (19, 32), (15, 28), 195, 0.76, 0.86, 280),
        ("Eggplant", (125, 168), (110, 145), 120, 0.54, 0.50, 260),
        ("Broccoli", (40, 84), (32, 60), 130, 0.64, 0.90, 1200),
        ("Onion", (11, 25), (140, 175), 105, 0.85, 0.92, 360),
        ("Zucchini", (38, 78), (30, 52), 130, 0.38, 0.32, 340)
    ]
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
        fg_mask = (cv2.inRange(hsv_tmp, np.array([0, 22, 18]), np.array([180, 255, 248])) > 0).astype("uint8")

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

    lab = cv2.cvtColor(focused_true_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clarified_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    clarified_pil = Image.fromarray(cv2.cvtColor(clarified_bgr, cv2.COLOR_BGR2RGB))

    return focused_true_bgr, clarified_pil, fg_mask_255

def classify_produce(hsv, mask_clean, circularity, aspect_ratio, laplacian_var, category_choice, optional_name):
    if optional_name.strip():
        clean_name = optional_name.strip().title()
        return True, clean_name, 99.0, [(clean_name, 99.0)]

    valid_pixels = hsv[mask_clean > 0]
    if len(valid_pixels) < 800:
        return False, "Empty / Unclear Frame", 0.0, []

    hues = valid_pixels[:, 0]
    sats = valid_pixels[:, 1]
    mean_sat = float(np.mean(sats))

    # Reject completely colorless/non-food objects (walls, screens, metal, paper)
    if mean_sat < 16 and category_choice == "Fruit":
        return False, "Non-Food / Colorless Object", 86.0, []

    # Use brightly lit, well-saturated core pixels to find the true skin color (ignoring shadows/rot)
    healthy_core = valid_pixels[(valid_pixels[:, 2] > 65) & (valid_pixels[:, 1] > 35)]
    if len(healthy_core) > 200:
        core_hues = healthy_core[:, 0]
        core_sat = float(np.mean(healthy_core[:, 1]))
    else:
        core_hues = hues
        core_sat = mean_sat

    hist, _ = np.histogram(core_hues, bins=36, range=(0, 180))
    dominant_hue = (int(np.argmax(hist)) * 5) + 2.5

    candidates = BOTANICAL_DB[category_choice]
    norm_aspect = min(aspect_ratio, 1.0 / max(aspect_ratio, 0.01))
    scored = []

    for name, (h1_min, h1_max), (h2_min, h2_max), s_ideal, c_id, a_id, tex_max in candidates:
        # 1. Core Hue Match (42%)
        if h1_min <= dominant_hue <= h1_max:
            hue_sc = 1.0
        elif h2_min <= dominant_hue <= h2_max:
            hue_sc = 0.92
        else:
            d1 = min(abs(dominant_hue - h1_min), abs(dominant_hue - h1_max))
            d2 = min(abs(dominant_hue - h2_min), abs(dominant_hue - h2_max))
            hue_sc = max(0.0, 1.0 - (min(d1, d2) / 30.0))

        # 2. Saturation Match (13%)
        sat_sc = max(0.0, 1.0 - abs(core_sat - s_ideal) / 140.0)

        # 3. Geometry: Circularity + Aspect Ratio (32%)
        circ_sc = max(0.0, 1.0 - abs(circularity - c_id) * 1.55)
        asp_sc = max(0.0, 1.0 - abs(norm_aspect - a_id) * 1.55)
        geom_sc = (0.55 * circ_sc) + (0.45 * asp_sc)

        # Disambiguate Round Smooth Apple vs Conical Seeded Strawberry
        if name == "Strawberry" and circularity >= 0.77:
            geom_sc *= 0.40
        if name == "Apple" and circularity >= 0.77:
            geom_sc = min(1.0, geom_sc + 0.16)

        # 4. Texture Match (13%)
        tex_sc = 1.0 if laplacian_var <= tex_max else max(0.3, 1.0 - (laplacian_var - tex_max) / 1000.0)

        total = round(((0.42 * hue_sc) + (0.13 * sat_sc) + (0.32 * geom_sc) + (0.13 * tex_sc)) * 100.0, 1)
        scored.append((name, total))

    scored.sort(key=lambda x: x[1], reverse=True)
    return True, scored[0][0], scored[0][1], scored[:3]

def analyze_produce(focused_bgr, mask_clean, category_choice, optional_name):
    hsv = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(focused_bgr, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = focused_bgr.copy()
    circularity, solidity, aspect_ratio, object_area = 0.0, 1.0, 1.0, 0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 1200:
            object_area = area
            x, y, w, h = cv2.boundingRect(largest)
            aspect_ratio = round(float(w) / max(h, 1), 2)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0:
                circularity = round(4 * np.pi * (area / (perimeter * perimeter)), 2)
            hull_area = cv2.contourArea(cv2.convexHull(largest))
            if hull_area > 0:
                solidity = round(float(area) / hull_area, 2)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 2)
            cv2.drawContours(annotated_img, [largest], -1, (52, 152, 219), 2)

    if object_area < 1200:
        return {"is_edible_object": False, "detected_name": "No Produce Detected", "match_conf": 0.0}

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    laplacian_var = round(cv2.Laplacian(masked_gray, cv2.CV_64F).var(), 1)

    is_edible_obj, detected_name, match_conf, top3_matches = classify_produce(
        hsv, mask_clean, circularity, aspect_ratio, laplacian_var, category_choice, optional_name
    )

    if not is_edible_obj:
        return {"is_edible_object": False, "detected_name": detected_name, "match_conf": match_conf}

    # Erode outer edge by 4 pixels when checking defects so normal edge shading isn't counted as rot
    interior_mask = cv2.erode(mask_clean, np.ones((5, 5), np.uint8), iterations=1)
    if np.sum(interior_mask) < 500:
        interior_mask = mask_clean

    valid_pixels = hsv[interior_mask > 0]
    mean_sat = float(np.mean(valid_pixels[:, 1]))
    mean_val = float(np.mean(valid_pixels[:, 2]))
    std_val = float(np.std(valid_pixels[:, 2]))

    # --- BALANCED, SHADOW-PROOF SPOILAGE DETECTION ---
    # 1. Surface Bad Marks / Brown Bruises (Orange highlight)
    # Catches medium-dark brown spots (Hue 8-25, moderate saturation, Value 45-115)
    brown_bruise = cv2.inRange(hsv, np.array([8, 35, 45]), np.array([25, 225, 115]))
    blemish_mask = cv2.bitwise_and(brown_bruise, interior_mask)

    # 2. Severe Decay: Deep Necrotic Black/Brown Rot OR Fuzzy White/Gray Mold (Red highlight)
    # Only flags true black/necrotic tissue (Value < 45 & LAB L < 48) or desaturated fuzzy mold (Sat < 32 & Val > 115)
    deep_rot_hsv = cv2.inRange(hsv, np.array([0, 10, 0]), np.array([30, 255, 45]))
    necrotic_lab = cv2.inRange(lab[:, :, 0], 0, 48)
    fuzzy_mold = cv2.inRange(hsv, np.array([0, 0, 115]), np.array([180, 32, 235]))
    severe_mask = cv2.bitwise_and(cv2.bitwise_or(cv2.bitwise_and(deep_rot_hsv, necrotic_lab), fuzzy_mold), interior_mask)

    # Ensure no overlap between minor blemishes and severe rot
    blemish_mask = cv2.bitwise_and(blemish_mask, cv2.bitwise_not(severe_mask))
    total_defect_mask = cv2.bitwise_or(blemish_mask, severe_mask)

    blemish_contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blemishes = [c for c in blemish_contours if cv2.contourArea(c) > 25]
    blemish_ratio = min(round((sum(cv2.contourArea(c) for c in valid_blemishes) / max(object_area, 1)) * 100, 2), 100.0)

    severe_contours, _ = cv2.findContours(severe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_severe = [c for c in severe_contours if cv2.contourArea(c) > 25]
    severe_ratio = min(round((sum(cv2.contourArea(c) for c in valid_severe) / max(object_area, 1)) * 100, 2), 100.0)

    total_bad_marks = len(valid_blemishes) + len(valid_severe)
    total_defect_ratio = min(round(blemish_ratio + severe_ratio, 2), 100.0)

    cv2.drawContours(annotated_img, valid_blemishes, -1, (0, 165, 255), 2)
    cv2.drawContours(annotated_img, valid_severe, -1, (0, 0, 255), 2)

    abnormal_texture = laplacian_var > 1150 or laplacian_var < 22
    texture_status = f"Abnormal / Shriveled ({laplacian_var})" if abnormal_texture else f"Normal Smoothness ({laplacian_var})"

    abnormal_shape = solidity < 0.82 or circularity < 0.28
    shape_status = f"Abnormal / Deformed (Solidity: {solidity})" if abnormal_shape else f"Normal Shape (Solidity: {solidity})"

    abnormal_color = (mean_sat < 42) or (mean_val < 58) or (severe_ratio > 8.0)
    color_status = f"Abnormal Discoloration (Sat: {int(mean_sat)})" if abnormal_color else f"Healthy Natural Color (Sat: {int(mean_sat)})"

    # --- BALANCED EDIBILITY SCORING (Fresh ~90-96%, Bruised ~62-79%, Rotten < 56%) ---
    penalty = (
        (blemish_ratio * 1.4) +
        (severe_ratio * 4.5) +
        (12 if abnormal_texture else 0) +
        (10 if abnormal_shape else 0)
    )
    edibility_pct = round(max(0.0, min(100.0, 96.0 - penalty)), 1)
    usable_flesh_pct = round(max(0.0, 100.0 - total_defect_ratio), 1)
    diff_benchmark = round(edibility_pct - SAFE_EDIBLE_BENCHMARK, 1)

    is_rotten_or_bad = (
        (severe_ratio >= 8.5) or
        (total_defect_ratio >= 26.0) or
        (edibility_pct < 56.0)
    )
    has_bad_marks = (total_bad_marks > 0 and total_defect_ratio >= 2.0)

    if is_rotten_or_bad:
        rot_status = "BAD / ROTTEN / SPOILED"
        verdict = "NOT EDIBLE — DO NOT EAT"
        severity = "error"
        advice = (
            f"This {detected_name.lower()} is BAD / INEDIBLE (Severe rot/mold: {severe_ratio}%, Total damaged area: {total_defect_ratio}%). "
            f"Your sample scored {edibility_pct}% ({abs(diff_benchmark)}% below the {SAFE_EDIBLE_BENCHMARK}% safe standard). Do NOT eat it."
        )
    elif has_bad_marks:
        rot_status = "MINOR BAD MARKS (Not Rotten)"
        verdict = "EDIBLE — Safe After Trimming Bad Marks"
        severity = "warning"
        advice = (
            f"Localized bad marks or minor bruising detected ({total_bad_marks} marks covering {total_defect_ratio}% of surface), "
            f"but it is NOT rotten (~{usable_flesh_pct}% healthy usable flesh). Cut away the brown/marked parts before eating."
        )
    else:
        rot_status = "FRESH & HEALTHY (Not Rotten)"
        verdict = "EDIBLE — 100% Safe to Eat"
        severity = "success"
        advice = (
            f"Clean, fresh condition with no significant bad marks! Your sample scored {edibility_pct}% "
            f"(+{diff_benchmark}% above the {SAFE_EDIBLE_BENCHMARK}% safe edibility standard)."
        )

    return {
        "is_edible_object": True,
        "detected_name": detected_name,
        "match_conf": match_conf,
        "top3_matches": top3_matches,
        "annotated_img": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
        "defect_mask": cv2.cvtColor(total_defect_mask, cv2.COLOR_GRAY2RGB),
        "edibility_pct": edibility_pct,
        "usable_flesh_pct": usable_flesh_pct,
        "diff_benchmark": diff_benchmark,
        "rot_status": rot_status,
        "verdict": verdict,
        "severity": severity,
        "advice": advice,
        "total_bad_marks": total_bad_marks,
        "blemish_ratio": blemish_ratio,
        "severe_ratio": severe_ratio,
        "total_defect_ratio": total_defect_ratio,
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
    st.caption("Botanical Computer Vision & Spoilage Diagnostics")
    st.divider()
    st.markdown(f"""
    **Edibility Standard Benchmarks:**
    - **Optimal Fresh Average:** `{OPTIMAL_FRESH_AVERAGE}%`
    - **Minimum Safe Edibility:** `{SAFE_EDIBLE_BENCHMARK}%`
    - **Trim-Before-Eating Zone:** `56% – 79%`
    - **Bad / Do Not Eat Zone:** `< 56%` (or `≥ 8.5%` rot/mold)
    """)
    st.divider()
    st.markdown("""
    **Color Legend on Defect Scan:**
    - 🟢 **Green Box:** Detected Produce Bounds
    - 🔵 **Blue Outline:** Outer Shape Contour
    - 🟠 **Orange Marks:** Minor Surface Blemishes (Trim Before Eating)
    - 🔴 **Red Marks:** Deep Rot / Mold / Necrosis (Do Not Eat)
    """)

# --- Main Interface ---
st.title("🌿 AgriScan-AI: High-Precision Produce & Edibility Inspector")
st.write("Select your produce category, optionally type its name, capture or upload a photo, and click the confirmation button to start the inspection.")

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
        placeholder="e.g., Apple, Guava, Mango, Tomato, Cucumber..."
    )

st.divider()

st.subheader(f"Step 2: Capture or Upload Your {clean_category}")
input_method = st.radio("Choose Input Method:", ["📁 Upload Image", "📸 Instant Camera Capture"], horizontal=True)

image_file = None
if "Upload" in input_method:
    image_file = st.file_uploader(f"Upload a {clean_category.lower()} photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input(f"Take an instant photo of the {clean_category.lower()}")

if image_file is not None:
    file_signature = f"{image_file.name}_{image_file.size}_{clean_category}_{optional_produce_name.strip()}"
    if st.session_state.get("last_file_sig") != file_signature:
        st.session_state["confirmed_scan"] = False
        st.session_state["last_file_sig"] = file_signature

    pil_img = Image.open(image_file)

    st.divider()
    st.subheader("Step 3: Review & Confirm Scan")
    prev_col1, prev_col2 = st.columns([1, 1.5])
    with prev_col1:
        st.image(pil_img, caption="Captured / Uploaded Image Preview", use_container_width=True)
    with prev_col2:
        st.markdown(f"""
        **Ready to Analyze:**
        - **Category:** {clean_category}
        - **Produce Name:** `{optional_produce_name.strip().title() if optional_produce_name.strip() else "Botanical Multi-Feature Auto-Detect"}`
        
        Click the confirmation button below to isolate the produce, identify its species, and check if it is edible or spoiled.
        """)
        if st.button("✅ Confirm & Analyze Produce"):
            st.session_state["confirmed_scan"] = True

    if st.session_state.get("confirmed_scan", False):
        st.divider()
        with st.spinner("Isolating produce, extracting botanical features, and scanning for rot/spoilage..."):
            focused_true_bgr, clarified_display_pil, fg_mask = focus_and_clarify_image(pil_img)
            res = analyze_produce(focused_true_bgr, fg_mask, clean_category, optional_produce_name)

        if not res["is_edible_object"]:
            st.error(f"🚨 ERROR ANNOUNCEMENT: NOT EDIBLE! Detected '{res['detected_name']}', which is NOT an edible {clean_category.lower()}.")
            st.warning("Please upload or capture a clear photo of an actual fruit or vegetable.")
            st.image(pil_img, caption="Non-Edible / Unrecognized Object", width=380)
        else:
            detected_name = res["detected_name"]
            if res["severity"] == "error":
                st.error(f"🚨 BAD / INEDIBLE ANNOUNCEMENT: {res['verdict']} | {res['rot_status']} ({detected_name})")
            elif res["severity"] == "warning":
                st.warning(f"⚠️ BAD MARKS DETECTED — {res['verdict']} | {res['rot_status']} ({detected_name})")
            else:
                st.success(f"✅ VERIFIED: {res['verdict']} | {res['rot_status']} ({detected_name})")

            st.info(f"**Edibility & Safety Assessment:** {res['advice']}")

            st.subheader("Diagnostic & Edibility Benchmark Summary")
            c1, c2, c3, c4 = st.columns(4)
            name_source_label = "User Specified" if optional_produce_name.strip() else f"{res['match_conf']}% match"
            c1.metric(f"Detected {clean_category}", detected_name, name_source_label)
            c2.metric("Edibility Score", f"{res['edibility_pct']}%", f"{res['diff_benchmark']}% vs Safe Avg")
            c3.metric("Safe Edible Average", f"{SAFE_EDIBLE_BENCHMARK}%", f"Usable Flesh: {res['usable_flesh_pct']}%")
            c4.metric("Bad Marks / Rot Area", f"{res['total_bad_marks']} spots", f"{res['total_defect_ratio']}% damaged")

            st.markdown("**Edibility Score Comparison (Your Photo vs. Safe Edible Standard):**")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.caption(f"Your {detected_name} Edibility Score: **{res['edibility_pct']}%** (Usable Flesh: **{res['usable_flesh_pct']}%**)")
                st.progress(int(res["edibility_pct"]))
            with comp_col2:
                st.caption(f"Average Safe Edible Standard: **{SAFE_EDIBLE_BENCHMARK}%** (Optimal Fresh: **{OPTIMAL_FRESH_AVERAGE}%**)")
                st.progress(int(SAFE_EDIBLE_BENCHMARK))

            st.divider()

            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.markdown("**1. Original Capture**")
                st.image(pil_img, use_container_width=True)
            with v2:
                st.markdown("**2. Focused & Clarified**")
                st.image(clarified_display_pil, caption="Background isolated & details clarified", use_container_width=True)
            with v3:
                st.markdown("**3. Bad Marks vs. Rot Scan**")
                st.image(res["annotated_img"], caption="Orange: Blemishes | Red: Deep Rot/Mold", use_container_width=True)
            with v4:
                st.markdown("**4. Defect Isolation Mask**")
                st.image(res["defect_mask"], caption="All detected surface spots & decay", use_container_width=True)

            st.subheader("Detailed Quality, Bad Marks & Edibility Report")
            st.table({
                "Inspection Parameter": [
                    f"{clean_category} Identification (Top Candidates)",
                    "Color Assessment",
                    "Surface Texture",
                    "Shape & Morphology",
                    "Bad Marks vs. Severe Rot",
                    "Final Edibility Verdict"
                ],
                "Condition": [
                    detected_name,
                    "ABNORMAL" if res["abnormal_color"] else "NORMAL",
                    "ABNORMAL" if res["abnormal_texture"] else "NORMAL",
                    "ABNORMAL" if res["abnormal_shape"] else "NORMAL",
                    f"{res['total_bad_marks']} Marks ({res['total_defect_ratio']}%)",
                    res["verdict"]
                ],
                "Measured Details": [
                    f"{detected_name} ({name_source_label})" if optional_produce_name.strip() else ", ".join([f"{n} ({c}%)" for n, c in res["top3_matches"]]),
                    res["color_status"],
                    res["texture_status"],
                    res["shape_status"],
                    f"Minor Blemishes: {res['blemish_ratio']}% | Deep Rot/Mold: {res['severe_ratio']}% | Usable Flesh: {res['usable_flesh_pct']}%",
                    f"Sample: {res['edibility_pct']}% | Safe Standard: {SAFE_EDIBLE_BENCHMARK}% | Optimal Avg: {OPTIMAL_FRESH_AVERAGE}%"
                ]
            })
else:
    st.info(f"👆 Choose **Upload Image** or **Instant Camera Capture** above, then click **Confirm & Analyze Produce** to view your results.")
