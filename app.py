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

# --- Load Fast & Memory-Safe CLIP Model ---
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
                candidate_boxes.append((x1, y1, x2, y2))

    crop_images = [Image.fromarray(img_rgb[y1:y2, x1:x2]) for (x1, y1, x2, y2) in candidate_boxes]
    target_text = [
        f"a close-up photo of a {category_choice.lower()} fruit or vegetable",
        "human face, person, hand, background wall, or non-food clutter"
    ]
    inputs = clip_processor(text=target_text, images=crop_images, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = clip_model(**inputs).logits_per_image.softmax(dim=1)[:, 0].tolist()

    best_idx = int(np.argmax(probs[1:])) + 1 if max(probs[1:]) > (probs[0] - 0.04) else 0
    x1, y1, x2, y2 = candidate_boxes[best_idx]
    cropped_rgb = img_rgb[y1:y2, x1:x2]

    bgr_crop = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)
    bgr_std = cv2.resize(bgr_crop, (400, 400))

    mask = np.zeros((400, 400), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (22, 22, 356, 356)
    try:
        cv2.grabCut(bgr_std, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    except Exception:
        fg_mask = np.ones((400, 400), np.uint8)

    if np.sum(fg_mask) < (400 * 400 * 0.12):
        hsv_tmp = cv2.cvtColor(bgr_std, cv2.COLOR_BGR2HSV)
        fg_mask = (cv2.inRange(hsv_tmp, np.array([0, 25, 25]), np.array([180, 255, 255])) > 0).astype("uint8")

    fg_mask_255 = (fg_mask * 255).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_CLOSE, kernel)
    fg_mask_255 = cv2.morphologyEx(fg_mask_255, cv2.MORPH_OPEN, kernel)

    blurred_bg = cv2.GaussianBlur(bgr_std, (35, 35), 0)
    blurred_bg = cv2.convertScaleAbs(blurred_bg, alpha=0.65, beta=0)

    gaussian_blur = cv2.GaussianBlur(bgr_std, (0, 0), 1.8)
    sharpened_true_bgr = cv2.addWeighted(bgr_std, 1.25, gaussian_blur, -0.25, 0)

    fg_alpha = (fg_mask_255.astype(np.float32) / 255.0)[:, :, None]
    focused_true_bgr = (sharpened_true_bgr.astype(np.float32) * fg_alpha + blurred_bg.astype(np.float32) * (1.0 - fg_alpha)).astype(np.uint8)
    focused_true_pil = Image.fromarray(cv2.cvtColor(focused_true_bgr, cv2.COLOR_BGR2RGB))

    lab = cv2.cvtColor(focused_true_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clarified_display_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    clarified_display_pil = Image.fromarray(cv2.cvtColor(clarified_display_bgr, cv2.COLOR_BGR2RGB))

    return focused_true_pil, clarified_display_pil, focused_true_bgr, fg_mask_255

def classify_and_assess_edibility(focused_pil, category_choice, optional_name=""):
    domain_prompts = [
        f"a photo of a {category_choice.lower()} fruit or vegetable food, either fresh or with spots or rotten",
        "a photo of a non-food object, electronics, plastic, metal, wall, or person"
    ]
    inputs_domain = clip_processor(text=domain_prompts, images=focused_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_domain = clip_model(**inputs_domain).logits_per_image.softmax(dim=1)[0].tolist()
    
    if (probs_domain[0] * 100) < 40.0 and not optional_name.strip():
        nf_prompts = [f"a photo of {item}" for item in NON_FOOD_LIST]
        inputs_nf = clip_processor(text=nf_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_nf = clip_model(**inputs_nf).logits_per_image.softmax(dim=1)[0].tolist()
        best_idx = int(np.argmax(probs_nf))
        return False, NON_FOOD_LIST[best_idx].title(), round(probs_nf[best_idx] * 100, 1), {}, []

    clean_user_name = optional_name.strip().title()
    if clean_user_name:
        top_item = clean_user_name
        verify_prompts = [
            f"a photo of a {clean_user_name.lower()}",
            "a photo of a non-food object, plastic, or person"
        ]
        inputs_v = clip_processor(text=verify_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_v = clip_model(**inputs_v).logits_per_image.softmax(dim=1)[0].tolist()
        top_conf = round(probs_v[0] * 100, 1)
        top3 = [(clean_user_name, top_conf)]
    else:
        target_list = FRUITS_LIST if category_choice == "Fruit" else VEGETABLES_LIST
        species_prompts = [f"a clear photo of a {item.lower()} {category_choice.lower()}" for item in target_list]
        inputs_sp = clip_processor(text=species_prompts, images=focused_pil, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs_sp = clip_model(**inputs_sp).logits_per_image.softmax(dim=1)[0].tolist()
        
        ranked_indices = np.argsort(probs_sp)[::-1]
        top_item = target_list[ranked_indices[0]]
        top_conf = round(probs_sp[ranked_indices[0]] * 100, 1)
        top3 = [(target_list[i], round(probs_sp[i] * 100, 1)) for i in ranked_indices[:3]]

    state_prompts = [
        f"a completely fresh, spotless, clean, healthy ripe {top_item.lower()} ready to eat",
        f"a {top_item.lower()} with minor brown spots, small bruises, or surface blemishes that is still edible if trimmed",
        f"a heavily rotten, moldy, decomposing, slimy, or spoiled {top_item.lower()} that is unsafe and not edible"
    ]
    inputs_st = clip_processor(text=state_prompts, images=focused_pil, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs_st = clip_model(**inputs_st).logits_per_image.softmax(dim=1)[0].tolist()

    ai_states = {
        "clean_fresh_prob": round(probs_st[0] * 100, 1),
        "blemished_edible_prob": round(probs_st[1] * 100, 1),
        "rotten_inedible_prob": round(probs_st[2] * 100, 1)
    }
    return True, top_item, top_conf, ai_states, top3

def analyze_defects_and_edibility(focused_bgr, mask_clean, ai_states):
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

    blemish_mask = cv2.inRange(hsv, np.array([8, 35, 35]), np.array([26, 220, 115]))
    blemish_mask = cv2.bitwise_and(blemish_mask, mask_clean)

    deep_rot_mask = cv2.inRange(hsv, np.array([0, 20, 5]), np.array([25, 255, 45]))
    mold_mask = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 35, 230]))
    severe_spoilage_mask = cv2.bitwise_and(cv2.bitwise_or(deep_rot_mask, mold_mask), mask_clean)

    total_defect_mask = cv2.bitwise_or(blemish_mask, severe_spoilage_mask)

    blemish_contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blemishes = [c for c in blemish_contours if cv2.contourArea(c) > 25]
    blemish_area = sum(cv2.contourArea(c) for c in valid_blemishes)
    blemish_ratio = min(round((blemish_area / max(object_area, 1)) * 100, 2), 100.0)

    severe_contours, _ = cv2.findContours(severe_spoilage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_severe = [c for c in severe_contours if cv2.contourArea(c) > 30]
    severe_area = sum(cv2.contourArea(c) for c in valid_severe)
    severe_ratio = min(round((severe_area / max(object_area, 1)) * 100, 2), 100.0)

    total_bad_marks = len(valid_blemishes) + len(valid_severe)
    total_defect_ratio = min(round(blemish_ratio + severe_ratio, 2), 100.0)

    cv2.drawContours(annotated_img, valid_blemishes, -1, (0, 165, 255), 2)
    cv2.drawContours(annotated_img, valid_severe, -1, (0, 0, 255), 2)

    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    laplacian_var = round(cv2.Laplacian(masked_gray, cv2.CV_64F).var(), 1)
    abnormal_texture = laplacian_var > 1250 or laplacian_var < 28
    texture_status = f"Abnormal ({laplacian_var})" if abnormal_texture else f"Normal ({laplacian_var})"

    abnormal_shape = solidity < 0.83 or circularity < 0.30
    shape_status = f"Abnormal / Deformed (Solidity: {solidity})" if abnormal_shape else f"Normal Shape (Solidity: {solidity})"

    valid_pixels = hsv[mask_clean > 0]
    mean_sat = np.mean(valid_pixels[:, 1]) if len(valid_pixels) > 0 else 0
    mean_val = np.mean(valid_pixels[:, 2]) if len(valid_pixels) > 0 else 0
    abnormal_color = (mean_sat < 48) or (mean_val < 58) or (severe_ratio > 8.0)
    color_status = f"Abnormal Discoloration (Sat: {int(mean_sat)})" if abnormal_color else f"Healthy Natural Color (Sat: {int(mean_sat)})"

    ai_edible_confidence = ai_states["clean_fresh_prob"] + (0.72 * ai_states["blemished_edible_prob"])
    cv_usable_flesh = max(0.0, min(100.0, 100.0 - (blemish_ratio * 1.1) - (severe_ratio * 3.5) - (10 if abnormal_texture else 0)))
    
    edibility_pct = round((0.70 * ai_edible_confidence) + (0.30 * cv_usable_flesh), 1)
    edibility_pct = max(0.0, min(100.0, edibility_pct))
    usable_flesh_pct = round(max(0.0, 100.0 - total_defect_ratio), 1)
    diff_benchmark = round(edibility_pct - SAFE_EDIBLE_BENCHMARK, 1)

    is_heavily_rotten = (ai_states["rotten_inedible_prob"] >= 52.0) or (severe_ratio >= 10.0) or (edibility_pct < 52.0)
    has_bad_marks = (total_bad_marks > 0 and total_defect_ratio >= 1.5) or (ai_states["blemished_edible_prob"] >= 35.0)

    if is_heavily_rotten:
        rot_status = "ROTTEN / MOLDY"
        verdict = "NOT EDIBLE — DO NOT CONSUME"
        severity = "error"
        advice = (
            f"Severe rot, mold, or deep spoilage detected (Rotten AI score: {ai_states['rotten_inedible_prob']}%). "
            f"Your sample scored {edibility_pct}% ({abs(diff_benchmark)}% below the {SAFE_EDIBLE_BENCHMARK}% safe standard). Unsafe to eat."
        )
    elif has_bad_marks:
        rot_status = "NOT ROTTEN (Surface Bad Marks Found)"
        verdict = "STILL EDIBLE — Safe After Trimming Bad Marks"
        severity = "warning"
        advice = (
            f"Surface bad marks/bruises detected ({total_bad_marks} marks covering {total_defect_ratio}% of surface), "
            f"but the produce is NOT rotten and remains **EDIBLE** (~{usable_flesh_pct}% healthy usable flesh). "
            f"Cut away the marked/bruised spots before eating."
        )
    else:
        rot_status = "NOT ROTTEN (Clean & Fresh)"
        verdict = "EDIBLE — 100% Safe to Consume"
        severity = "success"
        advice = (
            f"Clean, fresh condition with no significant bad marks! Your sample scored {edibility_pct}% "
            f"(+{diff_benchmark}% above the {SAFE_EDIBLE_BENCHMARK}% safe edibility standard)."
        )

    return {
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
    st.caption("Precision Produce & Edibility Diagnostics")
    st.divider()
    st.markdown(f"""
    **Edibility Standard Benchmarks:**
    - **Optimal Fresh Average:** `{OPTIMAL_FRESH_AVERAGE}%`
    - **Minimum Safe Edibility:** `{SAFE_EDIBLE_BENCHMARK}%`
    - **Trim-Before-Eating Zone:** `52% – 79%`
    - **Rotten / Discard Zone:** `< 52%`
    """)
    st.divider()
    st.markdown("""
    **Color Legend on Defect Scan:**
    - 🟢 **Green Box:** Detected Produce Bounds
    - 🔵 **Blue Outline:** Outer Shape Contour
    - 🟠 **Orange Marks:** Minor Surface Blemishes (Still Edible if Trimmed)
    - 🔴 **Red Marks:** Deep Rot / Fungal Mold (Inedible)
    """)

# --- Main Interface ---
st.title("🌿 AgriScan-AI: High-Precision Produce & Edibility Inspector")
st.write("Select your produce category, optionally type its name, capture or upload a photo, and click the confirmation button to start the AI inspection.")

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
        - **Produce Name:** `{optional_produce_name.strip().title() if optional_produce_name.strip() else "Auto-Detect via AI"}`
        
        Click the confirmation button below to isolate the produce, scan for bad marks or rot, and check edibility.
        """)
        if st.button("✅ Confirm & Analyze Produce"):
            st.session_state["confirmed_scan"] = True

    if st.session_state.get("confirmed_scan", False):
        st.divider()
        with st.spinner("Focusing on produce, clarifying surface details, and running AI diagnostics..."):
            focused_true_pil, clarified_display_pil, focused_true_bgr, fg_mask = clarify_and_focus_produce(pil_img, clean_category)
            is_edible_food, detected_name, match_conf, ai_states, top3_matches = classify_and_assess_edibility(
                focused_true_pil, clean_category, optional_produce_name
            )

        if not is_edible_food:
            st.error(f"🚨 ERROR ANNOUNCEMENT: NOT EDIBLE! This image shows '{detected_name}' ({match_conf}% confidence), which is NOT an edible {clean_category.lower()}.")
            st.warning("Please upload or capture an actual fruit or vegetable.")
            st.image(pil_img, caption=f"Non-Edible Object Detected: {detected_name}", width=380)
        else:
            res = analyze_defects_and_edibility(focused_true_bgr, fg_mask, ai_states)

            if res["severity"] == "error":
                st.error(f"🚨 ERROR ANNOUNCEMENT: {res['verdict']} | {res['rot_status']} ({detected_name})")
            elif res["severity"] == "warning":
                st.warning(f"⚠️ BAD MARKS DETECTED — {res['verdict']} | {res['rot_status']} ({detected_name})")
            else:
                st.success(f"✅ VERIFIED: {res['verdict']} | {res['rot_status']} ({detected_name})")

            st.info(f"**Edibility & Safety Assessment:** {res['advice']}")

            st.subheader("Diagnostic & Edibility Benchmark Summary")
            c1, c2, c3, c4 = st.columns(4)
            name_source_label = "User Specified" if optional_produce_name.strip() else f"{match_conf}% AI match"
            c1.metric(f"Target {clean_category}", detected_name, name_source_label)
            c2.metric("Edibility Score", f"{res['edibility_pct']}%", f"{res['diff_benchmark']}% vs Safe Avg")
            c3.metric("Safe Edible Average", f"{SAFE_EDIBLE_BENCHMARK}%", f"Usable Flesh: {res['usable_flesh_pct']}%")
            c4.metric("Bad Marks / Spots", f"{res['total_bad_marks']} marks", f"{res['total_defect_ratio']}% surface area")

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
                    f"{clean_category} Identification",
                    "AI State Breakdown",
                    "Color Assessment",
                    "Surface Texture",
                    "Shape & Morphology",
                    "Bad Marks vs. Severe Rot",
                    "Final Edibility Verdict"
                ],
                "Condition": [
                    detected_name,
                    "Fresh" if res["severity"] == "success" else ("Blemished (Edible)" if res["severity"] == "warning" else "Rotten (Inedible)"),
                    "ABNORMAL" if res["abnormal_color"] else "NORMAL",
                    "ABNORMAL" if res["abnormal_texture"] else "NORMAL",
                    "ABNORMAL" if res["abnormal_shape"] else "NORMAL",
                    f"{res['total_bad_marks']} Marks ({res['total_defect_ratio']}%)",
                    res["verdict"]
                ],
                "Measured Details": [
                    f"{detected_name} ({name_source_label})" if optional_produce_name.strip() else ", ".join([f"{n} ({c}%)" for n, c in top3_matches]),
                    f"Clean: {ai_states['clean_fresh_prob']}% | Blemished-Edible: {ai_states['blemished_edible_prob']}% | Rotten: {ai_states['rotten_inedible_prob']}%",
                    res["color_status"],
                    res["texture_status"],
                    res["shape_status"],
                    f"Minor Blemishes: {res['blemish_ratio']}% | Deep Rot/Mold: {res['severe_ratio']}% | Usable Flesh: {res['usable_flesh_pct']}%",
                    f"Sample: {res['edibility_pct']}% | Safe Standard: {SAFE_EDIBLE_BENCHMARK}% | Optimal Avg: {OPTIMAL_FRESH_AVERAGE}%"
                ]
            })
else:
    st.info(f"👆 Choose **Upload Image** or **Instant Camera Capture** above, then click **Confirm & Analyze Produce** to view your results.")
