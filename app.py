import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# --- Page Configuration ---
st.set_page_config(
    page_title="AgriScan-AI | Full Produce & Edibility Diagnostics",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8faf9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Pre-trained CNN Model ---
@st.cache_resource
def load_classifier():
    return MobileNetV2(weights="imagenet")

model = load_classifier()

EDIBLE_PRODUCE_KEYWORDS = {
    "apple", "granny_smith", "banana", "orange", "lemon", "fig", "pineapple",
    "strawberry", "pomegranate", "custard_apple", "jackfruit", "grape", "melon",
    "watermelon", "cantaloupe", "peach", "pear", "plum", "mango", "kiwi",
    "bell_pepper", "cucumber", "zucchini", "spaghetti_squash", "acorn_squash",
    "butternut_squash", "broccoli", "cauliflower", "cabbage", "head_cabbage",
    "artichoke", "cardoon", "mushroom", "agaric", "gyromitra", "stinkhorn",
    "earthstar", "hen-of-the-woods", "bolete", "corn", "ear", "potato",
    "sweet_potato", "carrot", "onion", "garlic", "tomato", "eggplant",
    "chili", "pepper", "lettuce", "spinach", "radish", "turnip", "beet",
    "pumpkin", "gourd", "berry", "cherry", "apricot", "avocado", "coconut",
    "papaya", "guava", "lychee", "persimmon", "date", "olive", "pea", "bean"
}

def classify_item(pil_image):
    img_resized = pil_image.convert("RGB").resize((224, 224))
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    
    top_label = decoded[0][1].replace("_", " ").title()
    top_conf = round(float(decoded[0][2]) * 100, 1)
    
    is_edible_item = False
    detected_produce_name = top_label
    
    for _, label, prob in decoded:
        clean_label = label.lower()
        if any(kw in clean_label for kw in EDIBLE_PRODUCE_KEYWORDS):
            is_edible_item = True
            detected_produce_name = label.replace("_", " ").title()
            break
            
    return is_edible_item, detected_produce_name, top_label, top_conf, decoded

def analyze_produce_defects(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    img_resized = cv2.resize(img_bgr, (400, 400))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Object Segmentation
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask_sat = cv2.inRange(blur, np.array([0, 30, 25]), np.array([180, 255, 255]))
    kernel = np.ones((7, 7), np.uint8)
    mask_clean = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_img = img_resized.copy()
    aspect_ratio = 1.0
    circularity = 0.0
    solidity = 1.0
    object_area = 400 * 400

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 1000:
            object_area = area
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = round(float(w) / h, 2)
            
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = round(4 * np.pi * (area / (perimeter * perimeter)), 2)
                
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = round(float(area) / hull_area, 2)
            
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 2)
            cv2.drawContours(annotated_img, [largest_contour], -1, (52, 152, 219), 2)

    # 2. Bad Marks, Dark Rot & Mold Detection
    rot_dark_mask = cv2.inRange(hsv, np.array([0, 25, 10]), np.array([28, 255, 105]))
    mold_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 42, 225]))
    
    combined_defect_mask = cv2.bitwise_or(rot_dark_mask, mold_mask)
    defect_mask = cv2.bitwise_and(combined_defect_mask, mask_clean)
    
    defect_contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_defects = [c for c in defect_contours if cv2.contourArea(c) > 25]
    defect_area = sum(cv2.contourArea(c) for c in valid_defects)
    defect_ratio = min(round((defect_area / max(object_area, 1)) * 100, 2), 100.0)

    cv2.drawContours(annotated_img, valid_defects, -1, (0, 0, 255), 2)

    # 3. Abnormal Texture Check
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_clean)
    laplacian_var = round(cv2.Laplacian(masked_gray, cv2.CV_64F).var(), 1)
    abnormal_texture = laplacian_var > 950 or laplacian_var < 35
    if laplacian_var > 950:
        texture_status = f"Abnormal (Severely Wrinkled/Rough: {laplacian_var})"
    elif laplacian_var < 35:
        texture_status = f"Abnormal (Mushy/Soft Surface: {laplacian_var})"
    else:
        texture_status = f"Normal Surface Integrity ({laplacian_var})"

    # 4. Abnormal Shape Check
    abnormal_shape = solidity < 0.85 or circularity < 0.35
    if abnormal_shape:
        shape_status = f"Abnormal / Deformed (Solidity: {solidity})"
    else:
        shape_status = f"Normal Morphology (Solidity: {solidity})"

    # 5. Abnormal Color Check
    valid_pixels = hsv[mask_clean > 0]
    if len(valid_pixels) > 0:
        mean_hue = np.mean(valid_pixels[:, 0])
        mean_sat = np.mean(valid_pixels[:, 1])
        mean_val = np.mean(valid_pixels[:, 2])
    else:
        mean_hue, mean_sat, mean_val = 0, 0, 0

    abnormal_color = (mean_sat < 55) or (mean_val < 65) or (defect_ratio > 12.0)
    if abnormal_color:
        color_status = f"Abnormal Discoloration / Dullness (Sat: {int(mean_sat)}, Val: {int(mean_val)})"
    else:
        color_status = f"Healthy Natural Pigmentation (Hue: {int(mean_hue)}, Sat: {int(mean_sat)})"

    # 6. Rot & Edibility Verdict
    penalty = (defect_ratio * 2.6)
    if abnormal_texture:
        penalty += 18
    if abnormal_shape:
        penalty += 15
    if abnormal_color:
        penalty += 15

    health_score = max(0, min(100, int(100 - penalty)))

    if health_score >= 78 and defect_ratio < 5.0:
        rot_status = "NOT ROTTEN (Fresh)"
        edibility_verdict = "EDIBLE — Safe to Consume"
        severity = "success"
        advice = "No significant rot, mold, or structural abnormalities detected. Safe for consumption."
    elif 45 <= health_score < 78 and defect_ratio < 18.0:
        rot_status = "EARLY SPOILAGE / BLEMISHED"
        edibility_verdict = "EDIBLE WITH CAUTION — Trim Bad Marks"
        severity = "warning"
        advice = "Localized bad marks, minor bruising, or early texture/color changes detected. Cut away affected areas before eating."
    else:
        rot_status = "ROTTEN / SPOILED"
        edibility_verdict = "NOT EDIBLE — DO NOT CONSUME"
        severity = "error"
        advice = "Severe rot, fungal mold, abnormal texture, or heavy discoloration detected. Discard immediately; unsafe to eat."

    return {
        "annotated_img": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
        "defect_mask": cv2.cvtColor(defect_mask, cv2.COLOR_GRAY2RGB),
        "health_score": health_score,
        "rot_status": rot_status,
        "edibility_verdict": edibility_verdict,
        "severity": severity,
        "advice": advice,
        "defect_count": len(valid_defects),
        "defect_ratio": defect_ratio,
        "abnormal_texture": abnormal_texture,
        "texture_status": texture_status,
        "abnormal_shape": abnormal_shape,
        "shape_status": shape_status,
        "abnormal_color": abnormal_color,
        "color_status": color_status,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity
    }

# --- Sidebar ---
with st.sidebar:
    st.title("🌱 AgriScan-AI")
    st.caption("Full-Spectrum Object, Rot & Edibility Diagnostic System")
    st.divider()
    input_mode = st.radio("Select Input Source:", ["Upload Image", "Camera Capture"])
    st.divider()
    st.markdown("""
    **Active Diagnostic Modules:**
    - **CNN Object Classifier:** MobileNetV2
    - **Rot & Mold Scanner:** Dark decay & fuzzy mold masking
    - **Morphology Engine:** Solidity & deformation check
    - **Texture Analyzer:** Laplacian surface integrity
    """)

# --- Main Interface ---
st.title("AgriScan-AI: Produce Recognition, Rot & Edibility Detector")
st.write("Scan any item to identify what it is, detect bad marks or abnormal color/texture/shape, and verify if it is rotten or edible.")

if input_mode == "Upload Image":
    image_file = st.file_uploader("Upload an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Capture a photo")

if image_file is not None:
    pil_img = Image.open(image_file)
    
    with st.spinner("Running Deep Learning Object Recognition & Defect Scan..."):
        is_produce, produce_name, raw_top_label, top_conf, top5_preds = classify_item(pil_img)

    # --- STEP 1: NON-EDIBLE OBJECT CHECK ---
    if not is_produce:
        st.error(f"🚨 ERROR: NOT EDIBLE! Detected object is '{raw_top_label}' ({top_conf}% confidence), which is NOT an edible fruit or vegetable.")
        st.warning("This item is classified as non-food / non-produce and cannot be consumed. Please scan an actual fruit or vegetable.")
        
        col_err1, col_err2 = st.columns([1, 1])
        with col_err1:
            st.image(pil_img, caption=f"Detected Non-Edible Item: {raw_top_label}", use_container_width=True)
        with col_err2:
            st.subheader("AI Object Detection Breakdown")
            st.table({
                "Rank": ["1st Match", "2nd Match", "3rd Match"],
                "Detected Object": [
                    top5_preds[0][1].replace("_", " ").title(),
                    top5_preds[1][1].replace("_", " ").title(),
                    top5_preds[2][1].replace("_", " ").title()
                ],
                "Confidence": [
                    f"{round(float(top5_preds[0][2])*100, 1)}%",
                    f"{round(float(top5_preds[1][2])*100, 1)}%",
                    f"{round(float(top5_preds[2][2])*100, 1)}%"
                ]
            })
    
    # --- STEP 2: PRODUCE QUALITY, ROT & EDIBILITY ANALYSIS ---
    else:
        results = analyze_produce_defects(pil_img)

        if results["severity"] == "error":
            st.error(f"🚨 ERROR ANNOUNCEMENT: {results['edibility_verdict']} | Status: {results['rot_status']} ({produce_name})")
        elif results["severity"] == "warning":
            st.warning(f"⚠️ CAUTION: {results['edibility_verdict']} | Status: {results['rot_status']} ({produce_name})")
        else:
            st.success(f"✅ VERIFIED: {results['edibility_verdict']} | Status: {results['rot_status']} ({produce_name})")

        st.info(f"**Diagnostic Guidance:** {results['advice']}")

        st.subheader("Diagnostic Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Detected Item", produce_name)
        col2.metric("Freshness Score", f"{results['health_score']}%")
        col3.metric("Rot Status", "ROTTEN" if results["severity"] == "error" else ("BLEMISHED" if results["severity"] == "warning" else "NOT ROTTEN"))
        col4.metric("Bad Marks / Spots", f"{results['defect_count']} ({results['defect_ratio']}%)")

        st.divider()

        vis_col1, vis_col2, vis_col3 = st.columns(3)
        with vis_col1:
            st.markdown("**1. Original Input**")
            st.image(pil_img, use_container_width=True)
        with vis_col2:
            st.markdown("**2. Feature & Defect Detection**")
            st.image(results["annotated_img"], caption="Green: Box | Blue: Contour | Red: Bad Marks/Rot", use_container_width=True)
        with vis_col3:
            st.markdown("**3. Rot & Blemish Mask**")
            st.image(results["defect_mask"], caption="Binary mask of detected decay/mold", use_container_width=True)

        st.subheader("Detailed Quality & Abnormality Inspection")
        st.table({
            "Inspection Parameter": [
                "Object Identification",
                "Color Assessment",
                "Surface Texture",
                "Shape & Morphology",
                "Bad Marks / Spot Coverage",
                "Final Edibility Verdict"
            ],
            "Status": [
                "Edible Produce" if is_produce else "Non-Edible",
                "ABNORMAL" if results["abnormal_color"] else "NORMAL",
                "ABNORMAL" if results["abnormal_texture"] else "NORMAL",
                "ABNORMAL" if results["abnormal_shape"] else "NORMAL",
                "HIGH" if results["defect_ratio"] >= 18 else ("MODERATE" if results["defect_ratio"] >= 5 else "LOW"),
                results["edibility_verdict"]
            ],
            "Measured Details": [
                f"{produce_name} ({top_conf}% match)",
                results["color_status"],
                results["texture_status"],
                results["shape_status"],
                f"{results['defect_count']} marks covering {results['defect_ratio']}% of surface",
                results["rot_status"]
            ]
        })
else:
    st.info("👆 Upload an image or capture a photo above to detect what the item is and check if it is rotten or edible.")
