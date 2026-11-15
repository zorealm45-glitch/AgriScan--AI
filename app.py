import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AgriScan-AI | Agricultural Diagnostic System",
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

def analyze_produce_image(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    img_resized = cv2.resize(img_bgr, (400, 400))
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 1. Object Segmentation
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask_sat = cv2.inRange(blur, np.array([0, 35, 30]), np.array([180, 255, 255]))
    kernel = np.ones((7, 7), np.uint8)
    mask_clean = cv2.morphologyEx(mask_sat, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_img = img_resized.copy()
    aspect_ratio = 1.0
    circularity = 0.0
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
            
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (46, 204, 113), 2)
            cv2.drawContours(annotated_img, [largest_contour], -1, (52, 152, 219), 2)

    # 2. Spot & Defect Detection
    brown_dark_mask = cv2.inRange(hsv, np.array([5, 40, 15]), np.array([25, 255, 110]))
    spot_mask = cv2.bitwise_and(brown_dark_mask, mask_clean)
    
    spot_contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_spots = [c for c in spot_contours if cv2.contourArea(c) > 25]
    spot_area = sum(cv2.contourArea(c) for c in valid_spots)
    spot_ratio = min(round((spot_area / max(object_area, 1)) * 100, 2), 100.0)

    cv2.drawContours(annotated_img, valid_spots, -1, (0, 0, 255), 2)

    # 3. Texture Analysis
    laplacian_var = round(cv2.Laplacian(gray, cv2.CV_64F).var(), 1)

    # 4. Color Profile
    valid_pixels = hsv[mask_clean > 0]
    mean_hue = np.mean(valid_pixels[:, 0]) if len(valid_pixels) > 0 else 0

    if mean_hue < 12 or mean_hue > 165:
        color_profile = "Red / Crimson Dominant (e.g., Tomato, Apple, Strawberry)"
    elif 12 <= mean_hue < 32:
        color_profile = "Yellow / Orange Dominant (e.g., Banana, Citrus, Mango)"
    elif 32 <= mean_hue <= 85:
        color_profile = "Green Dominant (e.g., Cucumber, Pepper, Leafy Produce)"
    else:
        color_profile = "Mixed / Neutral Tone"

    # 5. Freshness Score
    health_score = max(0, min(100, int(100 - (spot_ratio * 2.8))))
    if health_score >= 80:
        status = "Healthy / Fresh"
        recommendation = "Optimal condition for storage, transport, or market distribution."
    elif 50 <= health_score < 80:
        status = "Early Blemishes / Moderate"
        recommendation = "Minor surface spots detected. Prioritize immediate consumption or sorting."
    else:
        status = "Abnormal / Spoilage Detected"
        recommendation = "High surface decay/spotting detected. Isolate from healthy batch."

    return {
        "annotated_img": cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
        "spot_mask": cv2.cvtColor(spot_mask, cv2.COLOR_GRAY2RGB),
        "health_score": health_score,
        "status": status,
        "recommendation": recommendation,
        "spot_count": len(valid_spots),
        "spot_ratio": spot_ratio,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "texture_score": laplacian_var,
        "color_profile": color_profile
    }

with st.sidebar:
    st.title("🌱 AgriScan-AI")
    st.caption("Computer Vision Produce Diagnostic Prototype")
    st.divider()
    input_mode = st.radio("Select Input Source:", ["Upload Image", "Camera Capture"])
    st.divider()
    st.markdown("""
    - **Color Space:** HSV Segmentation
    - **Defect Detection:** Blemish Masking
    - **Morphology:** Aspect Ratio & Circularity
    - **Texture:** Laplacian Surface Variance
    """)

st.title("AgriScan-AI: Automated Produce Quality & Defect Diagnostics")
st.write("Upload or capture an image of fruits or vegetables to analyze external features (color, texture, spots, and shape).")

if input_mode == "Upload Image":
    image_file = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
else:
    image_file = st.camera_input("Take a photo of the produce")

if image_file is not None:
    pil_img = Image.open(image_file)
    with st.spinner("Analyzing visual features..."):
        results = analyze_produce_image(pil_img)

    st.subheader("Diagnostic Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Freshness Score", f"{results['health_score']}%")
    col2.metric("Condition Status", results['status'])
    col3.metric("Detected Spots", f"{results['spot_count']} ({results['spot_ratio']}%)")
    col4.metric("Shape Circularity", f"{results['circularity']}")

    if results['health_score'] >= 80:
        st.success(f"**Recommendation:** {results['recommendation']}")
    elif results['health_score'] >= 50:
        st.warning(f"**Recommendation:** {results['recommendation']}")
    else:
        st.error(f"**Recommendation:** {results['recommendation']}")

    st.divider()

    vis_col1, vis_col2, vis_col3 = st.columns(3)
    with vis_col1:
        st.markdown("**1. Original Input**")
        st.image(pil_img, use_container_width=True)
    with vis_col2:
        st.markdown("**2. Feature & Defect Detection**")
        st.image(results["annotated_img"], caption="Green: Box | Blue: Contour | Red: Spots", use_container_width=True)
    with vis_col3:
        st.markdown("**3. Blemish Isolation Mask**")
        st.image(results["spot_mask"], caption="Binary mask of dark/abnormal regions", use_container_width=True)

    st.subheader("Extracted Morphological & Visual Features")
    st.table({
        "Feature Category": ["Color Profile (HSV)", "Surface Texture (Laplacian)", "Bounding Aspect Ratio (W/H)", "Shape Circularity", "Abnormal Spot Coverage"],
        "Measured Value": [
            results["color_profile"],
            f"{results['texture_score']} (Sharpness/roughness index)",
            str(results["aspect_ratio"]),
            f"{results['circularity']} (1.0 = Perfect Circle)",
            f"{results['spot_ratio']}% of total surface area"
        ]
    })
else:
    st.info("👆 Please upload an image or use the camera from above to start the diagnostic scan.")
