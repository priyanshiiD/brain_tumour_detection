import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def get_model():
    # compile=False avoids legacy training config/metric deserialize issues on cloud.
    return load_model("brain_tumor_model.h5", compile=False)


@st.cache_data
def get_class_indices():
    with open("class_indices.json", "r") as f:
        return json.load(f)


try:
    model = get_model()
    class_indices = get_class_indices()
except Exception as e:
    st.error("Model failed to load. Please check deployment logs.")
    st.exception(e)
    st.stop()

# ---------------------------
# UI
# ---------------------------

with st.sidebar:
    st.header("Dataset")
    st.markdown("**Brain MRI Images for Brain Tumor Detection**")
    st.markdown("[Open on Kaggle](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)")
    st.divider()
    st.caption("Upload one MRI image and the model will classify it as Tumor or No Tumor.")

st.title("Brain Tumor Detection")
st.write("Upload an MRI image or pick a sample image to test the model.")

# ---------------------------
# Input Selection
# ---------------------------

def render_prediction(img, caption_text):
    display_col, result_col = st.columns([1.2, 1])

    with display_col:
        st.image(img, caption=caption_text, use_container_width=True)

    img_pil = Image.fromarray(img).convert("RGB")
    img_resized = img_pil.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    tumor_prob = float(prediction)
    no_tumor_prob = 1 - tumor_prob

    with result_col:
        st.subheader("Prediction")
        if prediction > 0.5:
            st.error("Tumor Detected")
            st.metric("Confidence", f"{tumor_prob * 100:.1f}%")
        else:
            st.success("No Tumor Detected")
            st.metric("Confidence", f"{no_tumor_prob * 100:.1f}%")

        st.progress(tumor_prob)
        st.caption(f"Tumor probability: {tumor_prob:.2f} | No Tumor probability: {no_tumor_prob:.2f}")


sample_dir = Path("sample_images")
sample_paths = []
if sample_dir.exists():
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        sample_paths.extend(sorted(sample_dir.glob(pattern)))

input_mode = st.radio("Choose image source", ["Upload image", "Use sample image"], horizontal=True)

if input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        img = np.array(uploaded_image)
        render_prediction(img, "Uploaded MRI")
else:
    if sample_paths:
        selected_sample = st.selectbox("Select sample image", sample_paths, format_func=lambda p: p.name)
        sample_img = np.array(Image.open(selected_sample).convert("RGB"))
        render_prediction(sample_img, f"Sample MRI: {selected_sample.name}")
    else:
        st.info("No sample images found in sample_images folder.")