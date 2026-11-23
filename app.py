import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Chilli Health Classifier", page_icon="🌶️", layout="centered")

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "keras_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("keras_model.h5 is missing. Keep it in the same folder as app.py.")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_labels():
    if not LABELS_PATH.exists():
        return ["healthy", "unhealthy"]
    labels = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        labels.append(parts[1] if len(parts) == 2 and parts[0].isdigit() else line)
    return labels or ["healthy", "unhealthy"]

def prepare_image(image):
    # Teachable Machine image models expect RGB pixels scaled to [-1, 1].
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)

st.title("🌶️ Chilli Health Classifier")
st.write("Upload a chilli image to classify it as **Healthy** or **Unhealthy**.")
st.caption("This app uses your Google Teachable Machine model.")

uploaded = st.file_uploader("Choose a chilli image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    try:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded image", use_container_width=True)
        if st.button("Classify image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                model = load_model()
                labels = load_labels()
                batch = prepare_image(image)
                prediction = np.asarray(model.predict(batch, verbose=0)).reshape(-1)

                if prediction.size == 0:
                    st.error("The model returned no predictions.")
                else:
                    # Teachable Machine classification models return one score per label.
                    best_index = int(np.argmax(prediction))
                    if best_index >= len(labels):
                        labels = [f"Class {i}" for i in range(prediction.size)]
                    label = labels[best_index].strip()
                    confidence = float(prediction[best_index])
                    normalized = label.lower()

                    st.subheader("Result")
                    if "unhealthy" in normalized:
                        st.error(f"Unhealthy chilli — {confidence:.1%} confidence")
                    elif "healthy" in normalized:
                        st.success(f"Healthy chilli — {confidence:.1%} confidence")
                    else:
                        st.info(f"{label} — {confidence:.1%} confidence")

                    st.write("**Prediction scores**")
                    for i, score in enumerate(prediction):
                        name = labels[i] if i < len(labels) else f"Class {i}"
                        st.write(f"{name}: {float(score):.1%}")
                        st.progress(float(np.clip(score, 0, 1)))

                    st.caption("AI predictions can be wrong. Use this result as an aid, not as a definitive food-safety assessment.")
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.info("Upload a clear photo of one chilli to get started.")

with st.expander("About this model"):
    st.write("Model input: 224 × 224 RGB image. The app scales pixel values to [-1, 1], the standard preprocessing used by Teachable Machine image models.")
    st.write("Labels are read from labels.txt: 0 healthy, 1 unhealthy.")
