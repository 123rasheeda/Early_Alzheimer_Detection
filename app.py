
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow.lite as tflite

st.set_page_config(page_title="Alzheimer Detection", layout="centered")
st.title("🧠 Early Alzheimer Detection")
st.write("Upload a brain MRI to detect the stage of Alzheimer's disease.")

alz_labels = ['Healthy', 'Mild', 'Moderate', 'VeryMild']

@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tflite.Interpreter(model_path="alzheimer_edge_model.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None, None

interpreter, input_details, output_details = load_tflite_model()

def is_brain_mri(image):
    try:
        img = image.resize((128, 128)).convert("RGB")
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_score = np.mean(edges)
        return edge_score > 5, f"Edge Score: {edge_score:.2f}"
    except:
        return False, "Edge detection failed"

def predict_mri(image):
    img = image.resize((128, 128))
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB) / 255.0
    input_data = np.expand_dims(edge_rgb.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    stage = alz_labels[np.argmax(preds)]
    confidence = round(100 * np.max(preds), 2)
    return stage, confidence, edge

uploaded_file = st.file_uploader("📤 Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file and interpreter:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Brain Image", use_column_width=True)

    valid, edge_feedback = is_brain_mri(img)
    if not valid:
        st.warning(f"❌ Not a brain MRI. Reason: {edge_feedback}")
    else:
        stage, conf, edge = predict_mri(img)
        st.success(f"🧠 Predicted Stage: {stage} ({conf}%)")
        st.image(edge, caption="Edge Detection", use_column_width=True)
elif not uploaded_file:
    st.info("Please upload an image to begin.")
