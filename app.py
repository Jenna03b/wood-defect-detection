import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Wood Defect Detecion", layout="wide")

st.title("Automated Wood Defect Detecion")
st.write("Please upload a photo of your board to identify any surface defects.")

@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

st.sidebar.header("Detection settings")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

upload_file = st.sidebar.file_uploader("Select a graphic file", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Oryginal photo")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing the image..."):
        results = model.predict(source=img_array, conf=conf_thresh, imgsz=768)
        res_plotted = results[0].plot()

    with col2:
        st.subheader("Surface defects")
        st.image(res_plotted, channels="BGR", use_container_width=True)

    st.markdown("---")
    st.subheader("Results details")
    boxes = results[0].boxes

    if len(boxes) > 0:
        for box in boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- **Disadvantage:** `{cls_name}` | **Confidence:** `{conf:.2%}`")
    else: 
        st.info("No defects were detected in the given image for the selected threshold.")