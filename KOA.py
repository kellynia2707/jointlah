import streamlit as st
from PIL import Image
import numpy as np
import base64
import io
import cv2
import os
import tempfile
import pandas as pd
import altair as alt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime

# Load KOA classification model
koa_model = load_model("C:/Usersuser/Downloads/Jointlah/Jointlah/koa_model.h5", compile=False)

# Class labels
classes = ['Healthy', 'Moderate', 'Severe']

# Streamlit config
st.set_page_config(page_title="Jointlah", layout="centered", page_icon="🦴")

# Session state init
if "page" not in st.session_state:
    st.session_state.page = "form"
if "image" not in st.session_state:
    st.session_state.image = None

# Header
def show_header():
    st.markdown("""
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <h1 style='color:#FF6F61;'>Jointlah</h1>
        </div>
        <p style='color: grey; font-style: italic;'>KOA X-ray Diagnosis – Healthy • Moderate • Severe</p>
    """, unsafe_allow_html=True)

show_header()

# Rule-based check for valid knee X-ray
def is_knee_xray(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 70, 180)
    edge_density = np.sum(edges > 100) / edges.size

    brightness = np.mean(gray)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    vertical_gradient_strength = np.mean(np.abs(sobel_y))

    h, w = gray.shape
    aspect_ratio = w / h

    return (
        0.02 < edge_density < 0.15 and
        90 < brightness < 170 and
        vertical_gradient_strength > 25 and
        0.8 < aspect_ratio < 1.25
    )

# Page 1: Form
if st.session_state.page == "form":
    uploaded_file = st.file_uploader("Upload Knee X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    image = None
    valid_xray = False

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        if is_knee_xray(image):
            valid_xray = True
            st.success("✅ Valid knee X-ray detected.")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.error("❌ This image does not appear to be a valid knee X-ray.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            age = st.number_input("Age", 1, 120, value=50)
        with col2:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col3:
            pain_level = st.slider("Pain Level (1–10)", 1, 10, value=5)

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        next_btn = st.form_submit_button("Next")
        st.markdown("</div>", unsafe_allow_html=True)

        if next_btn:
            if not uploaded_file:
                st.error("❌ Please upload an image before continuing.")
            elif not valid_xray:
                st.error("❌ The uploaded image is not a valid knee X-ray. You cannot continue.")
            else:
                st.session_state.age = age
                st.session_state.gender = gender
                st.session_state.pain_level = pain_level
                st.session_state.image = image
                st.session_state.page = "result"
                st.rerun()

# Page 2: Result
elif st.session_state.page == "result":
    image = st.session_state.image

    if st.button("Back"):
        st.session_state.page = "form"
        st.rerun()

    st.markdown("### 👤 Patient Details")
    st.markdown(f"- **Age:** {st.session_state.age}")
    st.markdown(f"- **Gender:** {st.session_state.gender}")
    st.markdown(f"- **Pain Level:** {st.session_state.pain_level}/10")

    st.image(image, caption="Uploaded Knee X-ray", use_container_width=True)

    resized = image.resize((224, 224))
    arr = np.array(resized)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    prediction = koa_model.predict(arr)
    predicted_grade = int(np.argmax(prediction))
    predicted_label = classes[predicted_grade]
    confidence = round(np.max(prediction) * 100, 2)
    severity_score = round(np.sum(prediction * np.array([0, 1, 2])) / 2 * 100, 2)

    desc = {
        0: "Healthy knee with normal joint spacing.",
        1: "Moderate signs of osteoarthritis present.",
        2: "Severe degeneration of knee joint structures.",
    }
    advice = {
        0: "Maintain a healthy lifestyle and exercise regularly.",
        1: "Consider physiotherapy and joint monitoring.",
        2: "Medical intervention is recommended. Please consult a specialist.",
    }

    st.markdown("---")
    st.markdown(f"""
        <div class='highlight-box' style='background:#fffbe6;padding:20px;border-left:6px solid #FF6F61;border-radius:10px;'>
            <h3>Prediction: {predicted_label}</h3>
            <p><b>Confidence:</b> {confidence}%</p>
            <p><b>Severity Score:</b> {severity_score}/100</p>
            <p><b>Description:</b> {desc[predicted_grade]}</p>
            <p><b>Advice:</b> {advice[predicted_grade]}</p>
        </div>
    """, unsafe_allow_html=True)

    df = pd.DataFrame({
        "Class": classes,
        "Confidence": prediction[0]
    })

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Class', sort=None),
        y=alt.Y('Confidence', scale=alt.Scale(domain=[0, 1])),
        color='Class'
    ).properties(height=250, title="Model Confidence Distribution")

    st.altair_chart(chart, use_container_width=True)

    def generate_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 800, "KOA X-ray Diagnosis Report")

        y = 770
        c.setFont("Helvetica", 12)
        lines = [
            f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Age: {st.session_state.age}",
            f"Gender: {st.session_state.gender}",
            f"Pain Level: {st.session_state.pain_level}/10",
            f"Prediction: {predicted_label}",
            f"Confidence: {confidence}%",
            f"Severity Score: {severity_score}/100",
            f"Description: {desc[predicted_grade]}",
            f"Advice: {advice[predicted_grade]}",
        ]
        for line in lines:
            y -= 20
            c.drawString(100, y, line)

        img_fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(img_fd)
        image.save(img_path)
        c.drawImage(ImageReader(img_path), 100, y - 220, width=200, height=200)

        chart_path = tempfile.mktemp(suffix=".png")
        chart.save(chart_path)
        c.drawImage(ImageReader(chart_path), 320, y - 220, width=200, height=200)

        c.showPage()
        c.save()
        buffer.seek(0)

        os.unlink(img_path)
        os.unlink(chart_path)

        return buffer

    pdf = generate_pdf()
    b64_pdf = base64.b64encode(pdf.read()).decode('utf-8')
    st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="KOA_Report.pdf">📄 Download PDF Report</a>', unsafe_allow_html=True)
