from PIL import Image
import streamlit as st
import google.generativeai as genai
import os
from io import StringIO
import time
import pyttsx3
import io  # To handle in-memory audio
import tempfile

from configs import SYSTEM_PROMPT, SAFETY_SETTINGS, GENERATION_CONFIG, MODEL_NAME

# ---- App Configuration ----
st.set_page_config(
    page_title='HealthMate AI: X-Ray Imaging Diagnosis',
    layout='wide',
    initial_sidebar_state="expanded"
)

# ---- Theme Toggle ----
theme = st.sidebar.selectbox("🌗 Theme", ["Light", "Dark"])

# ---- Enhanced CSS Styling ----
# Use data-testid to reliably target sidebar
css = f"""
<style>
  /* Global gradient background */
  .stApp {{
    background: {'linear-gradient(135deg, #e0f7fa, #80deea)' if theme == 'Light' else 'linear-gradient(135deg, #2c3e50, #34495e)'};
    color: {'#000' if theme == 'Light' else '#fff'};
    font-family: 'Segoe UI', Tahoma, sans-serif;
  }}

  /* Sidebar background and text */
  [data-testid="stSidebar"] {{
    background: {'#b2dfdb' if theme == 'Light' else '#22313f'} !important;
    color: {'#000' if theme == 'Light' else '#fff'} !important;
    padding: 1rem;
    border-radius: 0 1rem 1rem 0;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
  }}
  [data-testid="stSidebar"] .css-1d391kg {{
    background: transparent;
    box-shadow: none;
    border: none;
  }}
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 {{
    color: {'#004d40' if theme == 'Light' else '#1abc9c'};
  }}
  [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
    color: {'#000' if theme == 'Light' else '#fff'};
  }}

  /* Report card */
  .report-box {{
    background: {'#fff' if theme == 'Light' else '#2c2c2c'};
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 1.5rem;
  }}

  /* Buttons */
  .stButton > button {{
    background: {'#00796b' if theme == 'Light' else '#1abc9c'} !important;
    color: #fff !important;
    font-weight: 600;
    border-radius: 0.75rem;
    padding: 0.75rem 1.5rem;
    transition: background 0.3s ease;
  }}
  .stButton > button:hover {{
    background: {'#004d40' if theme == 'Light' else '#16a085'} !important;
  }}

  /* File uploader */
  .stFileUploader {{
    border: 2px dashed {'#00796b' if theme == 'Light' else '#1abc9c'};
    border-radius: 0.75rem;
    padding: 1rem;
    background: {'#e0f2f1' if theme == 'Light' else '#34495e'};
  }}
  .stFileUploader label {{
    color: {'#000' if theme == 'Light' else '#fff'} !important;
    font-weight: 600;
  }}

  /* Text inputs */
  .stTextInput > div > input {{
    border-radius: 0.5rem;
    padding: 0.5rem;
    border: 1px solid {'#00796b' if theme == 'Light' else '#1abc9c'};
  }}
  .stTextInput label {{
    color: {'#000' if theme == 'Light' else '#fff'} !important;
    font-weight: 600;
  }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---- Header ----
st.markdown(
    """
    <div style='text-align:center; margin-bottom:2rem;'>
      <h1>🩺 <span style='color:#00796b'>HealthMate AI</span></h1>
      <h4>Analyzing X-Ray medical images with AI insights</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Sidebar Info ----
st.sidebar.header("ℹ️ How to Use")
st.sidebar.write(
    "1. Upload a clear X-Ray image (under 5MB).  \n"
    "2. Click **Analyze** to get AI feedback.  \n"
    "3. Download the report if needed."
)
st.sidebar.header("💡 Tips")
st.sidebar.write(
    "- Use high-quality chest or bone X-Rays.\n"
    "- Avoid blurry or noisy images.\n"
    "- Note: AI can make Mistakes!"
)

# ---- Configure Gemini AI ----
from api_key import API_KEY

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    safety_settings=SAFETY_SETTINGS,
    generation_config=GENERATION_CONFIG,
    system_instruction=SYSTEM_PROMPT
)

# ---- Initialize session state ----
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

init_state('history', [])
init_state('image_uploaded', False)
init_state('image_data', None)
init_state('report_text', '')

# ---- Upload and Analyze UI ----
col1, col2 = st.columns([1, 3])
with col1:
    uploaded_file = st.file_uploader(
        "📤 Upload X-Ray Image (PNG/JPG, <5MB)", type=["png", "jpg", "jpeg"]
    )
    submit_btn = st.button("🔍 Analyze X-Ray Image")

with col2:
    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.warning("⚠️ File too large. Please upload an image under 5MB.")
        else:
            image_data = Image.open(uploaded_file)
            st.image(image_data, caption="🖼️ Uploaded X-Ray Image", use_column_width=True)
            st.session_state['image_uploaded'] = True
            st.session_state['image_data'] = image_data

            if submit_btn:
                progress = st.progress(0)
                for i in range(1, 6):
                    progress.progress(i * 20)
                    time.sleep(0.2)

                content = [
                    "Analyze this image and describe possible medical conditions.",
                    image_data
                ]
                with st.spinner("🧠 AI is analyzing your image..."):
                    chat_session = model.start_chat()
                    response = chat_session.send_message(content)

                st.session_state['report_text'] = response.text
                st.markdown(
                    "<div class='report-box'><h4>📝 Analysis Report:</h4>",
                    unsafe_allow_html=True
                )
                st.write(st.session_state['report_text'])
                st.markdown("</div>", unsafe_allow_html=True)

                download_str = StringIO(st.session_state['report_text'])
                st.download_button(
                    label="📥 Download Report",
                    data=download_str.getvalue(),
                    file_name="xray_diagnosis_report.txt",
                    mime="text/plain"
                )

# ---- Audio Feedback using pyttsx3 ----
if st.session_state['report_text']:
    if st.button("🔊 Listen to Report", key="listen_report"):
        text = st.session_state['report_text']
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, 'rb') as f:
            wav_bytes = f.read()
        audio_buffer = io.BytesIO(wav_bytes)
        st.audio(audio_buffer, format='audio/wav')
        os.remove(wav_path)

# ---- Chatbot Interaction ----
user_message = st.text_input("💬 Chat with the AI:", "")
if user_message:
    if st.session_state['image_uploaded']:
        st.session_state['history'].append({"role": "user", "content": user_message})
        with st.spinner("🧠 AI is responding..."):
            chat_session = model.start_chat()
            response = chat_session.send_message([user_message])
        st.write(f"**AI Response:** {response.text}")
        st.session_state['history'].append({"role": "assistant", "content": response.text})
    else:
        st.warning(
            "⚠️ Please upload an X-ray image first for analysis before chatting."
        )

# ---- Disclaimer ----
st.markdown(
    "<div class='disclaimer'>⚠️ This tool is for educational and experimental purposes only."
    " It does not replace professional medical advice or diagnosis.</div>",
    unsafe_allow_html=True
)
