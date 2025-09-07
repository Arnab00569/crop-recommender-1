# streamlit_app.py
import os, pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from typing import List

st.set_page_config(page_title="Crop Recommender", layout="centered")

MODEL_DIR = "/app/models" if "STREAMLIT_SERVER" in os.environ else "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(os.path.join(MODEL_DIR, "scaler_x.pkl"), "rb") as f: scaler_x = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "dur_scaler.pkl"), "rb") as f: dur_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "wreq_scaler.pkl"), "rb") as f: wreq_scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_name.pkl"), "rb") as f: le_name = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "le_water.pkl"), "rb") as f: le_water = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "soil_to_idx.pkl"), "rb") as f: soil_to_idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "sown_to_idx.pkl"), "rb") as f: sown_to_idx = pickle.load(f)
    return model, scaler_x, dur_scaler, wreq_scaler, le_name, le_water, soil_to_idx, sown_to_idx

model, scaler_x, dur_scaler, wreq_scaler, le_name, le_water, soil_to_idx, sown_to_idx = load_artifacts()

def map_or_unknown(k, mapping):
    k = str(k)
    return mapping.get(k, len(mapping))

def parse_sown(x):
    s = str(x).strip().lower()
    month_map = {"jan":"1","feb":"2","mar":"3","apr":"4","may":"5","jun":"6","jul":"7","aug":"8","sep":"9","oct":"10","nov":"11","dec":"12"}
    return month_map.get(s, s)

st.title("🌾 Crop Recommender")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    soil = col1.text_input("SOIL", value="sandy")
    sown = col1.text_input("SOWN (month/text)", value="jan")
    soil_ph = col2.number_input("SOIL_PH", value=6.5, format="%.2f")
    temp = col1.number_input("TEMP (°C)", value=25.0, format="%.2f")
    humidity = col2.number_input("RELATIVE HUMIDITY (%)", value=60.0, format="%.2f")
    n = col1.number_input("N", value=10.0, format="%.2f")
    p = col2.number_input("P", value=5.0, format="%.2f")
    k = col1.number_input("K", value=5.0, format="%.2f")
    top_k = col2.slider("Top k", 1, 10, 5)
    submitted = st.form_submit_button("Recommend crops")

if submitted:
    soil_idx = map_or_unknown(soil.lower(), soil_to_idx)
    sown_token = parse_sown(sown)
    sown_idx = map_or_unknown(sown_token, sown_to_idx)
    num_vec = scaler_x.transform([[soil_ph, temp, humidity, n, p, k]]).astype(np.float32)
    preds = model.predict({"soil_in": np.array([soil_idx]), "sown_in": np.array([sown_idx]), "num_in": np.array(num_vec)}, verbose=0)
    name_probs = preds[0][0]
    top_idxs = np.argsort(name_probs)[::-1][:top_k]
    results = []
    for idx in top_idxs:
        crop = le_name.inverse_transform([idx])[0]
        prob = float(name_probs[idx])
        results.append((crop, prob))
    st.subheader("Top crop recommendations")
    for crop, prob in results:
        st.write(f"- **{crop}** — probability {prob:.3f}")

st.markdown("---")
st.caption("Model and UI running on Streamlit Community Cloud. If you hit memory/cold-start issues, use tensorflow-cpu or upgrade runtime.")
