# app.py
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import base64

# ======================================================
# 🔥 Set page config
# ======================================================
st.set_page_config(
    page_title="🌲 Forest Cover Type Predictor",
    page_icon="🌲",
    layout="centered"
)

# ======================================================
# 🌲 Background Image Function
# ======================================================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        [data-testid="stHeader"] {{
            background: rgba(255,255,255,0.4);
        }}

        [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.4);
        }}

        .block-container {{
            background: rgba(255,255,255,0.7);
            padding: 2rem;
            border-radius: 15px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the forest background
set_background("bg.jpg")

st.title("🌲 Forest Cover Type Predictor")

# ======================================================
# 📁 File paths
# ======================================================
CSV_PATH = "cover_type.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# ======================================================
# 📦 Load ML Assets
# ======================================================
@st.cache_resource
def load_assets():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    df = pd.read_csv(CSV_PATH)
    feature_names = [c for c in df.columns if c != "Cover_Type"]
    return model, scaler, le, feature_names, df

model, scaler, le, feature_names, df = load_assets()

st.caption(f"Model loaded. Expecting {len(feature_names)} features.")

# ======================================================
# 🧩 Split features
# ======================================================
soil_cols = [c for c in feature_names if c.startswith("Soil_Type_")]
wild_cols = [c for c in feature_names if c.startswith("Wilderness_Area_")]
num_cols = [c for c in feature_names if c not in soil_cols + wild_cols]

# ======================================================
# 🧮 INPUT UI
# ======================================================
st.subheader("Enter Feature Values")

values = {}
cols = st.columns(2)

# 🔢 Numeric Inputs
for i, col in enumerate(num_cols):
    default_val = float(df[col].mean())
    with cols[i % 2]:
        values[col] = st.number_input(
            col,
            value=default_val,
            step=1.0,
            format="%.4f"
        )

# ======================================================
# 🌲 Dropdown: Wilderness Area
# ======================================================
st.markdown("---")
st.subheader("🌲 Wilderness Area")

wild_labels = {col: col.replace("_", " ") for col in wild_cols}
selected_wild_label = st.selectbox(
    "Select Wilderness Area",
    list(wild_labels.values())
)
selected_wild_col = [k for k, v in wild_labels.items()
                     if v == selected_wild_label][0]

for col in wild_cols:
    values[col] = 1 if col == selected_wild_col else 0

# ======================================================
# 🧱 Dropdown: Soil Type
# ======================================================
st.subheader("🧱 Soil Type")

soil_labels = {col: col.replace("_", " ") for col in soil_cols}
selected_soil_label = st.selectbox(
    "Select Soil Type",
    list(soil_labels.values())
)
selected_soil_col = [k for k, v in soil_labels.items()
                     if v == selected_soil_label][0]

for col in soil_cols:
    values[col] = 1 if col == selected_soil_col else 0

# ======================================================
# 🧠 Predict
# ======================================================
if st.button("Predict Cover Type"):
    try:
        X_input = pd.DataFrame(
            [[values[c] for c in feature_names]],
            columns=feature_names
        )

        X_scaled = scaler.transform(X_input)
        y_int = int(model.predict(X_scaled)[0])
        y_label = le.inverse_transform([y_int])[0]

        st.success(f"🌳 Predicted Forest Cover Type: **{y_label}**")

        # 🎯 Top 3 probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[0]
            top_idx = np.argsort(probs)[::-1][:3]
            top_labels = le.inverse_transform(top_idx)
            top_probs = probs[top_idx]

            st.write("### 🔍 Top Class Probabilities:")
            for lab, p in zip(top_labels, top_probs):
                st.write(f"- **{lab}**: {p*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
