import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import joblib
import numpy as np

st.set_page_config(page_title="PETase Tm Predictor", page_icon="🧬", layout="centered")
st.title("🧬 PETase Thermal Stability Predictor")
st.markdown("""
Predict the melting temperature (Tm) of *Is*PETase mutants for plastic degradation. 
This tool extracts 640-D structural features using the **ESM-2 150M** protein language model and predicts thermodynamic stability using an **XGBoost ensemble**.
""")

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
    esm_model.eval()
    xgb_model = joblib.load('petase_xgboost_model (1).pkl')
    return tokenizer, esm_model, xgb_model

with st.spinner("Loading AI Models..."):
    tokenizer, esm_model, xgb_model = load_models()

wt_seq = (
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSR"
    "PSGYGAGTVYYPTNAGGTVIGASTGVGATAIGIGQSNSSGGIQPIQASQWGGTIGTNN"
    "GYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGT"
    "IGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNNGYTVST"
    "QWGGTIGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNNGYTVSTQWGGTIGTNN"
)[:290]

user_sequence = st.text_area("Enter PETase Mutant Sequence (FASTA/Text):", value=wt_seq, height=150)

if st.button("Predict Melting Temperature 🌡️"):
    if user_sequence:
        with st.spinner("Extracting biological features and predicting Tm..."):
            inputs = tokenizer(user_sequence.strip(), return_tensors="pt", truncation=True, max_length=300)
            with torch.no_grad():
                outputs = esm_model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            predicted_tm = xgb_model.predict(embedding)[0]
            
            st.success("Prediction Complete!")
            st.metric(label="Predicted Melting Temperature (Tm)", value=f"{predicted_tm:.2f} °C")
            
            if predicted_tm > 49.0:
                st.info("🔥 This mutant is predicted to be MORE stable than the wild-type (49.0 °C).")
            else:
                st.warning("❄️ This mutant is predicted to be LESS stable than the wild-type (49.0 °C).")
    else:
        st.error("Please enter a protein sequence.")
