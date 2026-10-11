
# Streamlit demo app, built with the help of Claude (Anthropic).
# The model itself (training and evaluation) is in creditcard.ipynb.
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Bank Fraud Detector", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_data
def load_sample():
    return pd.read_csv("sample.csv")

model = load_model()
sample = load_sample()

st.title("🛡️ Bank Fraud Detector")
st.caption("Python · scikit-learn · Streamlit")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Model comparison (F1)")
    results = {
        "Random Forest": 0.85,
        "Logistic Regression": 0.73,
        "Logistic Regression + balanced": 0.14,
    }
    for name, f1 in results.items():
        st.write(name)
        st.progress(f1)
with col2:
    st.subheader("Try the model")
    b1, b2 = st.columns(2)

    if b1.button("🎲 Random transaction", use_container_width=True):
        st.session_state["row"] = sample.sample(1)
    if b2.button("🚨 Fraud transaction", use_container_width=True):
        st.session_state["row"] = sample[sample["Class"] == 1].sample(1)

    if "row" in st.session_state:
        row = st.session_state["row"]
        x = row.drop(columns=["Class"])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0, 1]
        actual = row["Class"].iloc[0]

        m1, m2 = st.columns(2)
        m1.metric("Amount", f"${row['Amount'].iloc[0]:,.2f}")
        m2.metric("Time (seconds)", int(row["Time"].iloc[0]))

        if pred == 1:
            st.error(f"⚠️ Possible fraud · Risk score {proba:.0%}")
        else:
            st.success(f"✅ Legitimate transaction · Risk score {proba:.0%}")
        st.progress(float(proba))

        st.caption("Actual label: " + ("Fraud" if actual == 1 else "Legitimate"))

st.divider()
p, r, f = st.columns(3)
p.metric("Precision", "0.96")
r.metric("Recall", "0.77")
f.metric("F1", "0.85")
