import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="كاشف الاحتيال البنكي", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {direction: rtl; text-align: right;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_data
def load_sample():
    return pd.read_csv("sample.csv")

model = load_model()
sample = load_sample()

st.title("🛡️ كاشف الاحتيال البنكي")
st.caption("Python · scikit-learn · Streamlit")

col1, col2 = st.columns(2)

# ---------- العمود الأول: تجربة النموذج ----------
with col1:
    st.subheader("جرّبي النموذج")
    b1, b2 = st.columns(2)

    if b1.button("🎲 عملية عشوائية", use_container_width=True):
        st.session_state["row"] = sample.sample(1)
    if b2.button("🚨 عملية احتيال", use_container_width=True):
        st.session_state["row"] = sample[sample["Class"] == 1].sample(1)

    if "row" in st.session_state:
        row = st.session_state["row"]
        x = row.drop(columns=["Class"])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0, 1]
        actual = row["Class"].iloc[0]

        m1, m2 = st.columns(2)
        m1.metric("المبلغ", f"${row['Amount'].iloc[0]:,.2f}")
        m2.metric("الوقت (بالثواني)", int(row["Time"].iloc[0]))

        if pred == 1:
            st.error(f"⚠️ احتيال محتمل · درجة الخطورة {proba:.0%}")
        else:
            st.success(f"✅ عملية سليمة · درجة الخطورة {proba:.0%}")
        st.progress(float(proba))

        st.caption("القرار الحقيقي: " + ("احتيال" if actual == 1 else "سليمة"))

# ---------- العمود الثاني: مقارنة النماذج ----------
with col2:
    st.subheader("مقارنة النماذج (f1)")
    results = {
        "Random Forest": 0.85,
        "Logistic Regression": 0.73,
        "Logistic Regression + balanced": 0.14,
    }
    for name, f1 in results.items():
        st.write(name)
        st.progress(f1)

st.divider()
p, r, f = st.columns(3)
p.metric("precision", "0.96")
r.metric("recall", "0.77")
f.metric("f1", "0.85")
