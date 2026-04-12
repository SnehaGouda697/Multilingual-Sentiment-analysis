import streamlit as st
import requests

st.title("EV Sentiment Analyzer")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Enter some text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text}
            )

            result = response.json()

            st.success(f"Sentiment: {result['label']}")
            st.info(f"Confidence: {result['confidence']}")
        except:
            st.error("Backend not running!")