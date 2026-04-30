import streamlit as st
import requests

st.set_page_config(page_title="EV Sentiment Analyzer")

st.title("🌍 EV Sentiment Analyzer")
st.write("Analyze sentiment of Electric Vehicle reviews")

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text}
            )

            result = response.json()
            label = result['label']
            confidence = result['confidence']

            # 🎯 Colored output
            if label == "positive":
                st.success(f"😊 Positive")
            elif label == "negative":
                st.error(f"😡 Negative")
            else:
                st.warning(f"😐 Neutral")

            st.info(f"Confidence: {confidence}")

        except:
            st.error("❌ Backend not running! Start FastAPI server")