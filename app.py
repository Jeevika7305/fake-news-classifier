import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("📰 Fake News Classifier")

# Input box
user_input = st.text_area("Enter a news article or headline:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform and predict
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        
        if prediction == 1:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")
