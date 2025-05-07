import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f1f1f1;
        font-family: 'Roboto', sans-serif;
    }
    .css-1v3fvcr {
        background-color: #003366 !important;  /* Dark blue header */
        color: white !important;
    }
    .stButton>button {
        background-color: #28a745 !important; /* Green button */
        color: white !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📰 Fake News AI Detector")

# Input text area for user to enter article
news_input = st.text_area("Enter a news article to check if it's real or fake:")

# Show a spinner while processing the text
with st.spinner('Analyzing...'):
    if st.button("Check News"):
        if news_input.strip() == "":
            st.warning("Please enter some text!")
        else:
            transformed_text = vectorizer.transform([news_input])
            prediction = model.predict(transformed_text)[0]
            confidence = model.predict_proba(transformed_text)[0][prediction]  # Confidence score

            # Display result
            if prediction == 1:
                st.success(f"✅ This news is **REAL**! Confidence: {confidence*100:.2f}%")
            else:
                st.error(f"🚨 This news is **FAKE**! Confidence: {confidence*100:.2f}%")
            
            # Show Word Cloud (Optional)
            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(news_input)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

# Sidebar with additional info
st.sidebar.header("About the Model:")
st.sidebar.text("This model detects if a news article is real or fake based on historical data.")
