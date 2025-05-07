import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("news_dataset.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["title"] + " " + data["text"], data["label"], test_size=0.2, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained! Accuracy: {accuracy:.2f}")

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
