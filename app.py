import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model & tokenizer (make sure these are saved or downloaded)
model = BertForSequenceClassification.from_pretrained("bert-imdb-sentiment")
tokenizer = BertTokenizer.from_pretrained("bert-imdb-sentiment")
model.eval()

st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Classifier")

review = st.text_area("Enter your movie review")

if st.button("Predict"):
    with st.spinner("Analyzing..."):
        inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: {sentiment}")