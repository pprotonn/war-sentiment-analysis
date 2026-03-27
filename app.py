import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.title("🌍 War Sentiment Analysis Dashboard")

# Load data
df = pd.read_csv("tweets.csv")

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

df["clean_text"] = df["text"].apply(clean_text)

# Sentiment
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_text"].apply(get_sentiment)

# Show data
st.subheader("📊 Data")
st.write(df)

# Sentiment count
st.subheader("📈 Sentiment Distribution")
st.bar_chart(df["sentiment"].value_counts())

# Country-wise
st.subheader("🌍 Country-wise Sentiment")
st.write(df.groupby("country")["sentiment"].value_counts())