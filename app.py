import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from openai import OpenAI
from typing import List, Optional

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="Game Market Intelligence", layout="wide")

# ================== NLP SETUP ==================
try:
    _ = SentimentIntensityAnalyzer()
except:
    nltk_download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

RAWG_BASE_URL = "https://api.rawg.io/api"

# ================== DATA FUNCTIONS ==================

@st.cache_data(show_spinner=False)
def fetch_trending_games(api_key, page_size=40, ordering="-added"):
    if not api_key:
        return pd.DataFrame()
    
    url = f"{RAWG_BASE_URL}/games"
    params = {"key": api_key, "page_size": page_size, "ordering": ordering}
    res = requests.get(url, params=params)
    data = res.json().get("results", [])

    rows = []
    for g in data:
        rows.append({
            "name": g.get("name"),
            "released": g.get("released"),
            "rating": g.get("rating"),
            "ratings_count": g.get("ratings_count"),
            "metacritic": g.get("metacritic"),
            "genres": ", ".join([x["name"] for x in g.get("genres", [])]),
            "platforms": ", ".join([p["platform"]["name"] for p in g.get("platforms", [])]) if g.get("platforms") else None
        })
    
    return pd.DataFrame(rows)

def fit_trend(df, target):
    df = df.dropna()
    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df[target]

    model = LinearRegression().fit(X, y)
    future_t = np.arange(len(df), len(df)+5).reshape(-1,1)
    preds = model.predict(future_t)

    future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(5)]
    df_future = pd.DataFrame({"date": future_dates, f"pred_{target}": preds})
    return model, df_future

def analyze_sentiment(texts: List[str]) -> pd.DataFrame:
    results = []
    for t in texts:
        scores = sia.polarity_scores(t)
        label = "positive" if scores["compound"] >= 0.05 else "negative" if scores["compound"] <= -0.05 else "neutral"
        results.append({"text": t, "label": label, "compound": scores["compound"]})
    return pd.DataFrame(results)

def fetch_steam_reviews(app_id, limit=50):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {"json": 1, "num_per_page": limit}
    r = requests.get(url, params=params).json()
    return [x["review"] for x in r.get("reviews", [])]

def fetch_steam_appdetails(app_id):
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": app_id}
    r = requests.get(url, params=params).json()
    if not r.get(app_id, {}).get("success"):
        return None
    d = r[app_id]["data"]

    price = d.get("price_overview", {})
    return {
        "name": d.get("name"),
        "price": price.get("final", 0)/100 if price else 0,
        "currency": price.get("currency"),
        "discount": price.get("discount_percent"),
        "metacritic": (d.get("metacritic") or {}).get("score"),
        "recommendations": (d.get("recommendations") or {}).get("total"),
        "is_free": d.get("is_free"),
        "genres": ", ".join([g["description"] for g in d.get("genres", [])])
    }

def collect_competitors(ids):
    data = []
    for appid in ids:
        info = fetch_steam_appdetails(appid)
        if info:
            data.append(info)
    return pd.DataFrame(data)

# ================== GENERATIVE AI ==================

def generate_ai_insight(api_key, system_prompt, user_prompt):
    try:
        client = OpenAI(api_key=api_key)
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
            max_tokens=800
        )
        return result.choices[0].message.content
    except Exception as e:
        return str(e)

# ================== SIDEBAR ==================
st.sidebar.title("Configuration")
rawg_key = st.sidebar.text_input("RAWG API Key", type="password")
openai_key = st.sidebar.text_input("OpenAI API Key (For AI Insight)", type="password")

page = st.sidebar.radio("Select Module", [
    "Market Overview",
    "Trend Prediction",
    "Sentiment Analysis",
    "Competitor Analysis"
])

# ================== 1. MARKET OVERVIEW ==================
if page == "Market Overview":
    st.title("🎮 Market Overview")

    if not rawg_key:
        st.warning("Please enter RAWG API Key.")
        st.stop()

    df = fetch_trending_games(rawg_key)

    st.dataframe(df)
    top = df.sort_values("rating", ascending=False).head(10)

    st.bar_chart(top.set_index("name")["rating"])

    if openai_key:
        if st.button("Generate AI Market Insight"):
            prompt = f"Analyze the following game market data:\n{top.to_string()}"
            insight = generate_ai_insight(
                openai_key,
                "You are a professional game market analyst.",
                prompt
            )
            st.markdown(insight)

# ================== 2. TREND PREDICTION ==================
elif page == "Trend Prediction":
    st.title("📈 Game Trend Forecasting")

    file = st.file_uploader("Upload CSV with date & rating")
    if file:
        df = pd.read_csv(file, parse_dates=["date"])
        df = df.sort_values("date")
        eq = st.selectbox("Select Metric", ["rating","ratings_count"])
        model, df_future = fit_trend(df, eq)

        st.line_chart(df.set_index("date")[eq])
        st.line_chart(df_future.set_index("date")[f"pred_{eq}"])

        if openai_key:
            if st.button("Generate AI Trend Insight"):
                prompt = f"Past:\n{df.tail().to_string()}\nForecast:\n{df_future.to_string()}"
                st.markdown(generate_ai_insight(openai_key, "You are a data scientist forecasting game trends.", prompt))

# ================== 3. SENTIMENT ANALYSIS ==================
elif page == "Sentiment Analysis":
    st.title("🗣️ Game Review Sentiment Analysis")

    mode = st.radio("Input Type", ["Manual", "Steam App ID"])

    reviews = []
    if mode == "Manual":
        text = st.text_area("Enter reviews (1 per line)")
        if st.button("Analyze"):
            reviews = text.split("\n")
    else:
        app_id = st.text_input("Steam App ID")
        if st.button("Fetch & Analyze"):
            reviews = fetch_steam_reviews(app_id)

    if reviews:
        df_sent = analyze_sentiment(reviews)
        st.dataframe(df_sent)
        st.bar_chart(df_sent["label"].value_counts())

        if openai_key:
            if st.button("Generate AI Sentiment Insight"):
                prompt = f"Sentiment Summary:\n{df_sent['label'].value_counts().to_string()}"
                st.markdown(generate_ai_insight(openai_key, "You are a UX and Game Product Analyst.", prompt))

# ================== 4. COMPETITOR ANALYSIS ==================
elif page == "Competitor Analysis":
    st.title("🏁 Steam Competitor Intelligence")

    ids = st.text_input("Enter Steam App IDs (comma separated)")
    if st.button("Analyze Competitors"):
        df = collect_competitors(ids.split(","))
        st.dataframe(df)

        st.bar_chart(df.set_index("name")["price"])

        if openai_key:
            if st.button("Generate AI Competitor Strategy"):
                prompt = f"Competitor dataset:\n{df.to_string()}"
                st.markdown(generate_ai_insight(openai_key, "You are a game business strategist.", prompt))
