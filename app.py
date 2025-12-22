import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

from google_play_scraper import search, app as gp_app, reviews, Sort

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Google Play Game Market Intelligence", layout="wide")

st.title("🎮 Google Play Game Market Intelligence (Streamlit Deploy Ready)")
st.caption("Market analysis, trend analysis, sentiment, competitor benchmarking, and auto-insights — powered by Google Play data.")

# -----------------------------
# NLTK VADER setup (download-safe)
# -----------------------------
@st.cache_resource
def get_vader():
    try:
        return SentimentIntensityAnalyzer()
    except Exception:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()

sia = get_vader()

# -----------------------------
# Helpers
# -----------------------------
def safe_parse_date(d):
    if d is None:
        return None
    if isinstance(d, (datetime, pd.Timestamp)):
        return d.date()
    if isinstance(d, date):
        return d
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None

def money_to_float(price_text):
    """
    google-play-scraper returns price as numeric sometimes, or text.
    We'll normalize to float if possible.
    """
    if price_text is None:
        return None
    if isinstance(price_text, (int, float)):
        return float(price_text)
    # Common values: 'Free', 'IDR 10.000', '$0.99'
    s = str(price_text).strip().lower()
    if s in ("free", "0", "0.0"):
        return 0.0
    # remove currency symbols and separators
    cleaned = (
        s.replace("idr", "")
         .replace("$", "")
         .replace("rp", "")
         .replace(".", "")
         .replace(",", ".")
         .strip()
    )
    try:
        return float(cleaned)
    except Exception:
        return None

def vader_label(compound: float):
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"

def rule_based_insights_market(df_apps: pd.DataFrame) -> str:
    if df_apps.empty:
        return "No data available to generate insights."

    lines = []
    # Ratings
    if "score" in df_apps.columns and df_apps["score"].notna().any():
        lines.append(f"- Average rating across results: **{df_apps['score'].mean():.2f}**.")
        top = df_apps.sort_values("score", ascending=False).head(1)
        if not top.empty:
            lines.append(f"- Highest-rated app: **{top.iloc[0]['title']}** (**{top.iloc[0]['score']:.2f}**).")

    # Installs proxy
    if "installs" in df_apps.columns and df_apps["installs"].notna().any():
        # installs is a string like "1,000,000+"
        lines.append("- Install counts are categorical (e.g., '1,000,000+'); use them as rough demand signals.")

    # Updated
    if "updated_on" in df_apps.columns and df_apps["updated_on"].notna().any():
        recent = df_apps.dropna(subset=["updated_on"]).sort_values("updated_on", ascending=False).head(3)
        if not recent.empty:
            lines.append("- Most recently updated apps (top 3): " + ", ".join([f"**{x}**" for x in recent["title"].tolist()]))

    # Price
    if "price_value" in df_apps.columns and df_apps["price_value"].notna().any():
        paid = df_apps[df_apps["price_value"] > 0]
        if len(paid) > 0:
            lines.append(f"- Paid apps in results: **{len(paid)}**. Consider whether premium pricing is viable in this niche.")
        else:
            lines.append("- Most apps appear **free-to-play** in this sample — monetisation likely depends on ads/IAP.")

    lines.append("- Recommendation: shortlist 5–10 close peers (category + mechanics), then benchmark **rating**, **installs**, and **update frequency** to identify market gaps.")
    return "\n".join(lines)

def rule_based_insights_trend(df_daily: pd.DataFrame) -> str:
    if df_daily.empty:
        return "No time-series data available to generate insights."

    lines = []
    if "reviews_count" in df_daily.columns and df_daily["reviews_count"].sum() > 0:
        lines.append(f"- Total reviews in selected window: **{int(df_daily['reviews_count'].sum())}**.")
        # trend slope
        x = np.arange(len(df_daily))
        y = df_daily["reviews_count"].values.astype(float)
        if len(df_daily) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.1:
                lines.append("- Review activity shows an **upward** trend (increasing attention).")
            elif slope < -0.1:
                lines.append("- Review activity shows a **downward** trend (cooling attention).")
            else:
                lines.append("- Review activity looks **stable** in the selected window.")

    if "avg_sentiment" in df_daily.columns and df_daily["avg_sentiment"].notna().any():
        mean_sent = df_daily["avg_sentiment"].mean()
        lines.append(f"- Average sentiment (VADER compound): **{mean_sent:.3f}**.")
        if mean_sent > 0.2:
            lines.append("- Overall sentiment is **strongly positive** — preserve core experience and focus on growth.")
        elif mean_sent < -0.05:
            lines.append("- Sentiment is **negative** — prioritize bug fixes, balance, performance, or monetization friction.")
        else:
            lines.append("- Sentiment is **mixed/neutral** — target specific pain points found in negative reviews.")

    lines.append("- Recommendation: correlate spikes in review volume with release notes / feature drops to infer what drives engagement.")
    return "\n".join(lines)

def rule_based_insights_sentiment(df_sent: pd.DataFrame) -> str:
    if df_sent.empty:
        return "No reviews available."

    counts = df_sent["label"].value_counts()
    total = int(counts.sum())
    pos = int(counts.get("positive", 0))
    neu = int(counts.get("neutral", 0))
    neg = int(counts.get("negative", 0))

    lines = [
        f"- Reviews analyzed: **{total}**",
        f"- Sentiment split: **{pos/total*100:.1f}% positive**, **{neu/total*100:.1f}% neutral**, **{neg/total*100:.1f}% negative**.",
        f"- Average sentiment score: **{df_sent['compound'].mean():.3f}**."
    ]
    if neg > pos:
        lines.append("- Negative sentiment dominates: focus on top complaint themes and ship fixes quickly.")
    elif pos > neg + max(5, int(0.2*total)):
        lines.append("- Positive sentiment dominates: optimize acquisition + retention (events, referrals, ASO).")
    else:
        lines.append("- Mixed sentiment: segment issues by device/performance vs gameplay vs ads/IAP friction.")
    return "\n".join(lines)

def rule_based_insights_competitors(df_comp: pd.DataFrame) -> str:
    if df_comp.empty:
        return "No competitor data available."
    lines = []
    if "score" in df_comp.columns and df_comp["score"].notna().any():
        best = df_comp.sort_values("score", ascending=False).head(1).iloc[0]
        lines.append(f"- Best competitor rating: **{best['title']}** (**{best['score']:.2f}**).")
    if "updated_on" in df_comp.columns and df_comp["updated_on"].notna().any():
        most_recent = df_comp.sort_values("updated_on", ascending=False).head(1).iloc[0]
        lines.append(f"- Most recently updated competitor: **{most_recent['title']}** (updated **{most_recent['updated_on']}**).")
    lines.append("- Positioning tip: compete on **update cadence**, **performance stability**, and **ASO keywords** aligned to your mechanics/genre.")
    return "\n".join(lines)

# -----------------------------
# Google Play fetch functions
# -----------------------------
@st.cache_data(show_spinner=False)
def gp_search_apps(keyword: str, country: str, lang: str, n: int = 30):
    """
    Search Play Store for apps matching keyword.
    Returns list of search dicts (contains appId/title/score/etc.).
    """
    return search(keyword, lang=lang, country=country, n_hits=n)

@st.cache_data(show_spinner=False)
def gp_get_app_details(app_id: str, country: str, lang: str):
    """
    Get app details, includes updated date, installs, score, priceText, genre, etc.
    """
    return gp_app(app_id, lang=lang, country=country)

@st.cache_data(show_spinner=False)
def gp_get_reviews(app_id: str, country: str, lang: str, count: int, sort: str):
    """
    Fetch reviews for app_id.
    google-play-scraper returns (reviews_list, token).
    """
    sort_mode = Sort.NEWEST if sort == "NEWEST" else Sort.MOST_RELEVANT
    all_reviews = []
    token = None
    remaining = count

    # Pull in chunks (library typically paginates)
    while remaining > 0:
        batch = min(200, remaining)
        r, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort_mode,
            count=batch,
            continuation_token=token
        )
        if not r:
            break
        all_reviews.extend(r)
        remaining -= len(r)
        if token is None:
            break

    return all_reviews

def build_market_df_from_search(search_rows, country: str, lang: str):
    """
    Enrich search results with app details (updated date, installs, price etc.)
    """
    records = []
    for row in search_rows:
        app_id = row.get("appId")
        if not app_id:
            continue
        try:
            d = gp_get_app_details(app_id, country, lang)
        except Exception:
            continue

        updated_on = safe_parse_date(d.get("updated"))
        price_val = money_to_float(d.get("priceText"))

        records.append({
            "appId": app_id,
            "title": d.get("title"),
            "developer": d.get("developer"),
            "genre": d.get("genre"),
            "score": d.get("score"),
            "ratings": d.get("ratings"),
            "reviews": d.get("reviews"),
            "installs": d.get("installs"),
            "priceText": d.get("priceText"),
            "price_value": price_val,
            "free": d.get("free"),
            "updated_on": updated_on,
            "url": d.get("url"),
        })

    df = pd.DataFrame(records)
    if not df.empty and "updated_on" in df.columns:
        df = df.sort_values("updated_on", ascending=False)
    return df

def filter_by_date(df: pd.DataFrame, col: str, start: date | None, end: date | None):
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.date
    if start:
        out = out[out[col] >= start]
    if end:
        out = out[out[col] <= end]
    return out

def sentiment_from_reviews(rev_rows):
    rows = []
    for r in rev_rows:
        text = (r.get("content") or "").strip()
        if not text:
            continue
        comp = sia.polarity_scores(text)["compound"]
        rows.append({
            "at": pd.to_datetime(r.get("at"), errors="coerce"),
            "score": r.get("score"),
            "text": text,
            "compound": comp,
            "label": vader_label(comp)
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("at")
    return df

def daily_trend(df_sent: pd.DataFrame, start: date | None, end: date | None):
    if df_sent.empty:
        return pd.DataFrame()

    df = df_sent.copy()
    df["day"] = df["at"].dt.date

    if start:
        df = df[df["day"] >= start]
    if end:
        df = df[df["day"] <= end]

    if df.empty:
        return pd.DataFrame()

    daily = df.groupby("day").agg(
        reviews_count=("text", "count"),
        avg_sentiment=("compound", "mean"),
        avg_review_score=("score", "mean"),
    ).reset_index()

    daily["avg_sentiment"] = daily["avg_sentiment"].astype(float)
    daily["avg_review_score"] = daily["avg_review_score"].astype(float)
    return daily

# -----------------------------
# Sidebar: shared filters (sync Market + Trend)
# -----------------------------
st.sidebar.header("Global Filters (Sync Market + Trend)")
country = st.sidebar.selectbox("Country", ["us", "id", "sg", "gb", "au"], index=1)
lang = st.sidebar.selectbox("Language", ["en", "id"], index=0)

start_date = st.sidebar.date_input("Start date", value=None)
end_date = st.sidebar.date_input("End date", value=None)

# Ensure proper types
start_date = safe_parse_date(start_date)
end_date = safe_parse_date(end_date)

if start_date and end_date and start_date > end_date:
    st.sidebar.warning("Start date is after end date. The date filter will behave unexpectedly.")

module = st.sidebar.radio(
    "Module",
    ["Market Analysis", "Trend Analysis", "Sentiment Analysis", "Competitor Analysis"],
)

# -----------------------------
# Market Analysis
# -----------------------------
if module == "Market Analysis":
    st.subheader("📊 Market Analysis (Google Play Search + App Details)")

    keyword = st.text_input("Keyword (e.g., 'MOBA', 'battle royale', 'idle RPG')", value="moba")
    n_hits = st.slider("Search result size", min_value=10, max_value=80, value=30, step=10)

    ordering = st.selectbox("Sort by (client-side)", ["updated_on desc", "score desc", "ratings desc", "reviews desc"])

    if st.button("Fetch Market Data"):
        with st.spinner("Searching Google Play and enriching app details..."):
            rows = gp_search_apps(keyword, country=country, lang=lang, n=n_hits)
            df_market = build_market_df_from_search(rows, country=country, lang=lang)

        if df_market.empty:
            st.error("No results. Try a different keyword.")
            st.stop()

        # Apply date filter by UPDATED date (sync filter)
        df_filtered = filter_by_date(df_market, "updated_on", start_date, end_date)

        # Ordering
        if ordering == "score desc":
            df_filtered = df_filtered.sort_values("score", ascending=False, na_position="last")
        elif ordering == "ratings desc":
            df_filtered = df_filtered.sort_values("ratings", ascending=False, na_position="last")
        elif ordering == "reviews desc":
            df_filtered = df_filtered.sort_values("reviews", ascending=False, na_position="last")
        else:
            df_filtered = df_filtered.sort_values("updated_on", ascending=False, na_position="last")

        st.markdown(
            f"**Date filter (Updated on):** {start_date or 'none'} → {end_date or 'none'}"
        )

        st.dataframe(
            df_filtered[[
                "title", "developer", "genre", "score", "ratings", "reviews", "installs", "priceText", "updated_on", "appId"
            ]],
            use_container_width=True
        )

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Top 10 by Rating")
            top = df_filtered.dropna(subset=["score"]).sort_values("score", ascending=False).head(10)
            if not top.empty:
                fig, ax = plt.subplots()
                ax.barh(top["title"][::-1], top["score"][::-1])
                ax.set_xlabel("Rating")
                ax.set_title("Top 10 Apps by Rating")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough rating data for chart.")

        with c2:
            st.markdown("#### Genres Distribution (Top 10)")
            g = df_filtered["genre"].dropna().value_counts().head(10)
            if not g.empty:
                fig, ax = plt.subplots()
                ax.bar(g.index, g.values)
                ax.set_ylabel("Count")
                ax.set_title("Genre Distribution")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough genre data for chart.")

        st.markdown("---")
        st.subheader("🧠 Auto Insight (No API, Rule-Based)")
        if st.button("Generate Market Insight"):
            st.markdown(rule_based_insights_market(df_filtered))

# -----------------------------
# Trend Analysis
# -----------------------------
elif module == "Trend Analysis":
    st.subheader("📈 Trend Analysis (Reviews Over Time)")

    st.write("This module builds trends from **review timestamps**. Date filter is synced from sidebar.")
    query = st.text_input("Game name keyword to search", value="mobile legends")
    limit = st.slider("Search results to show", 5, 20, 10)

    if st.button("Search games"):
        with st.spinner("Searching Google Play..."):
            rows = gp_search_apps(query, country=country, lang=lang, n=limit)
        df_s = pd.DataFrame([{"title": r.get("title"), "appId": r.get("appId")} for r in rows if r.get("appId")])
        if df_s.empty:
            st.error("No matches.")
            st.stop()
        st.session_state["trend_search_df"] = df_s

    df_s = st.session_state.get("trend_search_df")
    if df_s is not None and not df_s.empty:
        choice = st.selectbox("Select app", [f"{r.title} ({r.appId})" for r in df_s.itertuples()])
        selected_appid = choice.split("(")[-1].replace(")", "").strip()

        review_count = st.slider("Reviews to fetch (max)", 200, 2000, 800, step=200)
        sort = st.selectbox("Review sort", ["NEWEST", "MOST_RELEVANT"], index=0)

        if st.button("Fetch reviews and build trend"):
            with st.spinner("Fetching reviews and computing sentiment..."):
                rev = gp_get_reviews(selected_appid, country=country, lang=lang, count=review_count, sort=sort)
                df_sent = sentiment_from_reviews(rev)

            if df_sent.empty:
                st.error("No reviews retrieved. Try increasing count or using NEWEST.")
                st.stop()

            # Filter by synced date range
            df_daily = daily_trend(df_sent, start_date, end_date)
            if df_daily.empty:
                st.warning("No reviews fall within the selected date range.")
                st.stop()

            st.markdown(f"**Date filter (Review date):** {start_date or 'none'} → {end_date or 'none'}")

            st.subheader("Daily Trend Table")
            st.dataframe(df_daily, use_container_width=True)

            # Plot: reviews count
            fig1, ax1 = plt.subplots()
            ax1.plot(df_daily["day"], df_daily["reviews_count"], marker="o")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Review count")
            ax1.set_title("Review Volume Over Time")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig1)

            # Plot: sentiment
            fig2, ax2 = plt.subplots()
            ax2.plot(df_daily["day"], df_daily["avg_sentiment"], marker="o")
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Avg sentiment (compound)")
            ax2.set_title("Average Sentiment Over Time")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("---")
            st.subheader("🧠 Auto Insight (No API, Rule-Based)")
            if st.button("Generate Trend Insight"):
                st.markdown(rule_based_insights_trend(df_daily))
    else:
        st.info("Search games first, then select an app to build trends.")

# -----------------------------
# Sentiment Analysis
# -----------------------------
elif module == "Sentiment Analysis":
    st.subheader("🗣️ Sentiment Analysis (Google Play Reviews)")

    mode = st.radio("Input mode", ["Search by Game Name", "Use App ID"], index=0)

    selected_appid = None

    if mode == "Use App ID":
        selected_appid = st.text_input("Google Play App ID (package name)", value="")
    else:
        query = st.text_input("Game name keyword", value="pubg")
        limit = st.slider("Search results to show", 5, 20, 10)
        if st.button("Search"):
            with st.spinner("Searching Google Play..."):
                rows = gp_search_apps(query, country=country, lang=lang, n=limit)
            df_s = pd.DataFrame([{"title": r.get("title"), "appId": r.get("appId")} for r in rows if r.get("appId")])
            if df_s.empty:
                st.error("No matches.")
                st.stop()
            st.session_state["sent_search_df"] = df_s

        df_s = st.session_state.get("sent_search_df")
        if df_s is not None and not df_s.empty:
            choice = st.selectbox("Select app", [f"{r.title} ({r.appId})" for r in df_s.itertuples()])
            selected_appid = choice.split("(")[-1].replace(")", "").strip()

    review_count = st.slider("Reviews to fetch (max)", 100, 2000, 600, step=100)
    sort = st.selectbox("Review sort", ["NEWEST", "MOST_RELEVANT"], index=0)

    if st.button("Fetch reviews and analyse sentiment"):
        if not selected_appid or not selected_appid.strip():
            st.warning("Please select or enter an App ID.")
            st.stop()

        with st.spinner("Fetching reviews and scoring sentiment..."):
            rev = gp_get_reviews(selected_appid.strip(), country=country, lang=lang, count=review_count, sort=sort)
            df_sent = sentiment_from_reviews(rev)

        if df_sent.empty:
            st.error("No reviews retrieved.")
            st.stop()

        # Apply synced date range on review date
        if start_date:
            df_sent = df_sent[df_sent["at"].dt.date >= start_date]
        if end_date:
            df_sent = df_sent[df_sent["at"].dt.date <= end_date]

        if df_sent.empty:
            st.warning("No reviews fall within the selected date range.")
            st.stop()

        st.markdown(f"**Date filter (Review date):** {start_date or 'none'} → {end_date or 'none'}")

        st.subheader("Review Sentiment Table")
        st.dataframe(df_sent[["at", "score", "label", "compound", "text"]].head(200), use_container_width=True)

        counts = df_sent["label"].value_counts()
        st.subheader("Sentiment Distribution")
        st.write(counts)

        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        ax.set_title("Sentiment Split")
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("🧠 Auto Insight (No API, Rule-Based)")
        if st.button("Generate Sentiment Insight"):
            st.markdown(rule_based_insights_sentiment(df_sent))

# -----------------------------
# Competitor Analysis
# -----------------------------
elif module == "Competitor Analysis":
    st.subheader("🏁 Competitor Analysis (Similar Apps + Benchmarking)")

    mode = st.radio("Select base game by", ["Search by Game Name", "Use App ID"], index=0)

    base_appid = None

    if mode == "Use App ID":
        base_appid = st.text_input("Base App ID (package name)", value="")
    else:
        query = st.text_input("Base game name keyword", value="mobile legends")
        limit = st.slider("Search results to show", 5, 20, 10)
        if st.button("Search base game"):
            with st.spinner("Searching Google Play..."):
                rows = gp_search_apps(query, country=country, lang=lang, n=limit)
            df_s = pd.DataFrame([{"title": r.get("title"), "appId": r.get("appId")} for r in rows if r.get("appId")])
            if df_s.empty:
                st.error("No matches.")
                st.stop()
            st.session_state["comp_search_df"] = df_s

        df_s = st.session_state.get("comp_search_df")
        if df_s is not None and not df_s.empty:
            choice = st.selectbox("Select base game", [f"{r.title} ({r.appId})" for r in df_s.itertuples()])
            base_appid = choice.split("(")[-1].replace(")", "").strip()

    top_n_similar = st.slider("Number of similar apps (competitors) to benchmark", 5, 30, 15)

    if st.button("Fetch competitors"):
        if not base_appid or not base_appid.strip():
            st.warning("Please provide a base App ID.")
            st.stop()

        with st.spinner("Fetching base app details + similar apps..."):
            base = gp_get_app_details(base_appid.strip(), country=country, lang=lang)
            similar = (base.get("similarApps") or [])[:top_n_similar]

        if not similar:
            st.error("No similar apps returned for this base game.")
            st.stop()

        # Build competitor dataframe
        records = []
        for sim in similar:
            app_id = sim.get("appId")
            if not app_id:
                continue
            try:
                d = gp_get_app_details(app_id, country=country, lang=lang)
            except Exception:
                continue
            records.append({
                "appId": app_id,
                "title": d.get("title"),
                "developer": d.get("developer"),
                "genre": d.get("genre"),
                "score": d.get("score"),
                "ratings": d.get("ratings"),
                "reviews": d.get("reviews"),
                "installs": d.get("installs"),
                "priceText": d.get("priceText"),
                "price_value": money_to_float(d.get("priceText")),
                "free": d.get("free"),
                "updated_on": safe_parse_date(d.get("updated")),
            })

        df_comp = pd.DataFrame(records)
        if df_comp.empty:
            st.error("Failed to build competitor dataset.")
            st.stop()

        # Apply synced date filter on updated date
        df_comp_f = filter_by_date(df_comp, "updated_on", start_date, end_date)

        st.markdown(f"**Date filter (Updated on):** {start_date or 'none'} → {end_date or 'none'}")
        st.subheader("Competitor Benchmark Table (Similar Apps)")
        st.dataframe(df_comp_f.sort_values("score", ascending=False, na_position="last"), use_container_width=True)

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Rating Comparison (Top 10)")
            top = df_comp_f.dropna(subset=["score"]).sort_values("score", ascending=False).head(10)
            if not top.empty:
                fig, ax = plt.subplots()
                ax.barh(top["title"][::-1], top["score"][::-1])
                ax.set_xlabel("Rating")
                ax.set_title("Top Competitors by Rating")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No rating data to plot.")

        with c2:
            st.markdown("#### Update Recency (Top 10 latest)")
            tmp = df_comp_f.dropna(subset=["updated_on"]).sort_values("updated_on", ascending=False).head(10)
            if not tmp.empty:
                fig, ax = plt.subplots()
                # Plot days since updated
                today = date.today()
                days = [(today - d).days for d in tmp["updated_on"]]
                ax.bar(tmp["title"], days)
                ax.set_ylabel("Days since update")
                ax.set_title("Update Recency (lower is better)")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No update date data to plot.")

        st.markdown("---")
        st.subheader("🧠 Auto Insight (No API, Rule-Based)")
        if st.button("Generate Competitor Insight"):
            st.markdown(rule_based_insights_competitors(df_comp_f))
