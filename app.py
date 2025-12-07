import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from typing import List, Optional
import math

# ================== STREAMLIT CONFIG ==================

st.set_page_config(
    page_title="Game Market Intelligence",
    layout="wide",
)

# ================== NLP / SENTIMENT SETUP ==================

try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk_download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

RAWG_BASE_URL = "https://api.rawg.io/api"

# ================== RAWG FETCH IMPLEMENTATION (WITH PAGINATION + DATES) ==================

def _fetch_trending_games_impl(
    rawg_api_key: str,
    total_size: int = 400,
    ordering: str = "-added",
    dates_param: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch trending or popular games from RAWG with pagination.
    total_size is the total number of games desired (max capped internally for safety).
    ordering options: -added, -rating, -metacritic, -released, etc.
    dates_param: "YYYY-MM-DD,YYYY-MM-DD" for RAWG date range filter.
    """
    if not rawg_api_key:
        return pd.DataFrame()

    # Safety cap to avoid extremely large loads
    max_total = 1000
    total_size = max(1, min(int(total_size), max_total))

    per_page = 40  # RAWG maximum page_size
    n_pages = math.ceil(total_size / per_page)

    all_records = []

    for page in range(1, n_pages + 1):
        page_size = min(per_page, total_size - len(all_records))
        if page_size <= 0:
            break

        endpoint = f"{RAWG_BASE_URL}/games"
        params = {
            "key": rawg_api_key,
            "page_size": page_size,
            "ordering": ordering,
            "page": page,
        }
        if dates_param:
            params["dates"] = dates_param  # RAWG date range filter

        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("results", [])

        for g in games:
            all_records.append(
                {
                    "id": g.get("id"),
                    "slug": g.get("slug"),
                    "name": g.get("name"),
                    "released": g.get("released"),
                    "rating": g.get("rating"),
                    "ratings_count": g.get("ratings_count"),
                    "metacritic": g.get("metacritic"),
                    "playtime": g.get("playtime"),
                    "suggestions_count": g.get("suggestions_count"),
                    "reviews_count": g.get("reviews_text_count"),
                    "genres": ", ".join([x["name"] for x in g.get("genres", [])]),
                    "platforms": ", ".join(
                        [p["platform"]["name"] for p in (g.get("platforms") or [])]
                    )
                    if g.get("platforms")
                    else None,
                    "stores": ", ".join(
                        [s["store"]["name"] for s in (g.get("stores") or [])]
                    )
                    if g.get("stores")
                    else None,
                }
            )

        if len(all_records) >= total_size:
            break

    return pd.DataFrame.from_records(all_records)


@st.cache_data(show_spinner=False)
def fetch_trending_games_cached(
    rawg_api_key: str,
    ordering: str,
    dates_param: Optional[str],
) -> pd.DataFrame:
    """
    Cached version for Market Overview, driven by ordering + date range.
    """
    # Internally we cap total_size (e.g., 400 games max)
    return _fetch_trending_games_impl(
        rawg_api_key,
        total_size=400,
        ordering=ordering,
        dates_param=dates_param,
    )


def fetch_trending_games_live(
    rawg_api_key: str,
    total_size: int = 100,
    ordering: str = "-added",
) -> pd.DataFrame:
    """
    Non-cached version for live snapshots (Trend Prediction),
    still based on count (snapshot size) not release date.
    """
    return _fetch_trending_games_impl(
        rawg_api_key,
        total_size=total_size,
        ordering=ordering,
        dates_param=None,
    )


# ================== STEAM HELPERS ==================

def fetch_steam_reviews(app_id: str, num: int = 100) -> List[str]:
    """
    Fetch reviews from Steam's public review endpoint.
    NOTE: Respect Steam's ToS and avoid heavy scraping.
    """
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "language": "english",
        "purchase_type": "all",
        "review_type": "all",
        "num_per_page": min(num, 100),
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    reviews = data.get("reviews", [])[:num]
    return [r.get("review", "") for r in reviews]


def fetch_steam_appdetails(app_id: str) -> Optional[dict]:
    """
    Fetch game details from Steam (price, genres, recommendations, etc.).
    Wishlist count is NOT available via public API, so it's not included.
    """
    url = "https://store.steampowered.com/api/appdetails"
    params = {"appids": app_id}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data or not data.get(app_id, {}).get("success"):
        return None

    d = data[app_id]["data"]

    price_info = d.get("price_overview")
    final_price = None
    currency = None
    discount = None
    if price_info:
        final_price = price_info.get("final")  # in cents
        currency = price_info.get("currency")
        discount = price_info.get("discount_percent")

    genres = ", ".join([g["description"] for g in d.get("genres", [])])
    categories = ", ".join([c["description"] for c in d.get("categories", [])])

    info = {
        "steam_appid": d.get("steam_appid"),
        "name": d.get("name"),
        "is_free": d.get("is_free"),
        "price_final_cent": final_price,
        "currency": currency,
        "discount_percent": discount,
        "metacritic_score": (d.get("metacritic") or {}).get("score"),
        "recommendations": (d.get("recommendations") or {}).get("total"),
        "genres": genres,
        "categories": categories,
    }
    return info


def collect_competitor_data(app_ids: List[str]) -> pd.DataFrame:
    rows = []
    for aid in app_ids:
        aid = aid.strip()
        if not aid:
            continue
        try:
            info = fetch_steam_appdetails(aid)
        except Exception:
            info = None
        if info:
            rows.append(info)
    return pd.DataFrame(rows)


# ================== CORE ANALYTICS FUNCTIONS ==================

def fit_linear_trend(df: pd.DataFrame, time_col: str, target_col: str):
    """
    Fit a simple linear trend: target ~ time (as numeric index).
    Returns: model, df_future (5 future points).
    """
    df = df.dropna(subset=[time_col, target_col]).copy()
    if df.empty:
        return None, None

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    if df[time_col].nunique() < 2:
        # Not enough variation in time
        return None, None

    df["t_numeric"] = np.arange(len(df))
    X = df[["t_numeric"]].values
    y = df[target_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Create 5 future time points (1 day apart)
    last_time = df[time_col].max()
    future_times = [last_time + timedelta(days=i) for i in range(1, 6)]
    future_idx = np.arange(len(df), len(df) + 5).reshape(-1, 1)
    future_pred = model.predict(future_idx)

    df_future = pd.DataFrame(
        {
            time_col: future_times,
            f"pred_{target_col}": future_pred,
        }
    )
    return model, df_future


def analyze_sentiment(texts: List[str]) -> pd.DataFrame:
    """
    Run VADER sentiment analysis over a list of review texts.
    """
    rows = []
    for t in texts:
        t_clean = t.strip()
        if not t_clean:
            continue
        scores = sia.polarity_scores(t_clean)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        rows.append(
            {
                "text": t_clean,
                "compound": compound,
                "neg": scores["neg"],
                "neu": scores["neu"],
                "pos": scores["pos"],
                "label": label,
            }
        )
    return pd.DataFrame(rows)


# ================== LOCAL RULE-BASED INSIGHT FUNCTIONS ==================

def generate_market_overview_insight(
    df_games: pd.DataFrame,
    top_rating: pd.DataFrame,
    genre_counts: pd.Series,
    platform_counts: pd.Series,
) -> str:
    if df_games.empty:
        return "No game data available to generate insights."

    lines = []

    # Basic rating stats
    if "rating" in df_games.columns:
        avg_rating = df_games["rating"].mean()
        median_rating = df_games["rating"].median()
        lines.append(
            f"- Average rating across the sampled games is **{avg_rating:.2f}** "
            f"(median: **{median_rating:.2f}**)."
        )

    # Genres
    if not genre_counts.empty:
        top_genre = genre_counts.index[0]
        top_genre_share = genre_counts.iloc[0] / genre_counts.sum() * 100
        lines.append(
            f"- The most represented genre is **{top_genre}**, accounting for ~**{top_genre_share:.1f}%** "
            "of the sampled titles."
        )
        if len(genre_counts) > 1:
            second_genre = genre_counts.index[1]
            lines.append(
                f"- Other strong genres in the current snapshot include **{second_genre}** and several niche categories."
            )

    # Platforms
    if not platform_counts.empty:
        top_platform = platform_counts.index[0]
        lines.append(
            f"- From a platform perspective, **{top_platform}** appears most frequently, indicating strong support or visibility on that platform."
        )

    # Top game detail
    if not top_rating.empty:
        top_game = top_rating.iloc[0]
        ratings_count = int(top_game.get("ratings_count") or 0)
        lines.append(
            f"- The highest-rated title in this snapshot is **{top_game['name']}** with a rating of "
            f"**{top_game['rating']:.2f}** based on **{ratings_count}** ratings."
        )

    # Opportunity hint
    if not genre_counts.empty and not platform_counts.empty:
        lines.append(
            "- Dominant genres and platforms validate player demand, but there may also be opportunities in "
            "underserved genres or alternative platforms if they align with your studio’s strengths."
        )

    return "\n".join(lines) or "Not enough information to generate a market overview insight."


def generate_trend_insight(df_game: pd.DataFrame, df_future: pd.DataFrame, metric: str) -> str:
    if df_game.empty or df_future is None or df_future.empty:
        return "Not enough data to generate a trend insight."

    series_hist = df_game[metric].dropna()
    if series_hist.empty:
        return "Historical values for the selected metric are missing."

    start_val = series_hist.iloc[0]
    end_val = series_hist.iloc[-1]
    delta = end_val - start_val
    n_points = len(series_hist)
    slope = delta / max(n_points - 1, 1)

    if slope > 0.01:
        direction = "an upward (improving)"
    elif slope < -0.01:
        direction = "a downward (declining)"
    else:
        direction = "a relatively flat/stable"

    min_val = series_hist.min()
    max_val = series_hist.max()

    future_vals = df_future[f"pred_{metric}"]
    future_min = future_vals.min()
    future_max = future_vals.max()

    lines = [
        f"- Historically, **{metric}** started at **{start_val:.2f}** and moved to **{end_val:.2f}**, "
        f"indicating {direction} trend over the observed period.",
        f"- The observed range so far is **{min_val:.2f} – {max_val:.2f}**.",
        f"- The simple linear forecast suggests future values in the range of approximately "
        f"**{future_min:.2f} – {future_max:.2f}** for the next few time steps.",
    ]

    if metric == "rating":
        lines.append(
            "- Rating trends typically reflect perceived quality and stability. If the trend is declining, "
            "it is worth reviewing recent patches, balance changes, or monetisation decisions that may be "
            "impacting player satisfaction."
        )
    else:
        lines.append(
            "- Changes in ratings count can act as a proxy for visibility and engagement. Sustained growth may "
            "indicate effective marketing, featuring, or word-of-mouth, while stagnation suggests the need "
            "for refreshed campaigns or in-game events."
        )

    return "\n".join(lines)


def generate_sentiment_insight(df_sent: pd.DataFrame) -> str:
    if df_sent.empty:
        return "No reviews available to analyse."

    counts = df_sent["label"].value_counts()
    total = counts.sum()
    pos = counts.get("positive", 0)
    neu = counts.get("neutral", 0)
    neg = counts.get("negative", 0)

    pos_pct = pos / total * 100
    neu_pct = neu / total * 100
    neg_pct = neg / total * 100

    lines = [
        f"- Total reviews analysed: **{total}**.",
        f"- Sentiment breakdown: **{pos_pct:.1f}% positive**, **{neu_pct:.1f}% neutral**, **{neg_pct:.1f}% negative**.",
    ]

    if pos_pct > neg_pct + 20:
        lines.append(
            "- Overall sentiment is clearly positive, suggesting that the core game experience resonates well with players."
        )
    elif neg_pct > pos_pct:
        lines.append(
            "- Negative sentiment outweighs positive feedback, indicating that key pain points should be prioritised "
            "for investigation and resolution."
        )
    else:
        lines.append(
            "- Sentiment is mixed. There are notable strengths, but also recurring issues that require targeted follow-up."
        )

    avg_compound = df_sent["compound"].mean()
    lines.append(
        f"- The average VADER compound score is **{avg_compound:.3f}**, on a scale from -1 (very negative) to +1 (very positive)."
    )

    return "\n".join(lines)


def generate_competitor_insight(df_comp: pd.DataFrame) -> str:
    if df_comp.empty:
        return "No competitor data available to generate insights."

    lines = []

    # Free vs premium breakdown
    is_free_counts = df_comp["is_free"].value_counts(dropna=False)
    num_free = is_free_counts.get(True, 0)
    num_paid = is_free_counts.get(False, 0)
    lines.append(
        f"- Competitors analysed: **{len(df_comp)}** "
        f"(Free-to-play: **{num_free}**, Premium: **{num_paid}**)."
    )

    # Price analysis
    if "price_final" in df_comp.columns:
        paid_games = df_comp[df_comp["is_free"] == False]
        if not paid_games.empty:
            avg_price = paid_games["price_final"].mean()
            lines.append(
                f"- Among premium titles, the average listed price is approximately **{avg_price:.2f}** (in the respective currency)."
            )

            highest = paid_games.sort_values("price_final", ascending=False).iloc[0]
            lowest = paid_games.sort_values("price_final", ascending=True).iloc[0]
            lines.append(
                f"- The highest-priced premium title in this set is **{highest['name']}** (~{highest['price_final']:.2f}), "
                f"while **{lowest['name']}** is positioned as the most affordable (~{lowest['price_final']:.2f})."
            )

    # Metacritic and recommendations
    if "metacritic_score" in df_comp.columns:
        valid_meta = df_comp.dropna(subset=["metacritic_score"])
        if not valid_meta.empty:
            best_meta = valid_meta.sort_values("metacritic_score", ascending=False).iloc[0]
            lines.append(
                f"- From a critical perspective, **{best_meta['name']}** leads with a Metacritic score of "
                f"**{best_meta['metacritic_score']}**."
            )

    if "recommendations" in df_comp.columns:
        valid_rec = df_comp.dropna(subset=["recommendations"])
        if not valid_rec.empty:
            best_rec = valid_rec.sort_values("recommendations", ascending=False).iloc[0]
            lines.append(
                f"- In terms of player recommendations, **{best_rec['name']}** is the strongest performer with "
                f"**{int(best_rec['recommendations'])}** total recommendations."
            )

    # Genre landscape
    if "genres" in df_comp.columns:
        genre_series = df_comp["genres"].dropna().str.split(", ").explode()
        if not genre_series.empty:
            genre_counts = genre_series.value_counts()
            top_genre = genre_counts.index[0]
            lines.append(
                f"- Genre-wise, **{top_genre}** appears most frequently among competitors, which indicates a validated but "
                "potentially crowded space."
            )

    lines.append(
        "- To stand out, consider either a differentiated subgenre, a distinct business model (e.g., premium vs F2P), or "
        "stronger execution in areas where leading competitors already perform well (quality, live-ops, community support)."
    )

    return "\n".join(lines)


# ================== SESSION STATE FOR SNAPSHOTS ==================

if "trend_snapshots" not in st.session_state:
    st.session_state["trend_snapshots"] = pd.DataFrame()


# ================== SIDEBAR CONFIG ==================

st.sidebar.title("Configuration")

rawg_api_key = st.sidebar.text_input(
    "RAWG API Key",
    type="password",
    help="Get an API key from https://rawg.io/ and paste it here.",
)

page = st.sidebar.radio(
    "Select Module",
    [
        "1️⃣ Market Overview (RAWG)",
        "2️⃣ Trend Prediction (Snapshots from RAWG)",
        "3️⃣ Sentiment Analysis (Reviews)",
        "4️⃣ Competitor Analysis (Steam)",
    ],
)


# ================== PAGE 1: MARKET OVERVIEW ==================

if page == "1️⃣ Market Overview (RAWG)":
    st.title("📊 Market Overview - RAWG Trending/Popular Games")

    ordering = st.selectbox(
        "Ordering",
        options=["-added", "-rating", "-metacritic", "-released"],
        help="-added: newly added / popular\n-rating: highest rating\n-metacritic: critic score\n-released: latest releases",
    )

    st.markdown("### Release Date Filter (used directly in RAWG API)")

    start_str = st.text_input("Start release date (YYYY-MM-DD, optional)", "")
    end_str = st.text_input("End release date (YYYY-MM-DD, optional)", "")

    dates_param = None
    # If both dates are provided and valid, we use them in RAWG query
    if start_str and end_str:
        try:
            # just to validate format
            start_date = pd.to_datetime(start_str).date()
            end_date = pd.to_datetime(end_str).date()
            if start_date <= end_date:
                dates_param = f"{start_date.isoformat()},{end_date.isoformat()}"
            else:
                st.warning("Start date is after end date. RAWG date filter will be ignored.")
        except Exception:
            st.warning("Invalid date format. Use YYYY-MM-DD. RAWG date filter will be ignored.")

    if not rawg_api_key:
        st.warning("Please provide a RAWG API key in the sidebar.")
        st.stop()

    with st.spinner("Fetching data from RAWG (filtered by dates if provided)..."):
        df_games = fetch_trending_games_cached(rawg_api_key, ordering=ordering, dates_param=dates_param)

    if df_games.empty:
        st.error("No data returned. Check your API key, connection, or try adjusting the date range.")
        st.stop()

    # Prepare datetime column for local filtering/visualisation
    df_games["released_dt"] = pd.to_datetime(df_games["released"], errors="coerce")
    df_view = df_games.copy()

    # Optionally also apply client-side filter if user provided start/end (works even when RAWG filter is ignored)
    try:
        if start_str:
            start_date_local = pd.to_datetime(start_str).date()
            df_view = df_view[df_view["released_dt"].dt.date >= start_date_local]
        if end_str:
            end_date_local = pd.to_datetime(end_str).date()
            df_view = df_view[df_view["released_dt"].dt.date <= end_date_local]
    except Exception:
        st.warning("Local release-date filter failed due to invalid format. Visuals use unfiltered dates.")

    st.subheader("Game Table (Filtered)")
    st.dataframe(df_view)

    if df_view.empty:
        st.warning("No games match the selected date range (after local filtering).")
        st.stop()

    # Top games by rating (after filters)
    st.subheader("Top 10 Games by Rating (Filtered)")
    top_rating = df_view.sort_values("rating", ascending=False).head(10)
    st.dataframe(top_rating[["name", "rating", "ratings_count", "metacritic", "genres", "platforms"]])

    # Rating bar chart
    fig1, ax1 = plt.subplots()
    top_plot = top_rating.sort_values("rating", ascending=True)
    ax1.barh(top_plot["name"], top_plot["rating"])
    ax1.set_xlabel("Rating")
    ax1.set_title("Top 10 Games by Rating (Filtered)")
    plt.tight_layout()
    st.pyplot(fig1)

    # Genre distribution
    st.subheader("Genre Distribution (Filtered Sample)")
    genres_exploded = df_view["genres"].dropna().str.split(", ").explode()
    genre_counts = genres_exploded.value_counts().head(15)
    st.write(genre_counts)

    fig2, ax2 = plt.subplots()
    ax2.bar(genre_counts.index, genre_counts.values)
    ax2.set_xlabel("Genre")
    ax2.set_ylabel("Count")
    ax2.set_title("Top Genres in Filtered Snapshot")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)

    # Platform distribution
    st.subheader("Platform Distribution (Filtered)")
    plat_exploded = df_view["platforms"].dropna().str.split(", ").explode()
    plat_counts = plat_exploded.value_counts().head(15)
    st.write(plat_counts)

    fig3, ax3 = plt.subplots()
    ax3.bar(plat_counts.index, plat_counts.values)
    ax3.set_xlabel("Platform")
    ax3.set_ylabel("Count")
    ax3.set_title("Top Platforms in Filtered Snapshot")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)

    # Local insight
    st.markdown("---")
    st.subheader("Automatic Market Insight (Rule-Based, No External AI)")

    if st.button("Generate Market Insight"):
        insight_text = generate_market_overview_insight(df_view, top_rating, genre_counts, plat_counts)
        st.markdown(insight_text)


# ================== PAGE 2: TREND PREDICTION (SNAPSHOTS FROM RAWG) ==================

elif page == "2️⃣ Trend Prediction (Snapshots from RAWG)":
    st.title("📈 Trend Prediction from RAWG Snapshots (No Upload)")

    st.markdown(
        """
        This module uses **RAWG snapshots stored in session state**.
        
        Workflow:
        1. Define the number of games and ordering.
        2. Click **“Capture new snapshot from RAWG”** multiple times over time (e.g., different days or hours).
        3. Use the **user-input date filter** to focus on a time window.
        4. Select a game and metric, then view trend + forecast.
        """
    )

    if not rawg_api_key:
        st.warning("Please provide a RAWG API key in the sidebar.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        snap_size = st.number_input(
            "Number of games per snapshot",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Each snapshot will fetch this many games from RAWG."
        )
    with col2:
        snap_ordering = st.selectbox(
            "Snapshot ordering",
            options=["-added", "-rating", "-metacritic", "-released"],
            help="Same as Market Overview ordering."
        )

    if st.button("Capture new snapshot from RAWG"):
        with st.spinner("Capturing snapshot from RAWG..."):
            df_snap = fetch_trending_games_live(rawg_api_key, total_size=int(snap_size), ordering=snap_ordering)
        if df_snap.empty:
            st.error("Snapshot fetch returned no data. Check API key or connection.")
        else:
            df_snap = df_snap.copy()
            df_snap["snapshot_time_utc"] = datetime.utcnow()
            if st.session_state["trend_snapshots"].empty:
                st.session_state["trend_snapshots"] = df_snap
            else:
                st.session_state["trend_snapshots"] = pd.concat(
                    [st.session_state["trend_snapshots"], df_snap],
                    ignore_index=True
                )
            st.success(f"Snapshot captured with {len(df_snap)} rows at {df_snap['snapshot_time_utc'].iloc[0]} (UTC).")

    df_rt = st.session_state["trend_snapshots"]

    if df_rt.empty:
        st.info("No snapshots captured yet. Use the button above to take the first snapshot.")
        st.stop()

    st.subheader("Current Snapshot Dataset (from session)")
    st.write(f"Total rows: {len(df_rt)}, Distinct snapshots: {df_rt['snapshot_time_utc'].nunique()}")
    st.dataframe(df_rt.head())

    required_cols = {"name", "snapshot_time_utc", "rating", "ratings_count"}
    if not required_cols.issubset(df_rt.columns):
        st.error(f"Snapshot data is missing required columns: {required_cols}")
        st.stop()

    # Ensure datetime
    df_rt["snapshot_time_utc"] = pd.to_datetime(df_rt["snapshot_time_utc"], errors="coerce")

    st.markdown("### Date Filter (by snapshot_time_utc - user input)")

    start_str_snap = st.text_input("Start snapshot date (YYYY-MM-DD, optional)", "")
    end_str_snap = st.text_input("End snapshot date (YYYY-MM-DD, optional)", "")

    df_rt_filtered = df_rt.copy()

    try:
        if start_str_snap:
            start_date_snap = pd.to_datetime(start_str_snap).date()
            df_rt_filtered = df_rt_filtered[df_rt_filtered["snapshot_time_utc"].dt.date >= start_date_snap]
        if end_str_snap:
            end_date_snap = pd.to_datetime(end_str_snap).date()
            df_rt_filtered = df_rt_filtered[df_rt_filtered["snapshot_time_utc"].dt.date <= end_date_snap]
    except Exception:
        st.warning("Invalid date format for snapshot filter. Use YYYY-MM-DD. Filter ignored.")
        df_rt_filtered = df_rt.copy()

    if df_rt_filtered.empty:
        st.warning("No snapshot records fall inside the selected user-defined date range.")
        st.stop()

    st.subheader("Filtered Snapshot Data (Used for Trend Modelling)")
    st.dataframe(df_rt_filtered.head())

    game_names = sorted(df_rt_filtered["name"].unique())
    game_choice = st.selectbox("Select Game", options=game_names)
    metric = st.selectbox("Select Metric to Model", options=["rating", "ratings_count"])

    df_game = df_rt_filtered[df_rt_filtered["name"] == game_choice].copy()
    df_game = df_game.sort_values("snapshot_time_utc")

    st.subheader("Historical Data (Filtered)")
    st.dataframe(df_game[["snapshot_time_utc", metric]])

    # Plot historical series
    fig_hist, axh = plt.subplots()
    axh.plot(df_game["snapshot_time_utc"], df_game[metric], marker="o")
    axh.set_xlabel("Time")
    axh.set_ylabel(metric)
    axh.set_title(f"Historical {metric} for {game_choice} (Filtered Range)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_hist)

    # Fit trend
    model, df_future = fit_linear_trend(df_game, "snapshot_time_utc", metric)
    if model is None:
        st.warning("Not enough data points or time variation to fit a trend model.")
    else:
        st.subheader("Forecast (Next 5 Time Steps)")
        st.dataframe(df_future)

        # Combined plot
        df_future_plot = df_future.rename(columns={"snapshot_time_utc": "time", f"pred_{metric}": metric})
        df_future_plot["type"] = "forecast"
        df_hist_plot = df_game[["snapshot_time_utc", metric]].rename(columns={"snapshot_time_utc": "time"})
        df_hist_plot["type"] = "historical"

        df_all_plot = pd.concat([df_hist_plot, df_future_plot], ignore_index=True)

        fig_trend, axt = plt.subplots()
        for t_type, dsub in df_all_plot.groupby("type"):
            axt.plot(dsub["time"], dsub[metric], marker="o", label=t_type.capitalize())
        axt.set_xlabel("Time")
        axt.set_ylabel(metric)
        axt.set_title(f"Historical vs Forecast {metric} for {game_choice}")
        axt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_trend)

        # Local insight
        st.markdown("---")
        st.subheader("Automatic Trend Insight (Rule-Based)")

        if st.button("Generate Trend Insight"):
            insight_trend = generate_trend_insight(df_game, df_future, metric)
            st.markdown(insight_trend)


# ================== PAGE 3: SENTIMENT ANALYSIS ==================

elif page == "3️⃣ Sentiment Analysis (Reviews)":
    st.title("🗣️ Sentiment Analysis of Game Reviews")

    st.markdown(
        """
        Choose how you want to provide reviews:
        - **Manual Input**: paste multiple reviews, one per line.
        - **Steam App ID**: fetch recent reviews from Steam for a specific app.
        """
    )

    mode = st.radio("Review Source", ["Manual Input", "Steam App ID"])

    reviews_texts: List[str] = []

    if mode == "Manual Input":
        txt = st.text_area("Enter reviews (one per line)", height=200)
        if st.button("Analyse Manual Reviews"):
            reviews_texts = [line for line in txt.split("\n") if line.strip()]
    else:
        app_id = st.text_input(
            "Steam App ID",
            help="Example: 570 for Dota 2, 730 for CS2. Check the Steam store URL: /app/<appid>/",
        )
        num = st.slider("Number of reviews to fetch", 10, 100, 50, step=10)
        if st.button("Fetch & Analyse Steam Reviews"):
            if not app_id:
                st.warning("Please provide a Steam App ID.")
            else:
                with st.spinner("Fetching reviews from Steam..."):
                    try:
                        reviews_texts = fetch_steam_reviews(app_id, num=num)
                    except Exception as e:
                        st.error(f"Failed to fetch reviews: {e}")

    if reviews_texts:
        with st.spinner("Running sentiment analysis..."):
            df_sent = analyze_sentiment(reviews_texts)

        st.subheader("Sentiment per Review")
        st.dataframe(df_sent)

        st.subheader("Sentiment Distribution")
        counts = df_sent["label"].value_counts()
        st.write(counts)

        fig_s, axs = plt.subplots()
        axs.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        axs.set_title("Sentiment Proportions")
        st.pyplot(fig_s)

        # Local insight
        st.markdown("---")
        st.subheader("Automatic Sentiment Insight (Rule-Based)")

        if st.button("Generate Sentiment Insight"):
            sent_insight = generate_sentiment_insight(df_sent)
            st.markdown(sent_insight)
    else:
        st.info("Provide reviews (manual or via Steam) and run the analysis to see results.")


# ================== PAGE 4: COMPETITOR ANALYSIS ==================

elif page == "4️⃣ Competitor Analysis (Steam)":
    st.title("🏁 Steam Competitor Analysis")

    st.markdown(
        """
        This module uses Steam's appdetails endpoint to gather basic competitor data:
        - Pricing and discount information
        - Metacritic score (if available)
        - Player recommendations
        - Genre and category tags

        Please use this responsibly and respect Steam's Terms of Service.
        """
    )

    base_app_id = st.text_input("Base Game Steam App ID (optional)")
    comp_app_ids_str = st.text_area(
        "Competitor Steam App IDs (comma separated)",
        help="Example: 570,730,440",
    )

    if st.button("Fetch Competitor Data"):
        app_ids = []
        if base_app_id:
            app_ids.append(base_app_id)
        if comp_app_ids_str.strip():
            app_ids += [x.strip() for x in comp_app_ids_str.split(",") if x.strip()]

        if not app_ids:
            st.warning("Please specify at least one Steam App ID.")
        else:
            with st.spinner("Fetching competitor data from Steam..."):
                df_comp = collect_competitor_data(app_ids)

            if df_comp.empty:
                st.error("No competitor data retrieved. Check App IDs or connection.")
            else:
                st.subheader("Competitor Data (Steam AppDetails)")
                st.dataframe(df_comp)

                # Price view
                st.subheader("Price and Discount Comparison")
                df_price = df_comp.copy()
                df_price["price_final"] = df_price["price_final_cent"] / 100.0
                st.dataframe(df_price[["name", "currency", "price_final", "discount_percent", "is_free"]])

                # Price bar chart
                df_price_plot = df_price.dropna(subset=["price_final"])
                if not df_price_plot.empty:
                    fig_p, axp = plt.subplots()
                    axp.bar(df_price_plot["name"], df_price_plot["price_final"])
                    axp.set_title("Final Price per Game")
                    axp.set_ylabel("Price")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_p)
                else:
                    st.info("No price data available to plot.")

                st.subheader("Metacritic & Recommendations")
                st.dataframe(df_comp[["name", "metacritic_score", "recommendations", "genres"]])

                # Scatter: price vs metacritic
                if not df_price_plot.dropna(subset=["metacritic_score"]).empty:
                    fig_sc, axsc = plt.subplots()
                    dplot = df_price_plot.dropna(subset=["metacritic_score"])
                    axsc.scatter(dplot["price_final"], dplot["metacritic_score"])
                    for _, row in dplot.iterrows():
                        axsc.text(row["price_final"], row["metacritic_score"], row["name"], fontsize=8)
                    axsc.set_xlabel("Final Price")
                    axsc.set_ylabel("Metacritic Score")
                    axsc.set_title("Price vs Metacritic")
                    plt.tight_layout()
                    st.pyplot(fig_sc)

                st.subheader("Genre / Category Overview")
                st.write(df_comp[["name", "genres", "categories", "is_free"]])

                # Local insight
                st.markdown("---")
                st.subheader("Automatic Competitor Insight (Rule-Based)")

                if st.button("Generate Competitor Insight"):
                    comp_insight = generate_competitor_insight(df_price)
                    st.markdown(comp_insight)
