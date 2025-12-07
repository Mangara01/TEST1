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
import time

# ============ BASIC CONFIG & GLOBALS ============

st.set_page_config(
    page_title="Game Market Intelligence",
    layout="wide",
    page_icon="🎮",
)

# --- Custom CSS untuk UI lebih beranimasi & modern ---
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Background gradient animated */
        .stApp {
            background: linear-gradient(120deg, #0f172a, #1e293b, #020617);
            background-size: 400% 400%;
            animation: gradientMove 18s ease infinite;
            color: #e5e7eb;
        }

        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Title glow */
        .glow-title h1 {
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.8);
        }

        /* Card-like containers */
        .glass-card {
            background: rgba(15, 23, 42, 0.7);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            backdrop-filter: blur(8px);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.65);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 24px 55px rgba(15, 23, 42, 0.9);
            border-color: rgba(56, 189, 248, 0.55);
        }

        /* Metric style */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,64,175,0.9));
            padding: 1rem 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(96,165,250,0.6);
            box-shadow: 0 12px 30px rgba(15,23,42,0.8);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.92);
            border-right: 1px solid rgba(148, 163, 184, 0.35);
        }

        /* Remove default dataframe background */
        .blank .dataframe {
            background: transparent !important;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 999px;
            border: 1px solid rgba(56,189,248,0.5);
            background: linear-gradient(135deg, #0f172a, #1d4ed8);
            color: #e5e7eb;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            border-color: rgba(56,189,248,0.9);
            transform: translateY(-1px) scale(1.01);
            box-shadow: 0 14px 30px rgba(30,64,175,0.9);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.4rem 1.0rem;
            background: rgba(15,23,42,0.65);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

# ====== NLTK VADER SETUP ======
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk_download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

RAWG_BASE_URL = "https://api.rawg.io/api"


# ============ HELPER FUNCTIONS ============

@st.cache_data(show_spinner=False)
def fetch_trending_games(rawg_api_key: str, page_size: int = 40, ordering: str = "-added") -> pd.DataFrame:
    """
    Ambil game populer/trending dari RAWG API.
    """
    if not rawg_api_key:
        return pd.DataFrame()

    endpoint = f"{RAWG_BASE_URL}/games"
    params = {
        "key": rawg_api_key,
        "page_size": page_size,
        "ordering": ordering,
    }

    resp = requests.get(endpoint, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    games = data.get("results", [])

    records = []
    for g in games:
        records.append(
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

    return pd.DataFrame.from_records(records)


def plot_bar(x, y, title: str, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title, color="#e5e7eb")
    ax.set_xlabel(xlabel, color="#e5e7eb")
    ax.set_ylabel(ylabel, color="#e5e7eb")
    plt.xticks(rotation=45, ha="right", color="#e5e7eb")
    ax.tick_params(colors="#e5e7eb")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    return fig


def fit_linear_trend(df: pd.DataFrame, time_col: str, target_col: str):
    df = df.dropna(subset=[time_col, target_col]).copy()
    if df.empty or df[time_col].nunique() < 2:
        return None, None

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    t0 = df[time_col].min()
    df["t_numeric"] = (df[time_col] - t0).dt.total_seconds()

    X = df[["t_numeric"]].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    last_time = df[time_col].max()
    future_times = [last_time + timedelta(days=i * 1) for i in range(1, 6)]
    future_numeric = np.array([(ft - t0).total_seconds() for ft in future_times]).reshape(-1, 1)
    future_scaled = scaler.transform(future_numeric)
    y_pred = model.predict(future_scaled)

    df_future = pd.DataFrame(
        {
            time_col: future_times,
            f"pred_{target_col}": y_pred,
        }
    )
    return model, df_future


def analyze_sentiment(texts: list[str]) -> pd.DataFrame:
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


def fetch_steam_reviews(app_id: str, num: int = 100) -> list[str]:
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


def fetch_steam_appdetails(app_id: str) -> dict | None:
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
        final_price = price_info.get("final")
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
        "wishlist_estimate": None,
    }
    return info


def collect_competitor_data(app_ids: list[str]) -> pd.DataFrame:
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


# ============ SIDEBAR CONFIG ============

st.sidebar.title("⚙️ Konfigurasi")

st.sidebar.markdown(
    """
    <div style="font-size: 0.9rem; opacity: 0.8;">
    🎮 <b>Game Market Intelligence</b><br>
    Real-time market overview, AI trend prediction,<br>
    sentiment review, dan kompetitor Steam.
    </div>
    """,
    unsafe_allow_html=True,
)

rawg_api_key = st.sidebar.text_input(
    "RAWG API Key",
    type="password",
    help="Daftar di https://rawg.io lalu buat API key, masukkan di sini.",
)

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "1️⃣ Market Overview (RAWG)",
        "2️⃣ Prediksi Tren (AI)",
        "3️⃣ Sentiment Analysis Review",
        "4️⃣ Scraper & Analisis Kompetitor (Steam)",
    ],
)


# ============ PAGE 1: MARKET OVERVIEW ============

if page == "1️⃣ Market Overview (RAWG)":
    st.markdown('<div class="glow-title"><h1>📊 Game Market Overview</h1></div>', unsafe_allow_html=True)

    col_top1, col_top2 = st.columns([2, 1])
    with col_top1:
        st.markdown(
            """
            <div class="glass-card">
            <h3>🎮 Snapshot Game Trending / Populer</h3>
            <p style="font-size:0.9rem; opacity:0.8;">
            Pantau genre, rating, platform dominan, dan struktur pasar game real-time dari RAWG API.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_top2:
        st.markdown(
            """
            <div class="glass-card">
            <b>Tips:</b><br>
            - Coba ganti <code>ordering</code> ke <i>-rating</i> atau <i>-metacritic</i>.<br>
            - Pakai ini untuk cari niche / genre yang lagi naik daun.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        page_size = st.slider("Jumlah game (page_size)", 10, 80, 40, step=10)
    with col2:
        ordering = st.selectbox(
            "Urutkan berdasarkan",
            options=[
                "-added",
                "-rating",
                "-metacritic",
                "-released",
            ],
            help="-added: baru ditambahkan / populer\n-rating: rating tinggi\n-metacritic: skor critic\n-released: terbaru",
        )

    if not rawg_api_key:
        st.warning("Masukkan RAWG API key di sidebar dulu.")
        st.stop()

    # Animasi progress kecil saat fetch
    progress = st.progress(0, text="Menghubungkan ke RAWG API...")
    time.sleep(0.3)
    progress.progress(30, text="Mengambil daftar game trending...")
    with st.spinner("Mengambil data dari RAWG..."):
        df_games = fetch_trending_games(rawg_api_key, page_size=page_size, ordering=ordering)
    progress.progress(100, text="Done ✓")
    time.sleep(0.2)
    progress.empty()

    if df_games.empty:
        st.error("Tidak ada data. Cek API key atau koneksi.")
        st.stop()

    # Metric animated feel
    total_games = len(df_games)
    unique_genres = df_games["genres"].dropna().str.split(", ").explode().nunique()
    unique_platforms = df_games["platforms"].dropna().str.split(", ").explode().nunique()

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Game (snapshot)", total_games)
    m2.metric("Unique Genres", unique_genres)
    m3.metric("Unique Platforms", unique_platforms)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📄 Tabel", "⭐ Rating & Genre", "🕹️ Platform"])
    with tab1:
        st.subheader("📄 Tabel Game")
        st.dataframe(df_games)
    with tab2:
        st.subheader("⭐ Top 10 Game berdasarkan Rating")
        top_rating = df_games.sort_values("rating", ascending=False).head(10)
        st.dataframe(top_rating[["name", "rating", "ratings_count", "metacritic", "genres", "platforms"]])

        fig1 = plot_bar(
            x=top_rating.sort_values("rating")["name"],
            y=top_rating.sort_values("rating")["rating"],
            title="Top 10 Rating",
            xlabel="Game",
            ylabel="Rating",
        )
        st.pyplot(fig1)

        st.subheader("🎮 Distribusi Genre (dalam sampel ini)")
        genres_exploded = df_games["genres"].dropna().str.split(", ").explode()
        genre_counts = genres_exploded.value_counts().head(15)
        st.write(genre_counts)

        fig2 = plot_bar(
            x=genre_counts.index,
            y=genre_counts.values,
            title="Top Genre di Daftar Trending/Populer",
            xlabel="Genre",
            ylabel="Jumlah Game",
        )
        st.pyplot(fig2)

    with tab3:
        st.subheader("🕹️ Distribusi Platform")
        plat_exploded = df_games["platforms"].dropna().str.split(", ").explode()
        plat_counts = plat_exploded.value_counts().head(15)
        st.write(plat_counts)

        fig3 = plot_bar(
            x=plat_counts.index,
            y=plat_counts.values,
            title="Top Platform di Daftar Trending/Populer",
            xlabel="Platform",
            ylabel="Jumlah Game",
        )
        st.pyplot(fig3)

    st.markdown("</div>", unsafe_allow_html=True)


# ============ PAGE 2: PREDIKSI TREN (AI) ============

elif page == "2️⃣ Prediksi Tren (AI)":
    st.markdown('<div class="glow-title"><h1>📈 AI Trend Prediction</h1></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>🤖 Regresi Linear Sederhana untuk Prediksi Tren</h3>
        <p style="font-size:0.9rem; opacity:0.85;">
        Upload CSV hasil snapshot berkala (misalnya dari Colab),
        lalu pilih game dan metrik (<code>rating</code> / <code>ratings_count</code>)
        untuk melihat tren historis dan prediksi ke depan.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV snapshot (contoh struktur: name, snapshot_time_utc, rating, ratings_count, ...)",
        type=["csv"],
    )

    if uploaded is not None:
        df_rt = pd.read_csv(uploaded)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.write("Contoh data:")
        st.dataframe(df_rt.head())

        required_cols = {"name", "snapshot_time_utc", "rating", "ratings_count"}
        if not required_cols.issubset(df_rt.columns):
            st.error(f"CSV minimal harus punya kolom: {required_cols}")
        else:
            game_names = sorted(df_rt["name"].unique())
            game_choice = st.selectbox("Pilih Game", options=game_names)

            metric = st.selectbox(
                "Pilih target yang diprediksi",
                options=["rating", "ratings_count"],
            )

            df_game = df_rt[df_rt["name"] == game_choice].copy()

            st.subheader("📉 Data Historis")
            df_game["snapshot_time_utc"] = pd.to_datetime(df_game["snapshot_time_utc"])
            df_game = df_game.sort_values("snapshot_time_utc")
            st.dataframe(df_game[["snapshot_time_utc", metric]])

            fig_hist, ax = plt.subplots()
            ax.plot(df_game["snapshot_time_utc"], df_game[metric], marker="o")
            ax.set_title(f"Historis {metric} untuk {game_choice}", color="#e5e7eb")
            ax.set_xlabel("Waktu", color="#e5e7eb")
            ax.set_ylabel(metric, color="#e5e7eb")
            plt.xticks(rotation=45, ha="right", color="#e5e7eb")
            ax.tick_params(colors="#e5e7eb")
            fig_hist.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            plt.tight_layout()
            st.pyplot(fig_hist)

            with st.spinner("Melatih model regresi & menghasilkan prediksi..."):
                model, df_future = fit_linear_trend(df_game, "snapshot_time_utc", metric)
                time.sleep(0.3)

            if model is None:
                st.warning("Data terlalu sedikit / tidak variatif untuk model regresi.")
            else:
                st.subheader("🤖 Prediksi 5 Titik ke Depan")
                st.dataframe(df_future)

                df_future_plot = df_future.rename(columns={f"pred_{metric}": metric})
                df_future_plot["type"] = "prediksi"
                df_hist_plot = df_game[["snapshot_time_utc", metric]].copy()
                df_hist_plot["type"] = "historis"

                df_all_plot = pd.concat(
                    [
                        df_hist_plot.rename(columns={"snapshot_time_utc": "time"}),
                        df_future_plot.rename(columns={"snapshot_time_utc": "time"}),
                    ]
                )

                fig_trend, ax2 = plt.subplots()
                for t_type, dsub in df_all_plot.groupby("type"):
                    ax2.plot(
                        dsub["time"],
                        dsub[metric],
                        marker="o",
                        label=t_type,
                    )
                ax2.set_title(f"Historis vs Prediksi {metric} untuk {game_choice}", color="#e5e7eb")
                ax2.set_xlabel("Waktu", color="#e5e7eb")
                ax2.set_ylabel(metric, color="#e5e7eb")
                ax2.legend()
                plt.xticks(rotation=45, ha="right", color="#e5e7eb")
                ax2.tick_params(colors="#e5e7eb")
                fig_trend.patch.set_alpha(0.0)
                ax2.set_facecolor("none")
                plt.tight_layout()
                st.pyplot(fig_trend)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload CSV dulu supaya bisa bangun model tren.")


# ============ PAGE 3: SENTIMENT ANALYSIS REVIEW ============

elif page == "3️⃣ Sentiment Analysis Review":
    st.markdown('<div class="glow-title"><h1>🗣️ Sentiment Analysis Review Game</h1></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>Analisis Sentiment Review</h3>
        <p style="font-size:0.9rem; opacity:0.85;">
        Pilih input manual (paste review) atau ambil langsung dari Steam <b>(App ID)</b>.
        Model VADER akan memberi label <code>positive / neutral / negative</code>.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Pilih sumber review",
        ["Manual Input", "Steam (App ID)"],
    )

    reviews_texts: list[str] = []

    if mode == "Manual Input":
        txt = st.text_area(
            "Masukkan review (1 baris = 1 review):",
            height=200,
        )
        if st.button("Analisis Sentiment"):
            reviews_texts = txt.split("\n")

    else:
        app_id = st.text_input(
            "Steam App ID",
            help="Contoh: 730 (CS2), 570 (Dota 2). Cek di URL store.steampowered.com/app/<appid>/",
        )
        num = st.slider("Jumlah review yang diambil", 10, 100, 50, step=10)

        if st.button("Ambil & Analisis Review dari Steam"):
            if not app_id:
                st.warning("Isi App ID dulu.")
            else:
                progress = st.progress(0, text="Mengambil review dari Steam...")
                with st.spinner("Mengambil review dari Steam..."):
                    try:
                        for i in range(1, 5):
                            time.sleep(0.15)
                            progress.progress(i * 20, text=f"Mengambil review... ({i*20}%)")
                        reviews_texts = fetch_steam_reviews(app_id, num=num)
                        progress.progress(100, text="Done ✓")
                        time.sleep(0.2)
                        progress.empty()
                    except Exception as e:
                        progress.empty()
                        st.error(f"Gagal ambil review: {e}")

    if reviews_texts:
        with st.spinner("Menghitung sentiment..."):
            df_sent = analyze_sentiment(reviews_texts)
            time.sleep(0.25)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 Hasil Sentiment per Review")
        st.dataframe(df_sent)

        st.subheader("📊 Distribusi Sentiment")
        counts = df_sent["label"].value_counts()
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Positive", int(counts.get("positive", 0)))
        col_p2.metric("Neutral", int(counts.get("neutral", 0)))
        col_p3.metric("Negative", int(counts.get("negative", 0)))

        fig_s, ax = plt.subplots()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
        )
        ax.set_title("Proporsi Sentiment")
        fig_s.patch.set_alpha(0.0)
        st.pyplot(fig_s)

        st.subheader("🔍 Contoh Review per Sentiment")
        for label in ["positive", "neutral", "negative"]:
            st.markdown(f"**{label.upper()}**")
            sample = df_sent[df_sent["label"] == label].head(3)
            if sample.empty:
                st.write("_Tidak ada contoh_")
            else:
                for _, row in sample.iterrows():
                    st.write(f"- {row['text']}")
        st.markdown("</div>", unsafe_allow_html=True)


# ============ PAGE 4: SCRAPER & KOMPETITOR (STEAM) ============

elif page == "4️⃣ Scraper & Analisis Kompetitor (Steam)":
    st.markdown('<div class="glow-title"><h1>🏁 Steam Competitor Intelligence</h1></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
        <h3>Scraper Ringan + Analisis Kompetitor</h3>
        <p style="font-size:0.9rem; opacity:0.85;">
        Gunakan Steam <code>appdetails</code> & <code>reviews</code> untuk membandingkan:
        harga, diskon, genre, dan rekomendasi (proxy popularitas).
        <br><br>
        ⚠️ <b>Catatan:</b> selalu patuhi Terms of Service Steam; gunakan scraping secara wajar.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    base_app_id = st.text_input(
        "Steam App ID game utama (base)",
        help="Contoh: 730, 570, 271590, dll.",
    )
    comp_app_ids_str = st.text_area(
        "Steam App ID kompetitor (pisahkan dengan koma)",
        help="Contoh: 730,570,440",
    )

    if st.button("Ambil data kompetitor"):
        app_ids = []
        if base_app_id:
            app_ids.append(base_app_id)
        if comp_app_ids_str.strip():
            app_ids += [x.strip() for x in comp_app_ids_str.split(",") if x.strip()]

        if not app_ids:
            st.warning("Isi minimal 1 App ID.")
        else:
            progress = st.progress(0, text="Mengambil data Steam...")
            with st.spinner("Mengambil data detail dari Steam..."):
                for i in range(1, 5):
                    time.sleep(0.15)
                    progress.progress(i * 20, text=f"Mengambil data Steam... ({i*20}%)")
                df_comp = collect_competitor_data(app_ids)
                progress.progress(100, text="Done ✓")
                time.sleep(0.2)
                progress.empty()

            if df_comp.empty:
                st.error("Tidak ada data kompetitor yang berhasil diambil.")
            else:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.subheader("📄 Data Kompetitor (Steam AppDetails)")
                st.dataframe(df_comp)

                st.subheader("💰 Perbandingan Harga & Diskon")
                df_price = df_comp.copy()
                df_price["price_final"] = df_price["price_final_cent"] / 100.0
                st.dataframe(df_price[["name", "currency", "price_final", "discount_percent", "is_free"]])

                fig_price, axp = plt.subplots()
                df_price_plot = df_price.dropna(subset=["price_final"])
                if not df_price_plot.empty:
                    axp.bar(df_price_plot["name"], df_price_plot["price_final"])
                    axp.set_title("Harga (final) per Game", color="#e5e7eb")
                    axp.set_ylabel("Harga", color="#e5e7eb")
                    axp.set_xticklabels(df_price_plot["name"], rotation=45, ha="right", color="#e5e7eb")
                    axp.tick_params(colors="#e5e7eb")
                    fig_price.patch.set_alpha(0.0)
                    axp.set_facecolor("none")
                    plt.tight_layout()
                    st.pyplot(fig_price)
                else:
                    st.info("Tidak ada data harga untuk diplot.")

                st.subheader("🎯 Metacritic & Recommendations (proxy popularitas)")
                st.dataframe(df_comp[["name", "metacritic_score", "recommendations", "genres"]])

                if df_price_plot["metacritic_score"].notna().sum() > 1:
                    fig_sc, axsc = plt.subplots()
                    axsc.scatter(df_price_plot["price_final"], df_price_plot["metacritic_score"])
                    for _, row in df_price_plot.dropna(subset=["metacritic_score"]).iterrows():
                        axsc.text(
                            row["price_final"],
                            row["metacritic_score"],
                            row["name"],
                            fontsize=8,
                            color="#e5e7eb",
                        )
                    axsc.set_xlabel("Harga final", color="#e5e7eb")
                    axsc.set_ylabel("Metacritic score", color="#e5e7eb")
                    axsc.set_title("Harga vs Kualitas (Metacritic)", color="#e5e7eb")
                    axsc.tick_params(colors="#e5e7eb")
                    fig_sc.patch.set_alpha(0.0)
                    axsc.set_facecolor("none")
                    plt.tight_layout()
                    st.pyplot(fig_sc)

                st.subheader("🧩 Segmentasi Kompetitor (Genre / Free to Play)")
                st.write(df_comp[["name", "genres", "categories", "is_free"]])

                st.markdown(
                    """
                    <div style="font-size:0.9rem; opacity:0.85;">
                    👉 Ide analisis lanjutan:
                    <ul>
                        <li>Kelompokkan kompetitor <b>F2P vs premium</b>.</li>
                        <li>Lihat genre dominan dari kompetitor utama.</li>
                        <li>Gabungkan dengan <b>sentiment review</b> (Halaman 3) untuk insight lebih dalam.</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
