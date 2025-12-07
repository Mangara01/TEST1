import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download


# ============ PAGE CONFIG & GLOBAL UI ============

st.set_page_config(
    page_title="Game Market Intelligence",
    layout="wide",
    page_icon="🎮",
)

# Inject CSS untuk animasi & styling
def inject_css():
    st.markdown(
        """
        <style>
        /* Background gradient halus */
        .stApp {
            background: radial-gradient(circle at top left, #1f2933, #020617);
            color: #e5e7eb;
        }

        /* Teks gradient animasi */
        .animated-gradient {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(270deg,#ff4b4b,#f9cb28,#2ecc71,#3498db,#9b59b6);
            background-size: 1200% 1200%;
            animation: gradientMove 12s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }

        @keyframes gradientMove {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Subjudul */
        .subtitle {
            font-size: 0.95rem;
            color: #cbd5f5;
            opacity: 0.9;
            margin-bottom: 1.5rem;
        }

        /* Glassmorphism card */
        .glass-card {
            background: rgba(15,23,42,0.82);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            border: 1px solid rgba(148,163,184,0.35);
            box-shadow: 0 22px 60px rgba(15,23,42,0.85);
        }

        /* Sedikit glow di tabel */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(15,23,42,0.6);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: rgba(15,23,42,0.96);
            border-right: 1px solid rgba(148,163,184,0.35);
        }

        /* Radio pill di sidebar */
        .stRadio > label {
            font-weight: 600;
        }

        .stRadio div[role='radiogroup'] > label {
            border-radius: 999px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def animated_title(text: str, subtitle: str | None = None):
    st.markdown(f"<div class='animated-gradient'>{text}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def play_progress(label: str = "Memproses...", steps: int = 12, delay: float = 0.03):
    """Progress bar kecil yang bergerak cepat (total ~0.3–0.4 detik)."""
    container = st.empty()
    bar = st.progress(0)
    for i in range(steps):
        pct = int((i + 1) / steps * 100)
        bar.progress(pct)
        container.text(f"{label} {pct}%")
        time.sleep(delay)
    container.empty()


inject_css()

# Pastikan VADER tersedia
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
    ordering bisa: -added, -rating, -metacritic, -released, dll.
    """
    if not rawg_api_key:
        return pd.DataFrame()

    endpoint = f"{RAWG_BASE_URL}/games"
    params = {
        "key": rawg_api_key,
    }

    # Untuk animasi, kita ambil 2 page jika page_size > 40
    records = []
    remaining = page_size
    page = 1

    while remaining > 0:
        size = min(40, remaining)
        params.update({"page_size": size, "ordering": ordering, "page": page})
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        games = data.get("results", [])
        if not games:
            break

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

        remaining -= size
        page += 1

    return pd.DataFrame.from_records(records)


def plot_bar(x, y, title: str, xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def fit_linear_trend(df: pd.DataFrame, time_col: str, target_col: str):
    """
    Fit regresi linear sederhana: target ~ time (as numeric).
    Return: model, df_future
    """
    df = df.dropna(subset=[time_col, target_col]).copy()
    if df.empty or df[time_col].nunique() < 2:
        return None, None

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # Numeric time (seconds since first)
    t0 = df[time_col].min()
    df["t_numeric"] = (df[time_col] - t0).dt.total_seconds()

    X = df[["t_numeric"]].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    # Buat horizon prediksi 5 titik ke depan (1 hari step)
    last_time = df[time_col].max()
    future_times = [last_time + timedelta(days=i * 1) for i in range(1, 6)]
    future_numeric = np.array([(ft - t0).total_seconds() for ft in future_times]).reshape(
        -1, 1
    )
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
    """
    Ambil review Steam (teks) via endpoint JSON resmi.
    Catatan: jangan spam, hormati rate limit & ToS Steam.
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


def fetch_steam_appdetails(app_id: str) -> dict | None:
    """
    Ambil detail game dari Steam (harga, genre, dll) via appdetails.
    Wishlist TIDAK tersedia dari API publik; bisa isi manual jika punya.
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
        final_price = price_info.get("final")  # dalam cent
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
        # wishlist tidak tersedia; column placeholder
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

st.sidebar.title("⚙️ Control Panel")

rawg_api_key = st.sidebar.text_input(
    "RAWG API Key",
    type="password",
    help="Daftar di https://rawg.io lalu buat API key, masukkan di sini.",
)

st.sidebar.markdown("---")
st.sidebar.caption("🎮 Real-time Game Market Intelligence Dashboard")

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
    animated_title(
        "Game Market Overview",
        "Snapshot real-time dari game yang sedang tren & populer (RAWG).",
    )

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 1.1])
        with col1:
            page_size = st.slider("🎯 Jumlah game diambil", 10, 80, 40, step=10)
        with col2:
            ordering = st.selectbox(
                "🔁 Urutkan berdasarkan",
                options=["-added", "-rating", "-metacritic", "-released"],
                help="-added: baru ditambahkan / populer\n-rating: rating tinggi\n-metacritic: skor critic\n-released: terbaru",
            )
        with col3:
            st.info(
                "Tips: gunakan **-rating** untuk lihat game dengan rating tertinggi, "
                "atau **-metacritic** untuk fokus ke skor kritikus.",
                icon="💡",
            )

        if not rawg_api_key:
            st.warning("Masukkan RAWG API key di sidebar dulu.", icon="⚠️")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        with st.spinner("Mengambil data dari RAWG..."):
            play_progress("Sync dengan RAWG...", steps=14, delay=0.025)
            df_games = fetch_trending_games(rawg_api_key, page_size=page_size, ordering=ordering)

        if df_games.empty:
            st.error("Tidak ada data. Cek API key atau koneksi.", icon="❌")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.toast("Berhasil ambil data dari RAWG ✅", icon="✅")

        tab_table, tab_genre, tab_platform = st.tabs(
            ["📄 Tabel & Highlight", "🎮 Genre Landscape", "🕹️ Platform Landscape"]
        )

        with tab_table:
            st.subheader("📄 Tabel Game Trending / Populer")
            st.dataframe(df_games, use_container_width=True)

            st.subheader("⭐ Top 10 Game berdasarkan Rating")
            top_rating = df_games.sort_values("rating", ascending=False).head(10)
            st.dataframe(
                top_rating[["name", "rating", "ratings_count", "metacritic", "genres", "platforms"]],
                use_container_width=True,
            )

            fig1 = plot_bar(
                x=top_rating.sort_values("rating")["name"],
                y=top_rating.sort_values("rating")["rating"],
                title="Top 10 Rating",
                xlabel="Game",
                ylabel="Rating",
            )
            st.pyplot(fig1, use_container_width=True)

        with tab_genre:
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
            st.pyplot(fig2, use_container_width=True)

        with tab_platform:
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
            st.pyplot(fig3, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ============ PAGE 2: PREDIKSI TREN (AI) ============

elif page == "2️⃣ Prediksi Tren (AI)":
    animated_title(
        "Prediksi Tren Game (AI)",
        "Prediksi rating atau jumlah review berdasarkan snapshot historis (regresi linear).",
    )

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        st.markdown(
            """
            Upload CSV yang berisi **snapshot berkala** seperti:
            - `name`
            - `snapshot_time_utc`
            - `rating`
            - `ratings_count`
            """
        )

        uploaded = st.file_uploader(
            "📤 Upload CSV snapshot",
            type=["csv"],
        )

        if uploaded is not None:
            df_rt = pd.read_csv(uploaded)
            st.write("Contoh data:")
            st.dataframe(df_rt.head(), use_container_width=True)

            # Validasi kolom
            required_cols = {"name", "snapshot_time_utc", "rating", "ratings_count"}
            if not required_cols.issubset(df_rt.columns):
                st.error(f"CSV minimal harus punya kolom: {required_cols}", icon="❌")
            else:
                game_names = sorted(df_rt["name"].unique())
                game_choice = st.selectbox("🎮 Pilih Game", options=game_names)

                metric = st.selectbox(
                    "📌 Target yang diprediksi",
                    options=["rating", "ratings_count"],
                )

                df_game = df_rt[df_rt["name"] == game_choice].copy()

                st.subheader("📉 Data Historis")
                df_game["snapshot_time_utc"] = pd.to_datetime(df_game["snapshot_time_utc"])
                df_game = df_game.sort_values("snapshot_time_utc")
                st.dataframe(df_game[["snapshot_time_utc", metric]], use_container_width=True)

                # Plot historis
                fig_hist, ax = plt.subplots()
                ax.plot(df_game["snapshot_time_utc"], df_game[metric], marker="o")
                ax.set_title(f"Historis {metric} untuk {game_choice}")
                ax.set_xlabel("Waktu")
                ax.set_ylabel(metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_hist, use_container_width=True)

                with st.spinner("Melatih model tren sederhana..."):
                    play_progress("Training model...", steps=10, delay=0.03)
                    model, df_future = fit_linear_trend(df_game, "snapshot_time_utc", metric)

                if model is None:
                    st.warning("Data terlalu sedikit / tidak variatif untuk model regresi.", icon="⚠️")
                else:
                    st.toast("Model tren AI berhasil dilatih 🤖", icon="🤖")
                    st.subheader("🔮 Prediksi 5 Titik ke Depan")
                    st.dataframe(df_future, use_container_width=True)

                    # Gabungkan untuk plot
                    df_future_plot = df_future.rename(columns={f"pred_{metric}": metric})
                    df_future_plot["type"] = "prediksi"
                    df_hist_plot = df_game[["snapshot_time_utc", metric]].copy()
                    df_hist_plot["type"] = "historis"

                    df_hist_plot = df_hist_plot.rename(columns={"snapshot_time_utc": "time"})
                    df_future_plot = df_future_plot.rename(columns={"snapshot_time_utc": "time"})
                    df_all_plot = pd.concat([df_hist_plot, df_future_plot])

                    fig_trend, ax2 = plt.subplots()
                    for t_type, dsub in df_all_plot.groupby("type"):
                        ax2.plot(
                            dsub["time"],
                            dsub[metric],
                            marker="o",
                            label=t_type,
                        )
                    ax2.set_title(f"Historis vs Prediksi {metric} untuk {game_choice}")
                    ax2.set_xlabel("Waktu")
                    ax2.set_ylabel(metric)
                    ax2.legend()
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_trend, use_container_width=True)
        else:
            st.info("Upload CSV dulu supaya bisa bangun model tren.", icon="📂")

        st.markdown("</div>", unsafe_allow_html=True)


# ============ PAGE 3: SENTIMENT ANALYSIS REVIEW ============

elif page == "3️⃣ Sentiment Analysis Review":
    animated_title(
        "Sentiment Analysis Review Game",
        "Analisis opini pemain dari review manual atau Steam (positif / netral / negatif).",
    )

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        st.markdown(
            """
            Pilihan input:
            1. **Manual**: ketik / paste banyak review (1 review per baris).
            2. **Steam**: masukkan Steam App ID (misalnya 730 untuk CS2).
            """
        )

        mode = st.radio(
            "Pilih sumber review",
            ["Manual Input", "Steam (App ID)"],
            horizontal=True,
        )

        reviews_texts: list[str] = []

        if mode == "Manual Input":
            txt = st.text_area(
                "✍️ Masukkan review (1 baris = 1 review):",
                height=200,
                placeholder="Contoh:\nThis game is amazing...\nThe matchmaking is terrible...\n...",
            )
            if st.button("Analisis Sentiment ✨"):
                reviews_texts = txt.split("\n")

        else:
            app_id = st.text_input(
                "Steam App ID",
                help="Contoh: 730 (CS2), 570 (Dota 2). Cek di URL store.steampowered.com/app/<appid>/",
            )
            num = st.slider("Jumlah review yang diambil", 10, 100, 50, step=10)

            if st.button("Ambil & Analisis Review dari Steam 🔍"):
                if not app_id:
                    st.warning("Isi App ID dulu.", icon="⚠️")
                else:
                    with st.spinner("Mengambil review dari Steam..."):
                        play_progress("Download review...", steps=12, delay=0.03)
                        try:
                            reviews_texts = fetch_steam_reviews(app_id, num=num)
                            st.toast("Berhasil ambil review dari Steam ✅", icon="✅")
                        except Exception as e:
                            st.error(f"Gagal ambil review: {e}", icon="❌")

        if reviews_texts:
            with st.spinner("Menghitung sentiment..."):
                play_progress("Scoring sentiment...", steps=12, delay=0.03)
                df_sent = analyze_sentiment(reviews_texts)

            tab_result, tab_examples = st.tabs(["📊 Ringkasan & Grafik", "🔍 Contoh Review"])

            with tab_result:
                st.subheader("📄 Hasil Sentiment per Review")
                st.dataframe(df_sent, use_container_width=True)

                st.subheader("📊 Distribusi Sentiment")
                counts = df_sent["label"].value_counts()
                st.write(counts)

                fig_s, ax = plt.subplots()
                ax.pie(
                    counts.values,
                    labels=counts.index,
                    autopct="%1.1f%%",
                )
                ax.set_title("Proporsi Sentiment")
                st.pyplot(fig_s, use_container_width=True)

            with tab_examples:
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
    animated_title(
        "Scraper & Analisis Kompetitor (Steam)",
        "Bandingkan harga, genre, dan popularitas beberapa game kompetitor.",
    )

    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

        st.markdown(
            """
            Di sini kita gunakan **Steam Web API appdetails + recommendations** sebagai *scraper ringan*:

            - Ambil info game (harga, discount, genre, metacritic, rekomendasi).
            - Bandingkan beberapa game sebagai kompetitor.

            ⚠️ **Catatan penting**:
            - Selalu cek & patuhi Terms of Service dari platform (Steam, dll).
            - Jangan melakukan scraping agresif / berlebihan.
            """
        )

        base_app_id = st.text_input(
            "Steam App ID game utama (base)",
            help="Contoh: 730, 570, 271590, dll.",
        )
        comp_app_ids_str = st.text_area(
            "Steam App ID kompetitor (pisahkan dengan koma)",
            help="Contoh: 730,570,440",
        )

        if st.button("Ambil data kompetitor 🚀"):
            app_ids = []
            if base_app_id:
                app_ids.append(base_app_id)
            if comp_app_ids_str.strip():
                app_ids += [x.strip() for x in comp_app_ids_str.split(",") if x.strip()]

            if not app_ids:
                st.warning("Isi minimal 1 App ID.", icon="⚠️")
            else:
                with st.spinner("Mengambil data detail dari Steam..."):
                    play_progress("Query Steam API...", steps=10, delay=0.035)
                    df_comp = collect_competitor_data(app_ids)

                if df_comp.empty:
                    st.error("Tidak ada data kompetitor yang berhasil diambil.", icon="❌")
                else:
                    st.toast("Data kompetitor berhasil diambil 🏁", icon="🏁")

                    st.subheader("📄 Data Kompetitor (Steam AppDetails)")
                    st.dataframe(df_comp, use_container_width=True)

                    # Harga (konversi ke satuan mata uang dasar)
                    st.subheader("💰 Perbandingan Harga & Diskon")
                    df_price = df_comp.copy()
                    df_price["price_final"] = df_price["price_final_cent"] / 100.0
                    st.dataframe(
                        df_price[["name", "currency", "price_final", "discount_percent", "is_free"]],
                        use_container_width=True,
                    )

                    fig_price, axp = plt.subplots()
                    # Hanya game yang punya harga
                    df_price_plot = df_price.dropna(subset=["price_final"])
                    if not df_price_plot.empty:
                        axp.bar(df_price_plot["name"], df_price_plot["price_final"])
                        axp.set_title("Harga (final) per Game")
                        axp.set_ylabel("Harga")
                        axp.set_xticklabels(df_price_plot["name"], rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig_price, use_container_width=True)
                    else:
                        st.info("Tidak ada data harga untuk diplot.", icon="ℹ️")

                    st.subheader("🎯 Metacritic & Recommendations (proxy popularitas)")
                    st.dataframe(
                        df_comp[["name", "metacritic_score", "recommendations", "genres"]],
                        use_container_width=True,
                    )

                    # Scatter plot harga vs metacritic
                    if df_price_plot["metacritic_score"].notna().sum() > 1:
                        fig_sc, axsc = plt.subplots()
                        sub = df_price_plot.dropna(subset=["metacritic_score"])
                        axsc.scatter(sub["price_final"], sub["metacritic_score"])
                        for _, row in sub.iterrows():
                            axsc.text(
                                row["price_final"],
                                row["metacritic_score"],
                                row["name"],
                                fontsize=8,
                            )
                        axsc.set_xlabel("Harga final")
                        axsc.set_ylabel("Metacritic score")
                        axsc.set_title("Harga vs Kualitas (Metacritic)")
                        plt.tight_layout()
                        st.pyplot(fig_sc, use_container_width=True)

                    st.subheader("🧩 Segmentasi Kompetitor (Genre / Free to Play)")
                    st.write(df_comp[["name", "genres", "categories", "is_free"]])

                    st.markdown(
                        """
                        👉 Insight yang bisa diambil:
                        - Siapa kompetitor **F2P vs premium** di genre yang sama.
                        - Apakah harga mereka selaras dengan kualitas (Metacritic & recommendations).
                        - Kombinasikan dengan **sentiment review** (Halaman 3) untuk melihat gap persepsi vs angka.
                        """
                    )

        st.markdown("</div>", unsafe_allow_html=True)
