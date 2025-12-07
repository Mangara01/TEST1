import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download as nltk_download

from typing import List, Optional
from openai import OpenAI


# ============ SETUP STREAMLIT ============

st.set_page_config(
    page_title="Game Market Intelligence",
    layout="wide",
)

# Pastikan VADER tersedia
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    nltk_download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

RAWG_BASE_URL = "https://api.rawg.io/api"


# ============ HELPER FUNCTIONS (DATA) ============

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

    # Horizon prediksi 5 titik ke depan (1 hari step)
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


def analyze_sentiment(texts: List[str]) -> pd.DataFrame:
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


def fetch_steam_reviews(app_id: str, num: int = 100) -> List[str]:
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


def fetch_steam_appdetails(app_id: str) -> Optional[dict]:
    """
    Ambil detail game dari Steam (harga, genre, dll) via appdetails.
    Wishlist TIDAK tersedia dari API publik; kolom wishlist_estimate jadi placeholder.
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


# ============ HELPER: GENERATIVE AI INSIGHT ============

def generate_ai_insight(openai_api_key: str, system_prompt: str, user_prompt: str) -> str:
    """
    Panggil model OpenAI untuk bikin insight otomatis.
    """
    if not openai_api_key:
        return "⚠️ OpenAI API key belum diisi di sidebar."

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # bisa diganti model lain
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=800,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal memanggil model: {e}"


# ============ SIDEBAR CONFIG ============

st.sidebar.title("⚙️ Konfigurasi")

rawg_api_key = st.sidebar.text_input(
    "RAWG API Key",
    type="password",
    help="Daftar di https://rawg.io/ lalu buat API key, masukkan di sini.",
)

openai_api_key = st.sidebar.text_input(
    "OpenAI API Key (opsional untuk AI Insight)",
    type="password",
    help="Dibutuhkan untuk fitur insight otomatis (Generative AI).",
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
    st.title("📊 Market Overview - Game Trending / Populer")

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

    with st.spinner("Mengambil data dari RAWG..."):
        df_games = fetch_trending_games(rawg_api_key, page_size=page_size, ordering=ordering)

    if df_games.empty:
        st.error("Tidak ada data. Cek API key atau koneksi.")
        st.stop()

    st.subheader("📄 Tabel Game")
    st.dataframe(df_games)

    # Top 10 rating
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

    # Genre distribusi
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

    # Platform distribusi
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

    # === AI Insight ===
    st.markdown("---")
    st.subheader("🔮 Insight Otomatis (Generative AI)")

    if not openai_api_key:
        st.info("Masukkan OpenAI API key di sidebar untuk mengaktifkan insight otomatis.")
    else:
        extra_instruction = st.text_area(
            "Instruksi tambahan untuk AI (opsional)",
            value="Fokus ke peluang pasar dan segmen yang menarik untuk game baru.",
        )

        if st.button("Generate Insight Market Overview"):
            # ringkas context untuk dikirim ke model
            context_str = f"""
Top 10 game (name, rating, ratings_count, genres, platforms):
{top_rating[['name','rating','ratings_count','genres','platforms']].to_string(index=False)}

Top genres (genre: count):
{genre_counts.to_string()}

Top platforms (platform: count):
{plat_counts.to_string()}
"""

            system_prompt = (
                "Kamu adalah analis pasar game yang sangat berpengalaman. "
                "Berikan insight dalam bahasa Indonesia, singkat tapi tajam, "
                "tentang tren market dari data yang diberikan."
            )

            user_prompt = f"""
Ini adalah ringkasan data market (game trending/populer):

{context_str}

Tugasmu:
- Jelaskan pola utama (genre mana yang kuat, platform mana yang dominan).
- Sebutkan minimal 2 peluang atau gap yang bisa dimanfaatkan untuk game baru.
- Gunakan bahasa yang mudah dipahami, bullet point jika perlu.

Instruksi tambahan dari user:
{extra_instruction}
"""

            with st.spinner("AI sedang menganalisis..."):
                insight = generate_ai_insight(openai_api_key, system_prompt, user_prompt)

            st.markdown(insight)


# ============ PAGE 2: PREDIKSI TREN (AI) ============

elif page == "2️⃣ Prediksi Tren (AI)":
    st.title("📈 Prediksi Tren Game (Regresi Linear Sederhana)")

    st.markdown(
        """
        Di halaman ini, kamu bisa:
        - Upload file CSV hasil **snapshot berkala** (misalnya dari script polling RAWG di Colab).
        - Pilih game tertentu.
        - Model tren **rating** atau **ratings_count** terhadap waktu.
        - Gunakan Generative AI untuk menjelaskan tren & implikasinya.
        """
    )

    uploaded = st.file_uploader(
        "Upload CSV snapshot (contoh struktur: name, snapshot_time_utc, rating, ratings_count, ...)",
        type=["csv"],
    )

    if uploaded is not None:
        df_rt = pd.read_csv(uploaded)
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

            # Plot historis
            fig_hist, ax = plt.subplots()
            ax.plot(df_game["snapshot_time_utc"], df_game[metric], marker="o")
            ax.set_title(f"Historis {metric} untuk {game_choice}")
            ax.set_xlabel("Waktu")
            ax.set_ylabel(metric)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_hist)

            # Fit model
            model, df_future = fit_linear_trend(df_game, "snapshot_time_utc", metric)
            if model is None:
                st.warning("Data terlalu sedikit / tidak variatif untuk model regresi.")
            else:
                st.subheader("🤖 Prediksi 5 Titik ke Depan")
                st.dataframe(df_future)

                # Gabungkan untuk plot
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
                ax2.set_title(f"Historis vs Prediksi {metric} untuk {game_choice}")
                ax2.set_xlabel("Waktu")
                ax2.set_ylabel(metric)
                ax2.legend()
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_trend)

                # === AI Insight untuk tren ===
                st.markdown("---")
                st.subheader("🔮 Insight Otomatis dari Tren")

                if not openai_api_key:
                    st.info("Masukkan OpenAI API key di sidebar untuk insight otomatis.")
                else:
                    extra_instruction = st.text_area(
                        "Instruksi tambahan (opsional)",
                        value="Fokus ke implikasi business & live-ops (event, promo, balancing).",
                    )

                    if st.button("Generate Insight Tren"):
                        hist_str = df_game[["snapshot_time_utc", metric]].tail(10).to_string(index=False)
                        future_str = df_future.to_string(index=False)

                        system_prompt = (
                            "Kamu adalah analis data game. "
                            "Jelaskan tren dan rekomendasi strategi live-ops / marketing."
                        )

                        user_prompt = f"""
Game: {game_choice}
Metric yang dianalisis: {metric}

Data historis (tail 10):
{hist_str}

Prediksi 5 titik ke depan:
{future_str}

Tolong jelaskan:
- Apakah trennya cenderung naik/turun/stagnan?
- Apa kemungkinan penyebab (secara hipotesis, misal event, update, kompetitor)?
- Rekomendasi langkah praktis untuk product / marketing / live-ops.

Instruksi tambahan:
{extra_instruction}
"""

                        with st.spinner("AI sedang menganalisis tren..."):
                            insight_trend = generate_ai_insight(openai_api_key, system_prompt, user_prompt)

                        st.markdown(insight_trend)
    else:
        st.info("Upload CSV dulu supaya bisa bangun model tren.")


# ============ PAGE 3: SENTIMENT ANALYSIS REVIEW ============

elif page == "3️⃣ Sentiment Analysis Review":
    st.title("🗣️ Sentiment Analysis Review Game")

    st.markdown(
        """
        Pilihan input:
        1. **Manual**: ketik / paste banyak review (1 review per baris).
        2. **Scrape Steam**: masukkan Steam App ID (misalnya 730 untuk CS2).

        Setelah sentiment dihitung, kamu bisa minta **AI Insight** untuk merangkum pain point & strength utama.
        """
    )

    mode = st.radio(
        "Pilih sumber review",
        ["Manual Input", "Steam (App ID)"],
    )

    reviews_texts: List[str] = []

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
                with st.spinner("Mengambil review dari Steam..."):
                    try:
                        reviews_texts = fetch_steam_reviews(app_id, num=num)
                    except Exception as e:
                        st.error(f"Gagal ambil review: {e}")

    if reviews_texts:
        with st.spinner("Menghitung sentiment..."):
            df_sent = analyze_sentiment(reviews_texts)

        st.subheader("📄 Hasil Sentiment per Review")
        st.dataframe(df_sent)

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

        # === AI Insight untuk sentiment ===
        st.markdown("---")
        st.subheader("🔮 Insight Otomatis dari Sentiment Review")

        if not openai_api_key:
            st.info("Masukkan OpenAI API key di sidebar untuk insight otomatis.")
        else:
            extra_instruction = st.text_area(
                "Instruksi tambahan (opsional)",
                value="Fokus pada rekomendasi perbaikan fitur & monetisasi.",
            )

            if st.button("Generate Insight Sentiment"):
                # Ringkas beberapa contoh review per label
                pos_examples = df_sent[df_sent["label"] == "positive"]["text"].head(5).tolist()
                neg_examples = df_sent[df_sent["label"] == "negative"]["text"].head(5).tolist()
                neu_examples = df_sent[df_sent["label"] == "neutral"]["text"].head(5).tolist()

                summary_str = f"""
Distribusi sentiment (label: count):
{counts.to_string()}

Contoh review positive:
- " + "\n- ".join(pos_examples) if pos_examples else "Tidak ada"

Contoh review negative:
- " + "\n- ".join(neg_examples) if neg_examples else "Tidak ada"

Contoh review neutral:
- " + "\n- ".join(neu_examples) if neu_examples else "Tidak ada"
"""

                system_prompt = (
                    "Kamu adalah analis UX/game designer yang membaca review pemain. "
                    "Tugasmu merangkum feedback utama dan memberi rekomendasi perbaikan produk."
                )

                user_prompt = f"""
Berikut ringkasan sentiment dan contoh review pemain:

{summary_str}

Tolong:
- Rangkum 3-5 poin kekuatan utama game (berdasarkan review).
- Rangkum 3-5 pain point / keluhan utama.
- Berikan rekomendasi konkret yang bisa dilakukan tim game (fitur, balancing, monetisasi, komunikasi).

Instruksi tambahan:
{extra_instruction}
"""

                with st.spinner("AI sedang menganalisis review..."):
                    insight_sent = generate_ai_insight(openai_api_key, system_prompt, user_prompt)

                st.markdown(insight_sent)


# ============ PAGE 4: SCRAPER & KOMPETITOR (STEAM) ============

elif page == "4️⃣ Scraper & Analisis Kompetitor (Steam)":
    st.title("🏁 Scraper & Analisis Kompetitor Game (Steam)")

    st.markdown(
        """
        Di sini kita gunakan **Steam Web API appdetails + reviews** sebagai *scraper ringan*:

        - Ambil info game (harga, discount, genre, metacritic, rekomendasi).
        - Bandingkan beberapa game sebagai kompetitor.
        - Gunakan Generative AI untuk insight positioning & diferensiasi.

        ⚠️ Catatan:
        - Selalu patuhi Terms of Service dari Steam dan jangan scraping berlebihan.
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

    if st.button("Ambil data kompetitor"):
        app_ids = []
        if base_app_id:
            app_ids.append(base_app_id)
        if comp_app_ids_str.strip():
            app_ids += [x.strip() for x in comp_app_ids_str.split(",") if x.strip()]

        if not app_ids:
            st.warning("Isi minimal 1 App ID.")
        else:
            with st.spinner("Mengambil data detail dari Steam..."):
                df_comp = collect_competitor_data(app_ids)

            if df_comp.empty:
                st.error("Tidak ada data kompetitor yang berhasil diambil.")
            else:
                st.subheader("📄 Data Kompetitor (Steam AppDetails)")
                st.dataframe(df_comp)

                # Harga (konversi ke satuan mata uang dasar)
                st.subheader("💰 Perbandingan Harga & Diskon")
                df_price = df_comp.copy()
                df_price["price_final"] = df_price["price_final_cent"] / 100.0
                st.dataframe(df_price[["name", "currency", "price_final", "discount_percent", "is_free"]])

                fig_price, axp = plt.subplots()
                # Hanya game yang punya harga
                df_price_plot = df_price.dropna(subset=["price_final"])
                if not df_price_plot.empty:
                    axp.bar(df_price_plot["name"], df_price_plot["price_final"])
                    axp.set_title("Harga (final) per Game")
                    axp.set_ylabel("Harga")
                    axp.set_xticklabels(df_price_plot["name"], rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_price)
                else:
                    st.info("Tidak ada data harga untuk diplot.")

                st.subheader("🎯 Metacritic & Recommendations (proxy popularitas)")
                st.dataframe(df_comp[["name", "metacritic_score", "recommendations", "genres"]])

                # Scatter plot harga vs metacritic
                if df_price_plot["metacritic_score"].notna().sum() > 1:
                    fig_sc, axsc = plt.subplots()
                    axsc.scatter(df_price_plot["price_final"], df_price_plot["metacritic_score"])
                    for _, row in df_price_plot.dropna(subset=["metacritic_score"]).iterrows():
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
                    st.pyplot(fig_sc)

                st.subheader("🧩 Segmentasi Kompetitor (Genre / Free to Play)")
                st.write(df_comp[["name", "genres", "categories", "is_free"]])

                st.markdown(
                    """
                    👉 Kamu bisa:
                    - Fokus ke kompetitor **F2P** vs **premium**.
                    - Lihat genre apa yang paling sering muncul.
                    - Gabungkan ini dengan sentiment review (Halaman 3) untuk insight yang lebih kaya.
                    """
                )

                # === AI Insight kompetitor ===
                st.markdown("---")
                st.subheader("🔮 Insight Otomatis Analisis Kompetitor")

                if not openai_api_key:
                    st.info("Masukkan OpenAI API key di sidebar untuk insight otomatis.")
                else:
                    extra_instruction = st.text_area(
                        "Instruksi tambahan (opsional)",
                        value="Fokus ke positioning produk baru dan diferensiasi fitur/monetisasi.",
                    )

                    if st.button("Generate Insight Kompetitor"):
                        # ringkas data kompetitor
                        comp_brief_cols = [
                            "name", "is_free", "price_final_cent", "currency",
                            "discount_percent", "metacritic_score", "recommendations",
                            "genres", "categories"
                        ]
                        df_brief = df_comp[comp_brief_cols]
                        brief_str = df_brief.to_string(index=False)

                        system_prompt = (
                            "Kamu adalah analis bisnis game yang melihat data kompetitor di Steam. "
                            "Berikan analisis kompetitif dan rekomendasi positioning."
                        )

                        user_prompt = f"""
Berikut data ringkas game + kompetitor:

{brief_str}

Tolong:
- Kelompokkan game berdasarkan tipe bisnis: F2P vs premium.
- Komentari perbedaan harga, diskon, dan kualitas (metacritic / recommendations).
- Identifikasi celah pasar atau strategi diferensiasi yang bisa diambil (mis: niche genre, pricing, bundling, live-ops).
- Jawab dalam bahasa Indonesia, terstruktur dengan bullet point.

Instruksi tambahan:
{extra_instruction}
"""

                        with st.spinner("AI sedang menganalisis kompetitor..."):
                            insight_comp = generate_ai_insight(openai_api_key, system_prompt, user_prompt)

                        st.markdown(insight_comp)
