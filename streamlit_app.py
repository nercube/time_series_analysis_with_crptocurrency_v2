import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "log_return",
    "ma7",
    "ma21",
    "volatility_7",
    "rsi14",
]

CRYPTO_CONFIG = {

    "Bitcoin (BTC)": {

        "ticker": "BTC-USD",

        "data": "btc.csv",

        "coingecko": "bitcoin",

        "arima": "arima_model.pkl",

        "prophet": "prophet_model.pkl",

        "lstm": "btc_lstm_logreturn_multifeature.h5",

        "scaler": "btc_feature_scaler.pkl",
    },

    "Ethereum (ETH)": {

        "ticker": "ETH-USD",

        "data": "eth.csv",

        "coingecko": "ethereum",

        "arima": "eth_arima.pkl",

        "prophet": "eth_prophet (1).pkl",

        "lstm": "eth_lstm (1).h5",

        "scaler": "eth_scaler (1).pkl",
    },

    "Tether (USDT)": {

        "ticker": "USDT-USD",

        "data": "usdt.csv",

        "coingecko": "tether",

        "arima": "usdt_arima.pkl",

        "prophet": "usdt_prophet (1).pkl",

        "lstm": "usdt_lstm (1).h5",

        "scaler": "usdt_scaler.pkl",
    },
}

# =========================================================
# STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Crypto AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {

    background:
    radial-gradient(circle at top left, rgba(99,102,241,0.35), transparent 30%),
    radial-gradient(circle at bottom right, rgba(236,72,153,0.25), transparent 25%),
    linear-gradient(
        135deg,
        #050816 0%,
        #091428 40%,
        #111c3d 100%
    );

    background-attachment: fixed;

    color: white;
}

.block-container {

    padding-top: 1.5rem;

    max-width: 1450px;
}

[data-testid="stSidebar"] {

    background:
    rgba(10,15,35,0.88);

    backdrop-filter: blur(20px);

    border-right:
    1px solid rgba(255,255,255,0.08);
}

.main-title {

    font-size: 4rem;

    font-weight: 900;

    color: white;

    margin-bottom: 0;
}
.main-title {

    text-shadow:
    0 0 20px rgba(255,255,255,0.15);
}

.subtitle {

    color: #94a3b8;

    font-size: 1.1rem;

    margin-bottom: 30px;
}

.glass {

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.10),
        rgba(255,255,255,0.04)
    );

    backdrop-filter: blur(28px);

    -webkit-backdrop-filter: blur(28px);

    border:
    1px solid rgba(255,255,255,0.14);

    border-radius: 32px;

    padding: 28px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.35),
    inset 0 1px 1px rgba(255,255,255,0.06);

    position: relative;

    overflow: hidden;
}
.glass::before {

    content: "";

    position: absolute;

    inset: 0;

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.10),
        transparent 45%
    );

    pointer-events: none;
}

.metric-card {

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.10),
        rgba(255,255,255,0.03)
    );

    backdrop-filter: blur(24px);

    border:
    1px solid rgba(255,255,255,0.10);

    border-radius: 28px;

    padding: 24px;

    min-height: 135px;

    box-shadow:
    0 8px 30px rgba(0,0,0,0.25),
    inset 0 1px 1px rgba(255,255,255,0.05);

    transition: all 0.35s ease;
}
.metric-card:hover {

    transform:
    translateY(-6px)
    scale(1.02);

    border:
    1px solid rgba(96,165,250,0.45);

    box-shadow:
    0 10px 40px rgba(59,130,246,0.18);
}

.metric-label {

    color: #94a3b8;

    font-size: 0.95rem;

    margin-bottom: 12px;
}

.metric-value {

    color: white;

    font-size: 2rem;

    font-weight: 800;
}

.prediction-card {

    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.09),
        rgba(255,255,255,0.03)
    );

    backdrop-filter: blur(20px);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius: 24px;

    padding: 24px;

    margin-bottom: 18px;

    box-shadow:
    0 8px 24px rgba(0,0,0,0.22);

    transition: 0.3s ease;
}
.prediction-card:hover {

    transform: translateY(-5px);

    border:
    1px solid rgba(236,72,153,0.35);

    box-shadow:
    0 10px 30px rgba(236,72,153,0.18);
}

.prediction-title {

    color: #94a3b8;

    font-size: 0.95rem;
}

.prediction-value {

    color: white;

    font-size: 1.7rem;

    font-weight: 700;
}

.js-plotly-plot {

    border-radius: 28px;

    overflow: hidden;

    backdrop-filter: blur(20px);
}

header {

    background: transparent !important;
}

footer {

    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

def compute_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)

def create_features(df):

    feat = df.copy()

    feat["log_return"] = np.log(
        feat["Close"] / feat["Close"].shift(1)
    )

    feat["ma7"] = feat["Close"].rolling(7).mean()

    feat["ma21"] = feat["Close"].rolling(21).mean()

    feat["volatility_7"] = feat["log_return"].rolling(7).std()

    feat["rsi14"] = compute_rsi(
        feat["Close"]
    )

    feat = feat.dropna()

    return feat

def make_lstm_sequence(
    feat_df,
    feature_cols,
    scaler,
    window_size=60,
):

    X_raw = feat_df[
        feature_cols
    ].values

    X_scaled = scaler.transform(
        X_raw
    )

    last_window = X_scaled[
        -window_size:
    ]

    X = np.expand_dims(
        last_window,
        axis=0
    )

    return X

def _load_pickle(path):

    return joblib.load(path)

# =========================================================
# DATA
# =========================================================

@st.cache_data(ttl=600)
def load_price_history(crypto_key):

    cfg = CRYPTO_CONFIG[
        crypto_key
    ]

    csv_path = DATA_DIR / cfg["data"]

    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df.set_index(
        "Date",
        inplace=True
    )

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    return df

# =========================================================
# MODELS
# =========================================================

@st.cache_resource
def load_models(crypto_key):

    cfg = CRYPTO_CONFIG[
        crypto_key
    ]

    arima_model = _load_pickle(
        MODELS_DIR / cfg["arima"]
    )

    prophet_model = _load_pickle(
        MODELS_DIR / cfg["prophet"]
    )

    lstm_model = load_model(
        MODELS_DIR / cfg["lstm"],
        compile=False
    )

    scaler = _load_pickle(
        MODELS_DIR / cfg["scaler"]
    )

    return (
        arima_model,
        prophet_model,
        lstm_model,
        scaler,
    )

# =========================================================
# PREDICTIONS
# =========================================================

def predict_arima(model, horizon):

    preds = model.predict(
        n_periods=horizon
    )

    return np.asarray(
        preds,
        dtype=float
    )

def predict_prophet(
    model,
    hist_df,
    horizon,
):

    future = model.make_future_dataframe(
        periods=horizon,
        freq="D",
    )

    forecast = model.predict(
        future
    )

    preds = forecast[
        "yhat"
    ].tail(horizon)

    return preds.to_numpy(
        dtype=float
    )

def predict_lstm(
    model,
    scaler,
    hist_df,
    horizon,
):

    future_df = hist_df.copy()

    predictions = []

    current_close = float(
        hist_df["Close"].iloc[-1]
    )

    for _ in range(horizon):

        feat_df = create_features(
            future_df
        )

        X_last = make_lstm_sequence(
            feat_df,
            FEATURE_COLUMNS,
            scaler,
        )

        pred_log_return = float(
            model.predict(
                X_last,
                verbose=0
            )[0][0]
        )

        next_close = (
            current_close
            * np.exp(pred_log_return)
        )

        predictions.append(
            next_close
        )

        next_date = (
            future_df.index[-1]
            + dt.timedelta(days=1)
        )

        new_row = future_df.iloc[-1].copy()

        new_row["Close"] = next_close
        new_row["Open"] = next_close
        new_row["High"] = next_close
        new_row["Low"] = next_close

        future_df.loc[next_date] = new_row

        current_close = next_close

    return np.array(predictions)

# =========================================================
# SNAPSHOT
# =========================================================

@st.cache_data(ttl=60)
def fetch_market_snapshot(coin_id):

    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    )

    try:

        r = requests.get(
            url,
            timeout=10,
        )

        data = r.json()

        market_data = data["market_data"]

        return {

            "price":
            market_data["current_price"]["usd"],

            "change_24h":
            market_data["price_change_percentage_24h"],

            "volume_24h":
            market_data["total_volume"]["usd"],

            "market_cap":
            market_data["market_cap"]["usd"],
        }

    except Exception:

        return None

# =========================================================
# DASHBOARD
# =========================================================

def render_dashboard(crypto_key):

    st.markdown(
        f"""
        <div class="main-title">
            {crypto_key}
        </div>

        <div class="subtitle">
            AI Powered Forecast Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:

        st.markdown("## Controls")

        horizon = st.slider(
            "Forecast horizon (days)",
            1,
            90,
            30
        )

        chart_range = st.selectbox(
            "Historical Range",
            [
                "6 Months",
                "1 Year",
                "3 Years",
                "All"
            ],
            index=2,
        )

        use_arima = st.checkbox(
            "ARIMA",
            True
        )

        use_prophet = st.checkbox(
            "Prophet",
            True
        )

        use_lstm = st.checkbox(
            "LSTM",
            True
        )

    price_df = load_price_history(
        crypto_key
    )

    (
        arima_model,
        prophet_model,
        lstm_model,
        scaler,
    ) = load_models(
        crypto_key
    )

    if chart_range == "6 Months":

        recent = price_df.tail(180)

    elif chart_range == "1 Year":

        recent = price_df.tail(365)

    elif chart_range == "3 Years":

        recent = price_df.tail(365 * 3)

    else:

        recent = price_df

    preds = {}

    if use_arima:

        preds["ARIMA"] = predict_arima(
            arima_model,
            horizon,
        )

    if use_prophet:

        try:

            preds["Prophet"] = predict_prophet(
                prophet_model,
                price_df,
                horizon,
            )

        except Exception:

            pass

    if use_lstm:

        preds["LSTM"] = predict_lstm(
            lstm_model,
            scaler,
            price_df,
            horizon,
        )

    snapshot = fetch_market_snapshot(
        CRYPTO_CONFIG[crypto_key]["coingecko"]
    )

    if snapshot:

        c1, c2, c3, c4 = st.columns(4)

        metrics = [

            (
                "Current Price",
                f"${snapshot['price']:,.0f}"
            ),

            (
                "24H Change",
                f"{snapshot['change_24h']:.2f}%"
            ),

            (
                "Market Cap",
                f"${snapshot['market_cap']/1e12:.2f}T"
            ),

            (
                "Volume",
                f"${snapshot['volume_24h']/1e9:.2f}B"
            ),
        ]

        for col, (label, value) in zip(
            [c1, c2, c3, c4],
            metrics
        ):

            with col:
                card_html = f"""
                <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                </div>
                """

                st.markdown(
                    card_html,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    chart_col, pred_col = st.columns(
        [3.2, 1],
        gap="large"
    )

    # =====================================================
    # CHART
    # =====================================================

    with chart_col:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Price Forecast Analysis"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=recent.index,
                open=recent["Open"],
                high=recent["High"],
                low=recent["Low"],
                close=recent["Close"],
                name="Historical",
            )
        )

        last_date = price_df.index[-1]

        last_close = float(
            price_df["Close"].iloc[-1]
        )

        future_dates = [

            last_date
            + dt.timedelta(days=i)

            for i in range(
                1,
                horizon + 1
            )
        ]

        colors = {

            "ARIMA": "#3b82f6",

            "Prophet": "#f472b6",

            "LSTM": "#ef4444",
        }

        for model_name, values in preds.items():

            fig.add_trace(
                go.Scatter(
                    x=[last_date]
                    + future_dates,

                    y=[last_close]
                    + values.tolist(),

                    mode="lines",

                    line=dict(
                        width=4,
                        color=colors[
                            model_name
                        ]
                    ),

                    name=model_name,
                )
            )

        fig.update_layout(

            height=560,

            hovermode="x unified",

            xaxis_rangeslider_visible=False,

            plot_bgcolor="rgba(0,0,0,0)",

            paper_bgcolor="rgba(0,0,0,0)",

            font=dict(
                color="#e5e7eb",
                size=14
            ),

            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    with pred_col:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Predictions"
        )

        for model_name, values in preds.items():

            predicted_value = float(
                values[-1]
            )

            pred_html = f"""
            <div class="prediction-card">
            <div class="prediction-title">{model_name}</div>
            <div class="prediction-value">${predicted_value:,.2f}</div>
            </div>
            """

            st.markdown(
                pred_html,
                unsafe_allow_html=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

# =========================================================
# APP
# =========================================================

selected_crypto = st.sidebar.radio(
    "Choose Cryptocurrency",
    list(CRYPTO_CONFIG.keys()),
)

render_dashboard(
    selected_crypto
)
