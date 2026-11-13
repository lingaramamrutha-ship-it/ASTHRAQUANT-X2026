import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# -------------------- PAGE --------------------
st.set_page_config(
    page_title="ASTHRAQUANT X 2026",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #0b1220;}
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: #121e31;
        border: 1px solid #26364d;
        padding: 14px;
        border-radius: 12px;
    }
    .hero {
        padding: 22px;
        border-radius: 16px;
        background: linear-gradient(110deg, #122b46, #14213a);
        border: 1px solid #294563;
        margin-bottom: 18px;
    }
    .muted {color: #a9b8ca;}
</style>
""", unsafe_allow_html=True)

st.title("📈 ASTHRAQUANT X 2026")
st.markdown(
    '<div class="hero"><h3>AI-Powered Nifty 50 Short-Term Trend Prediction</h3>'
    '<p class="muted">Historical market analysis • Technical indicators • Machine learning</p></div>',
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.header("Project settings")
period = st.sidebar.selectbox("Historical data period", ["2y", "5y", "10y"], index=1)
threshold_pct = st.sidebar.slider(
    "Trend classification threshold (%)",
    min_value=0.1, max_value=1.0, value=0.25, step=0.05
)
st.sidebar.caption(
    "Bullish/Bearish labels are based on the next trading day's close-to-close return. "
    "This is an educational research prototype."
)

# -------------------- DATA --------------------
@st.cache_data(ttl=3600)
def download_market_data(selected_period):
    frame = yf.download(
        "^NSEI",
        period=selected_period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )
    if frame.empty:
        raise ValueError(
            "Yahoo Finance returned no data. Check your internet connection "
            "or upload a CSV containing Date, Open, High, Low, Close and Volume."
        )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]

    # Yahoo Finance may call the date field either Date or Datetime.
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=["date", "close"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"]).reset_index(drop=True)
    return frame


@st.cache_data(ttl=3600)
def load_uploaded_csv(file_bytes):
    from io import BytesIO
    frame = pd.read_csv(BytesIO(file_bytes))
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "datetime" in frame.columns and "date" not in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"CSV must contain these columns: {required}. Missing: {missing}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=required).sort_values("date").drop_duplicates("date").reset_index(drop=True)


def make_features(raw):
    df = raw.copy()

    # Returns and price-relative features
    df["return_1d"] = df["close"].pct_change()
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    # Moving averages
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["close_ema10_gap"] = df["close"] / df["ema_10"] - 1
    df["ema10_ema20_gap"] = df["ema_10"] / df["ema_20"] - 1
    df["ema20_ema50_gap"] = df["ema_20"] / df["ema_50"] - 1

    # RSI (14-period)
    change = df["close"].diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD and normalized MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_gap_pct"] = (df["macd"] - df["macd_signal"]) / df["close"]

    # Volatility and volume activity
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volume_change"] = df["volume"].pct_change()
    df["volume_change"] = df["volume_change"].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


FEATURES = [
    "return_1d", "return_3d", "return_5d",
    "close_ema10_gap", "ema10_ema20_gap", "ema20_ema50_gap",
    "rsi_14", "macd_gap_pct", "volatility_10d", "volume_change"
]
CLASS_NAMES = {0: "SIDEWAYS", 1: "BULLISH", 2: "BEARISH"}

def create_training_data(feature_frame, threshold):
    data = feature_frame.copy()
    next_return = data["close"].shift(-1) / data["close"] - 1
    cut = threshold / 100.0
    data["target"] = np.select(
        [next_return > cut, next_return < -cut],
        [1, 2],
        default=0
    )
    # Last row has no known next-day outcome and must not be used for training/testing.
    data.loc[next_return.isna(), "target"] = np.nan
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def fit_and_evaluate(data):
    usable = data.dropna(subset=FEATURES + ["target"]).copy()
    if len(usable) < 150:
        raise ValueError(
            "Not enough usable rows after indicator calculation. "
            "Try selecting a longer historical period or using a fuller CSV."
        )

    X = usable[FEATURES]
    y = usable["target"].astype(int)

    split_at = int(len(usable) * 0.80)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains fewer than two trend classes. "
            "Try a different trend threshold or longer data period."
        )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision (macro)": precision_score(y_test, test_predictions, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, test_predictions, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, test_predictions, average="macro", zero_division=0)
    }

    # Refit on all known labelled rows for the latest available prediction.
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X, y)
    return final_model, metrics, y_test, test_predictions, usable, split_at


# -------------------- DATA SOURCE --------------------
st.sidebar.subheader("Data source")
uploaded = st.sidebar.file_uploader("Optional: upload your own OHLCV CSV", type=["csv"])

try:
    with st.spinner("Loading Nifty 50 data and calculating indicators..."):
        if uploaded is not None:
            raw_data = load_uploaded_csv(uploaded.getvalue())
            source_label = "Uploaded CSV"
        else:
            raw_data = download_market_data(period)
            source_label = "Yahoo Finance (^NSEI)"

        feature_data = make_features(raw_data)
        training_data = create_training_data(feature_data, threshold_pct)
        model, metrics, y_test, test_predictions, usable, split_at = fit_and_evaluate(training_data)

        latest_row = feature_data.dropna(subset=FEATURES).iloc[-1]
        latest_X = latest_row[FEATURES].to_frame().T
        predicted_class = int(model.predict(latest_X)[0])
        probabilities = model.predict_proba(latest_X)[0]
        probability_by_class = dict(zip(model.classes_.astype(int), probabilities))
        confidence = float(max(probabilities)) * 100

except Exception as exc:
    st.error(f"Could not prepare the dashboard: {exc}")
    st.info("Check your internet connection, try again, or upload a CSV with Date, Open, High, Low, Close and Volume columns.")
    st.stop()

# -------------------- OVERVIEW --------------------
st.caption(f"Data source: {source_label} • Latest available observation: {latest_row['date'].date()}")

st.subheader("Market snapshot")
latest_close = float(latest_row["close"])
previous_close = float(feature_data["close"].iloc[-2])
daily_change = (latest_close / previous_close - 1) * 100

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest available close", f"{latest_close:,.2f}", f"{daily_change:+.2f}%")
m2.metric("RSI (14)", f"{latest_row['rsi_14']:.1f}")
m3.metric("EMA (20)", f"{latest_row['ema_20']:,.2f}")
m4.metric("MACD", f"{latest_row['macd']:.2f}")

# -------------------- PREDICTION --------------------
st.divider()
st.subheader("Machine-learning trend classification")
trend = CLASS_NAMES[predicted_class]
trend_color = {"BULLISH": "🟢", "BEARISH": "🔴", "SIDEWAYS": "🟡"}[trend]

p1, p2 = st.columns([1, 2])
with p1:
    st.markdown(f"## {trend_color} {trend}")
    st.write("Estimated class for the next trading session.")
    st.metric("Highest class probability", f"{confidence:.1f}%")
with p2:
    prob_df = pd.DataFrame({
        "Trend": [CLASS_NAMES[int(c)] for c in model.classes_],
        "Model probability": probabilities
    }).set_index("Trend")
    st.bar_chart(prob_df)

st.caption(
    "The displayed probability is the model's class probability, not a guarantee "
    "that the market will move in that direction."
)

# -------------------- PRICE CHART --------------------
st.divider()
st.subheader("Nifty 50 price and moving averages")
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["close"], name="Close"))
price_fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["ema_20"], name="EMA 20"))
price_fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["ema_50"], name="EMA 50"))
price_fig.update_layout(
    template="plotly_dark", height=480,
    xaxis_title="Date", yaxis_title="Index level",
    legend=dict(orientation="h", y=1.08)
)
st.plotly_chart(price_fig, use_container_width=True)

# -------------------- INDICATORS --------------------
st.divider()
st.subheader("Technical indicators")
tab1, tab2, tab3 = st.tabs(["RSI", "MACD", "Volatility"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["rsi_14"], name="RSI 14"))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(template="plotly_dark", height=350, yaxis_title="RSI")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["macd"], name="MACD"))
    fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["macd_signal"], name="Signal"))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feature_data["date"], y=feature_data["volatility_10d"], name="10-day volatility"))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- MODEL EVALUATION --------------------
st.divider()
st.subheader("Out-of-sample model evaluation")
st.caption("The first 80% of chronological labelled observations train the evaluation model; the final 20% are held out for testing.")

metric_cols = st.columns(4)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value * 100:.1f}%")

st.markdown("**Confusion matrix** (rows = actual class, columns = predicted class)")
cm = confusion_matrix(y_test, test_predictions, labels=[0, 1, 2])
cm_frame = pd.DataFrame(
    cm,
    index=["Actual Sideways", "Actual Bullish", "Actual Bearish"],
    columns=["Predicted Sideways", "Predicted Bullish", "Predicted Bearish"]
)
st.dataframe(cm_frame, use_container_width=True)

with st.expander("Detailed classification report"):
    report = classification_report(
        y_test, test_predictions,
        labels=[0, 1, 2],
        target_names=["Sideways", "Bullish", "Bearish"],
        zero_division=0,
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# -------------------- DATA TABLE --------------------
st.divider()
with st.expander("View recent historical data"):
    show_cols = ["date", "open", "high", "low", "close", "volume", "rsi_14", "macd", "ema_20"]
    st.dataframe(feature_data[show_cols].tail(30), use_container_width=True)

st.info(
    "Research prototype only. This project classifies historical price movement for "
    "educational evaluation. It does not execute trades and is not financial advice. "
    "Past market behaviour does not guarantee future results."
)
st.caption("ASTHRAQUANT X 2026 • Built with Python, Streamlit, yfinance and scikit-learn")
