import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from newspaper import Article, Config
from transformers import pipeline
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. SETUP & CONFIGURATION
st.set_page_config(
    page_title="EarningsAlpha",
    page_icon="📈",
    layout="wide"
)

# 2. LOAD SAVED MODELS
@st.cache_resource
def load_components():
    model = joblib.load('models/final_xgb_model.joblib')
    scaler = joblib.load('models/standard_scaler.joblib')
    le = joblib.load('models/target_label_encoder.joblib')
    gain_enc = joblib.load('models/ticker_gain_encoder.joblib')
    loss_enc = joblib.load('models/ticker_loss_encoder.joblib')
    gain_enc.handle_unknown = 'error'   #for unknown ticker will give error
    loss_enc.handle_unknown = 'error'
    finbert = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True, device=-1)
    return model, scaler, le, gain_enc, loss_enc, finbert


with st.spinner("Loading AI Models..."):
    xgb_model, scaler, le, gain_enc, loss_enc, finbert_pipeline = load_components()

# 3. HELPER FUNCTIONS
def fetch_news_callback():
    """
    This function runs BEFORE the app reruns, ensuring variables are updated so the text box populates correctly.
    """
    # Getting ticker from ticker box
    ticker = st.session_state.get("ticker_input_key", "AAPL")
    try:
        stock = yf.Ticker(ticker)
        news_list = stock.news
        if news_list:
            chosen = news_list[0]
            for article in news_list:
                details = article.get('content', article)
                title = details.get('title', '').upper()
                if ticker in title:   # Finding the most relevant article
                    chosen = article
                    break
            if 'content' in chosen:
                chosen = chosen['content']
            # Extracting Metadata
            headline = chosen.get('title') or chosen.get('headline') or 'No Title'
            link = chosen.get('link') or chosen.get('url') or ''
            if not link and 'clickThroughUrl' in chosen:
                if isinstance(chosen['clickThroughUrl'], dict):
                    link = chosen['clickThroughUrl'].get('url', '')
                else:
                    link = chosen['clickThroughUrl']

            # Scraping Content
            final_text = headline  # Default
            st.session_state.news_link = link
            if link:
                try:
                    config = Config()
                    config.browser_user_agent = 'Mozilla/5.0'
                    config.request_timeout = 5

                    article = Article(link, config=config)
                    article.download()
                    article.parse()
                    text = article.text[:3000]

                    # If scraping failed/blocked, using summary
                    if len(text) < 50:
                        text = chosen.get('summary', 'No summary available.')
                        st.session_state.msg_type = "warning"
                        st.session_state.msg_text = "Paywall/Video detected. Loaded summary instead."
                    else:
                        st.session_state.msg_type = "success"
                        st.session_state.msg_text = f"Successfully fetched: {headline}"

                    final_text = f"{headline}\n\n{text}"

                except Exception:
                    summary = chosen.get('summary', '')
                    final_text = f"{headline}\n\n{summary}"
                    st.session_state.msg_type = "warning"
                    st.session_state.msg_text = "Could not scrape link. Loaded summary."
            else:
                st.session_state.msg_type = "warning"
                st.session_state.msg_text = "No link found in API response."

            # Updating the Widget State directly
            st.session_state.news_text = final_text
            st.session_state['news_input_box'] = final_text

        else:
            st.session_state.msg_type = "error"
            st.session_state.msg_text = "No news found for this ticker."

    except Exception as e:
        st.session_state.msg_type = "error"
        st.session_state.msg_text = f"API Error: {str(e)}"


def get_finbert_scores(text):
    truncated_text = text[:2000]
    results = finbert_pipeline(truncated_text)
    # Parse results: [{'label': 'positive', 'score': 0.9}, ...]
    scores = {item['label']: item['score'] for item in results[0]}
    return scores.get('positive', 0), scores.get('negative', 0), scores.get('neutral', 0)

# 4. MAIN UI LAYOUT
st.title("📈 EarningsAlpha: Event-Driven Sentiment Engine")
st.markdown("Enter a ticker and earnings news to analyze **Market Reaction Probability**.")
# Creating 2 Columns: Input (Left) vs Analysis (Right)
left_col, right_col = st.columns([1, 1], gap="large")

# LEFT COLUMN: INPUTS
with left_col:
    st.subheader("📝 Market Input")
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_input = st.text_input("Ticker Symbol:", value="AAPL", key="ticker_input_key").upper()
    with c2:
        st.write("")
        st.write("")
        # on_click calls the function BEFORE the app reloads
        st.button("🔄 Fetch News", on_click=fetch_news_callback)

    # Display Status Messages from the Callback
    if 'msg_type' in st.session_state:
        if st.session_state.msg_type == 'success':
            st.success(st.session_state.msg_text)
        elif st.session_state.msg_type == 'warning':
            st.warning(st.session_state.msg_text)
        elif st.session_state.msg_type == 'error':
            st.error(st.session_state.msg_text)
        # Clearing message after showing so it doesn't stick forever
        del st.session_state.msg_type
        del st.session_state.msg_text

    # Initializing session state if needed
    if 'news_text' not in st.session_state:
        st.session_state.news_text = ""
    if 'news_link' not in st.session_state:
        st.session_state.news_link = ""
    if 'news_link' in st.session_state and st.session_state.news_link:
        st.markdown(f"**[📄 Read Full Source Article]({st.session_state.news_link})**")
    news_input = st.text_area(
        "Earnings Transcript / News Source:",
        value=st.session_state.news_text,
        key="news_input_box",  # allows the callback to update it
        height=2140,
        help="Paste the full text here for better accuracy."
    )


# RIGHT COLUMN: PREDICTION
with right_col:
    st.subheader("⚡ AI Forecast")
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
    if run_btn:
        if not ticker_input:
            st.warning("⚠️ Please enter a Ticker Symbol.")
        elif not news_input:
            st.warning("⚠️ Please enter News Text.")
        else:
            with st.spinner("Analyzing Sentiment & Market History..."):
                # FEATURE ENGINEERING ---
                # 1. Date Features (Simulated Today)
                today = pd.Timestamp.now()
                day_of_week = today.dayofweek
                month = today.month
                quarter = today.quarter

                # 2. FinBERT Inference
                pos, neg, neu = get_finbert_scores(news_input)
                ticker_is_unknown = False

                # 3. Target Encoding
                try:
                    # Try to get known history
                    ticker_df = pd.DataFrame({'ticker_clean': [ticker_input]})
                    t_gain = gain_enc.transform(ticker_df).iloc[0, 0]
                    t_loss = loss_enc.transform(ticker_df).iloc[0, 0]
                except:
                    # Unknown Ticker
                    t_gain = 0.45
                    t_loss = 0.45
                    ticker_is_unknown = True

                t_gain_orig = t_gain
                t_loss_orig = t_loss
                # GLOBAL SENTIMENT OVERRIDE (runs for EVERYONE (Known & Unknown))
                override_msg = ""
                if pos > 0.70:
                    t_gain = max(t_gain, 0.55)
                    t_loss = min(t_loss, 0.25)
                    if t_gain != t_gain_orig:
                        override_msg = "🚀 **Note:** Strong News is overriding negative history."
                elif neg > 0.70:
                    t_gain = min(t_gain, 0.25)
                    t_loss = max(t_loss, 0.55)
                    if t_loss != t_loss_orig:
                        override_msg = "📉 **Note:** Negative News is overriding bullish history."

                # Constructing DataFrame for Model (to match X_train)
                input_df = pd.DataFrame([[
                    day_of_week, month, quarter,
                    pos, neg, neu,
                    t_gain, t_loss
                ]], columns=['DateOfWeek', 'Month', 'Quarter', 'FinBERT_Positive', 'FinBERT_Negative',
                             'FinBERT_Neutral',
                             'Ticker_Gain_Prob', 'Ticker_Loss_Prob'])

                # SCALING & PREDICTION
                input_scaled = scaler.transform(input_df)
                probs = xgb_model.predict_proba(input_scaled)[0]

                # Getting probabilities
                classes = list(le.classes_)
                try:
                    gain_idx = classes.index('Gain')
                    loss_idx = classes.index('Loss')
                    neutral_idx = classes.index('Neutral')
                    prob_gain = probs[gain_idx]
                    prob_loss = probs[loss_idx]
                    prob_neutral = probs[neutral_idx]

                except ValueError as e:
                    st.error(f"Error mapping classes. Encoder expects: {classes}")
                    st.stop()

                # DISPLAY RESULTS
                st.markdown("### Final Verdict")
                STRONG_THRESH = 0.60
                WEAK_THRESH = 0.50
                if prob_gain > STRONG_THRESH:
                    st.success(f"## 🚀 STRONG BUY (GAIN)\n**High Confidence: {prob_gain:.1%}**")

                elif prob_gain > WEAK_THRESH and prob_gain > prob_loss:
                    st.success(f"## ✅ LEANING BULLISH (GAIN)\n**Moderate Confidence: {prob_gain:.1%}**")
                    st.caption("Signal is positive but below the strong threshold.")

                elif prob_loss > STRONG_THRESH:
                    st.error(f"## 📉 STRONG SELL (LOSS)\n**High Confidence: {prob_loss:.1%}**")

                elif prob_loss > WEAK_THRESH and prob_loss > prob_gain:
                    st.error(f"## 🔻 LEANING BEARISH (LOSS)\n**Moderate Confidence: {prob_loss:.1%}**")

                elif prob_neutral > STRONG_THRESH:
                    st.info(f"## 😴 PREDICTED STABLE (NEUTRAL)\n**High Confidence: {prob_neutral:.1%}**")
                    st.caption("Model expects low volatility. No significant move detected.")

                else:
                    winner_label = "Gain" if prob_gain > prob_loss else "Loss"
                    winner_prob = max(prob_gain, prob_loss)
                    st.warning(f"## 😐 NEUTRAL / HOLD\n**Uncertain Signal** (Leaning {winner_label}: {winner_prob:.1%})")
                    st.caption(f"Model sees a {winner_label} signal, but it is too weak to trust (<{WEAK_THRESH:.0%}).")
                st.divider()

                st.write("### 🧭 Signal Strength")
                # Calculate a "Sentiment Score" (-1 to 1) for the gauge
                gauge_value = prob_gain - prob_loss
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=gauge_value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Bullish vs Bearish Signal"},
                    delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "rgba(0,0,0,0)"},  # Hide the default bar, we use the needle
                        'steps': [
                            {'range': [-1, -0.5], 'color': "#ff4b4b"},  # Strong Sell (Red)
                            {'range': [-0.5, -0.1], 'color': "#ffbaba"},  # Weak Sell (Light Red)
                            {'range': [-0.1, 0.1], 'color': "#f0f2f6"},  # Neutral (Grey)
                            {'range': [0.1, 0.5], 'color': "#d4f7d4"},  # Weak Buy (Light Green)
                            {'range': [0.5, 1], 'color': "#28a745"}  # Strong Buy (Green)
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': gauge_value
                        }
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Probability Breakdown
                st.write("**Probability Breakdown**")
                chart_data = pd.DataFrame({
                    "Outcome": ["Gain", "Loss", "Neutral"],
                    "Probability": [prob_gain, prob_loss, prob_neutral]
                })
                st.bar_chart(chart_data.set_index("Outcome"))
                st.divider()

                # Explainability
                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    st.info(f"**News Sentiment**\n\nPositive: {pos:.2f}\nNegative: {neg:.2f}")
                with exp_c2:
                    if ticker_is_unknown:
                        st.info(f"**Historical Bias**\n\n⚠️ Unknown Ticker. Using market baseline.")
                    else:
                        if t_gain_orig > 0.55:
                            bias_msg = f"📈 Bullish ({t_gain_orig:.1%} Win Rate)"
                        elif t_loss_orig > 0.55:
                            bias_msg = f"📉 Bearish ({t_loss_orig:.1%} Loss Rate)"
                        else:
                            bias_msg = f"😐 Neutral ({t_gain_orig:.1%} Win / {t_loss_orig:.1%} Loss)"
                        st.info(f"**Historical Bias**\n\n{bias_msg}")
                        if override_msg:
                            st.caption(override_msg)

                st.divider()
                try:
                    stock = yf.Ticker(ticker_input)
                    history = stock.history(period="3mo")
                    if not history.empty:
                        current_price = history['Close'].iloc[-1]
                        start_price = history['Close'].iloc[0]
                        pct_change = ((current_price - start_price) / start_price)
                        if pct_change >= 0:
                            header_emoji = "📈"
                            trend_emoji = "🟢"
                            trend_label = "Up"
                            trend_color = "green"
                        else:
                            header_emoji = "📉"
                            trend_emoji = "🔴"
                            trend_label = "Down"
                            trend_color = "red"
                        st.markdown(f"#### {header_emoji} Market Context (Last 3 Months)")
                        st.line_chart(history['Close'])
                        st.caption(
                            f"Trend: {trend_emoji} :{trend_color}[**{trend_label} {pct_change:.1%}**] over the last quarter.")
                    else:
                        st.warning("Could not fetch price history.")
                except Exception as e:
                    st.markdown("#### 📉 Market Context")
                    st.caption(f"Chart unavailable: {e}")

                st.divider()
                st.write("### ☁️ Key Topics in News")
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(
                        news_input)
                    fig_wc, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig_wc)
                except Exception as e:
                    st.caption(f"Could not generate word cloud: {e}")



