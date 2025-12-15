# ====================================================
#  Open Finance & International Consumer Behavior
#  Streamlit App (Clean, Final Version)
# ====================================================

# ---------- Core Python / System ----------
import os
import sys
import time
import random
import json
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")

# Thread limiting (useful on Windows / sklearn)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ---------- Streamlit ----------
import streamlit as st

# ===============================
# SESSION STATE INITIALIZATION  ✅ FIXED POSITION
# ===============================
if "step" not in st.session_state:
    st.session_state.step = 1
if "bank" not in st.session_state:
    st.session_state.bank = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "consent" not in st.session_state:
    st.session_state.consent = False
if "last_page" not in st.session_state:
    st.session_state.last_page = None

# ---------- Data & Plotting ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------- ML / Analytics ----------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import joblib

# ---------- External Feeds / APIs ----------
import feedparser

# ---------- Streamlit ----------
import streamlit as st

# ---------- AI (Gemini) + Voice ----------
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3

# ---------- Document reading (for uploads) ----------
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None



# ====================================================
#  CONFIG: Gemini & Streamlit
# ====================================================

# 🔑 Configure Gemini API (use your real key here)
GEMINI_API_KEY = "AIzaSyBicYdo9KhbKiwRKbW8mtIP193U6i4ps3Y"  # <-- REPLACE THIS
genai.configure(api_key=GEMINI_API_KEY)

# Choose models (easy to change in one place)
TEXT_MODEL = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"  # vision-capable model

# Streamlit page config
st.set_page_config(
    page_title="Open Finance Analytics",
    page_icon="🌍",
    layout="wide"
)

# ====================================================
#  🌈 BACKGROUND ANIMATION (Gemini-style gradient)
# ====================================================
st.markdown("""
<style>

    /* =====================================================
       FULL PAGE BACKGROUND (Main Content)
       ===================================================== */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #e8d8c3 !important;  /* Soft beige */
        color: #3a2f29 !important;             /* Dark cocoa text */
    }

    /* Remove any animation layers that override colors */
    [data-testid="stAppViewContainer"]::before {
        background: none !important;
    }

    /* =====================================================
       SIDEBAR (Contrast Color)
       ===================================================== */
    [data-testid="stSidebar"] {
        background-color: #2f4f4f !important;  /* Deep teal / dark slate */
        color: #ffffff !important;             /* White text for readability */
        border-right: 2px solid #c7b299 !important;  /* subtle beige border */
    }

    /* Sidebar text and icons */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 500;
    }

    /* Highlight active navigation (dot) */
    .css-1siy2j7, .css-17eq0hr {
        color: #ffe7c2 !important;
    }

    /* =====================================================
       INPUT FIELDS (Matching Beige Theme)
       ===================================================== */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div,
    .stNumberInput > div > input {
        background-color: #f2e6d9 !important;
        color: #3a2f29 !important;
        border-radius: 8px !important;
        border: 1px solid #bba58b !important;
    }

    /* =====================================================
       BUTTONS (Gold-Brown Buttons)
       ===================================================== */
    .stButton > button {
        background-color: #b38755 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #8c6538 !important;
    }

    /* =====================================================
       DATAFRAMES / PLOTS BACKGROUND
       ===================================================== */
    .stDataFrame,
    .stPlotlyChart {
        background-color: #f7ecdf !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    /* Header bar transparency */
    [data-testid="stHeader"] {
        background-color: rgba(255,255,255,0.0) !important;
    }

</style>
""", unsafe_allow_html=True)



# ====================================================
#  GAMIFICATION STATE
# ====================================================

if "points" not in st.session_state:
    st.session_state.points = 0
if "badges" not in st.session_state:
    st.session_state.badges = []
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = pyttsx3.init()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # for Gemini Assistant

def add_points(p: int):
    """Simple gamification: add points and unlock badge."""
    st.session_state.points += p
    st.toast(f"🎯 +{p} points earned!")
    if st.session_state.points >= 100 and "Pro Analyst" not in st.session_state.badges:
        st.session_state.badges.append("Pro Analyst")
        st.balloons()

# ============================================================
# SIMULATED OPEN FINANCE API LAYER
# ============================================================

def open_finance_api(bank_name):
    """
    Simulated Open Finance API
    Represents secure, standardised, consent-based access
    """
    time.sleep(1)  # simulate API latency
    return BANKS.get(bank_name, {})

# ====================================================
#  DATA / MODEL LOADING
# ====================================================

@st.cache_data
def load_models_and_data():
    """
    Load trained ML models and enriched sample data.
    If files are missing, return simulated fallback data.
    """
    try:
        rf_model = joblib.load("rf_model.pkl")
        kmeans_model = joblib.load("kmeans_model.pkl")
        sample_df = pd.read_csv("sample_df.csv")
        rf_accuracy = float(np.load("rf_accuracy.npy")[0])
    except Exception:
        # Fallback: simulated data for demo
        rf_model = lambda X: np.array([random.choice([0, 1]) for _ in range(len(X))])
        kmeans_model = None
        np.random.seed(42)
        sample_df = pd.DataFrame({
            "Country": ["India", "UK", "Brazil", "USA"] * 10,
            "Adoption": np.random.uniform(5, 90, 40),
            "Age": np.random.randint(18, 70, 40),
            "Income": np.random.randint(20, 100, 40),
            "Privacy_Concern": np.random.randint(1, 10, 40),
            "Digital_Literacy": np.random.randint(30, 100, 40)
        })
        rf_accuracy = 0.96

    return rf_model, kmeans_model, sample_df, rf_accuracy

rf_model, kmeans_model, sample_df, rf_accuracy = load_models_and_data()



# ====================================================
#  HELPER: GEMINI TEXT & VISION
# ====================================================

def ask_gemini(prompt: str, model_name: str = TEXT_MODEL) -> str:
    """Call Gemini with a plain text prompt and return text."""
    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return "⚠️ Gemini could not generate a response. Please try again."

def analyze_image_with_gemini(file_bytes: bytes, mime_type: str) -> str:
    """Use Gemini vision model to analyze an image."""
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        resp = model.generate_content([
            {"mime_type": mime_type, "data": file_bytes},
            {
                "text": (
                    "Analyze this image. Extract any visible text, detect charts/tables, "
                    "identify financial or analytical content if present, and summarize "
                    "the main insights in a structured way."
                )
            }
        ])
        return resp.text.strip()
    except Exception as e:
        st.error(f"Gemini Vision Error: {e}")
        return "⚠️ Could not analyze this image."

# ====================================================
#  SIDEBAR NAVIGATION
# ====================================================

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio("Go to:", [
    "🗺️ Global Map",
    "🔮 Prediction Simulator",
    "📱 Sentiment Feed",
    "📊 Visualizations Dashboard",
    "🏦 Banking Connection",
    "📰 Real-Time AI News Summaries",
    "⚖️ Policy Comparator",
    "🚨 Fraud Detection",
    "📄 Auto Report Generator",
    "🤖 Gemini Assistant",  # Text + Voice + File Upload
])

st.sidebar.markdown("---")
st.sidebar.write(f"🏅 **Points:** {st.session_state.points}")
if st.session_state.badges:
    st.sidebar.write(f"🎖️ **Badges:** {', '.join(st.session_state.badges)}")
st.sidebar.info("💡 Explore modules to earn points and unlock badges!")

# ====================================================
#  PAGE 1: GLOBAL MAP + AI ANALYSIS
# ====================================================

if page == "🗺️ Global Map":
    st.header("🌍 Global Open Finance Adoption Map")

    # Choropleth of adoption
    fig = px.choropleth(
        sample_df,
        locations="Country",
        locationmode="country names",
        color="Adoption",
        title="Open Finance Adoption Rates (Simulated / Derived)",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)
    add_points(5)

    # AI Interpretation of country-level adoption
    st.subheader("🤖 AI Interpretation of Global Adoption Patterns")

    summary_df = (
        sample_df.groupby("Country")["Adoption"]
        .mean()
        .reset_index()
        .sort_values("Adoption", ascending=False)
    )

    st.dataframe(summary_df, use_container_width=True)

    summary_txt = summary_df.to_string(index=False)

    prompt = f"""
    You are an expert in global Open Finance policy and financial inclusion.

    Below is a table of average Open Finance adoption scores by country:

    {summary_txt}

    Please:
    - Identify high-adoption countries and explain plausible reasons (e.g., digital infrastructure, regulation, fintech ecosystem).
    - Identify mid-adoption countries and describe potential barriers and opportunities.
    - Identify low-adoption countries and hypothesize structural or regulatory challenges.
    - Keep the explanation well-structured and suitable for a Master's dissertation discussion section.
    """

    ai_analysis = ask_gemini(prompt)
    st.write(ai_analysis)

# ====================================================
#  PAGE 2: PREDICTION SIMULATOR + AI EXPLANATION
# ====================================================

elif page == "🔮 Prediction Simulator":
    st.header("🔮 Predict Open Finance Adoption (Individual-Level Simulation)")

    # User input form
    c1, c2, c3 = st.columns(3)
    with c1:
        country = st.selectbox("Country", sample_df["Country"].unique())
        age = st.slider("Age", 18, 70, 35)
    with c2:
        income = st.slider("Income Score (0–100)", 0, 100, 50)
        literacy = st.slider("Digital Literacy (0–100)", 0, 100, 60)
    with c3:
        privacy = st.slider("Privacy Concern (1–10)", 1, 10, 5)
        sentiment = st.slider("Sentiment Towards Open Finance (0–1)", 0.0, 1.0, 0.6)

    # Build feature vector
    X_input = np.array([[age, income, privacy, literacy, sentiment]])

    # Predict
    try:
        if callable(rf_model):
            pred = rf_model(X_input)[0]
        else:
            pred = rf_model.predict(X_input)[0]
    except Exception:
        pred = random.choice([0, 1])

    label = "✅ Likely to Adopt Open Finance" if pred == 1 else "❌ Unlikely to Adopt Open Finance"
    st.subheader("Prediction")
    st.write(label)
    st.caption(f"Model Accuracy (Random Forest): {rf_accuracy:.2f}")
    add_points(10)

    # AI explanation
    st.subheader("🤖 AI Explanation of This Prediction")

    country_mean = (
        sample_df.groupby("Country")["Adoption"]
        .mean()
        .reset_index()
        .to_string(index=False)
    )

    explanation_prompt = f"""
    You are explaining a machine learning prediction for Open Finance adoption.

    USER PROFILE:
    - Country: {country}
    - Age: {age}
    - Income Score: {income}
    - Digital Literacy: {literacy}
    - Privacy Concern: {privacy}
    - Sentiment: {sentiment:.2f}

    MODEL OUTPUT: {label}

    CONTEXT:
    Below is the average adoption rate by country (higher means more adoption):
    {country_mean}

    TASK:
    - Explain why the model might have predicted {label} for this profile.
    - Comment on how age, income, literacy, privacy concern, and sentiment each affect adoption.
    - Compare {country} to the other countries in the table.
    - Provide suggestions on what could increase this user's likelihood of adoption.
    - Keep it concise but insightful (3–5 short paragraphs).
    """

    explanation = ask_gemini(explanation_prompt)
    st.write(explanation)

# ====================================================
#  PAGE 3: SENTIMENT FEED (SIMULATED)
# ====================================================

elif page == "📱 Sentiment Feed":
    st.header("📱 Open Finance Sentiment Feed (Simulation)")

    st.markdown("Use this module to simulate public sentiment around Open Finance across regions.")

    text = st.text_area("Enter a tweet/comment about Open Finance to classify sentiment:")

    if st.button("🔍 Analyze Sentiment"):
        sentiment_label = random.choice(["Positive 😊", "Neutral 😐", "Negative 😞"])
        score = random.uniform(0.5, 0.95)
        st.success(f"Predicted Sentiment: {sentiment_label}")
        st.metric("Confidence Score", f"{score*100:.1f}%")

    sim = pd.DataFrame({
        "Region": ["Asia", "Europe", "North America", "Africa", "Latin America"],
        "Positive": np.random.randint(40, 90, 5),
        "Negative": np.random.randint(10, 50, 5),
    })

    fig = px.bar(sim, x="Region", y=["Positive", "Negative"], barmode="group",
                 title="Simulated Sentiment Distribution by Region")
    st.plotly_chart(fig, use_container_width=True)
    add_points(10)

# ====================================================
#  PAGE 4: VISUALIZATIONS DASHBOARD 
# ====================================================

elif page == "📊 Visualizations Dashboard":
    st.header("📊 Exploratory Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Adoption by Country")
        avg_adopt = sample_df.groupby("Country")["Adoption"].mean().reset_index()
        fig1 = px.bar(avg_adopt, x="Country", y="Adoption",
                      title="Average Open Finance Adoption")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Correlation Heatmap")
        numeric_cols = ["Age", "Income", "Privacy_Concern", "Digital_Literacy", "Adoption"]
        corr = sample_df[numeric_cols].corr()
        fig2 = px.imshow(corr, text_auto=True, title="Feature Correlations")
        st.plotly_chart(fig2, use_container_width=True)

    add_points(10)

# ===============================
# HOME PAGE (FIXED)
# ===============================
if page == "🏠 Home":
    st.title("🏦 Open Banking & AI Fintech App")
    st.subheader("Welcome to the Platform")

    st.write("""
    This application demonstrates a **simulated Open Banking ecosystem**
    using **Open Finance principles**, **customer consent**, and
    **AI-powered fintech services**.
    """)

    st.markdown("### 🔍 What you can explore:")
    st.markdown("""
    - Secure bank connection flow  
    - AI-based spending analysis  
    - Usage prediction & budget safety  
    - Credit risk insights  
    """)

    st.success("👉 Use the sidebar to start with **Banking Connection**")

# ====================================================
# PAGE: 🏦 BANKING CONNECTION (FIXED)
# ====================================================
elif page == "🏦 Banking Connection":

    st.header("🏦 Open Banking & AI Fintech App")
    st.caption("Customer → Bank → Consent → Open Banking API → AI Services")

    # -------------------------------
    # SAFE BANK DEFINITION
    # -------------------------------
    BANKS = {
        "MockBank India": ["Savings", "Checking"],
        "MockBank UK": ["Savings", "Credit Card"],
        "MockBank USA": ["Checking", "Loan"]
    }

    # -------------------------------
    # RESET FLOW ON PAGE ENTRY
    # -------------------------------
    if st.session_state.last_page != "🏦 Banking Connection":
        st.session_state.step = 1
        st.session_state.bank = None
        st.session_state.consent = False

    st.session_state.last_page = "🏦 Banking Connection"

    # =================================================
    # STEP 1: SELECT BANK
    # =================================================
    if st.session_state.step == 1:
        st.subheader("Step 1: Select Bank")
        bank = st.selectbox("Select Bank", list(BANKS.keys()))

        if st.button("Continue"):
            st.session_state.bank = bank
            st.session_state.step = 2
            st.rerun()

    # =================================================
    # STEP 2: LOGIN (FIXED)
    # =================================================
    elif st.session_state.step == 2:
        st.subheader("Step 2: Secure Bank Login")

        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if not user_id or not password:
                st.warning("Please enter valid credentials.")
            else:
                with st.spinner("Authenticating securely..."):
                    time.sleep(1)

                st.session_state.logged_in = True
                st.session_state.step = 3
                st.success("Login successful")
                st.rerun()

    # =================================================
    # STEP 3: CONSENT
    # =================================================
    elif st.session_state.step == 3:
        st.subheader("Step 3: Customer Consent")

        consent = st.checkbox(
            "I consent to share my financial data via Open Finance APIs"
        )

        if st.button("Grant Access"):
            if not consent:
                st.warning("Consent is mandatory under Open Banking.")
            else:
                st.session_state.consent = True
                st.session_state.step = 4
                st.rerun()

    # =================================================
    # STEP 4: AI FINTECH DASHBOARD ✅
    # =================================================
    elif st.session_state.step == 4:
        st.subheader("Step 4: AI Fintech Dashboard")

        # -------------------------------
        # MOCK OPEN BANKING DATA
        # -------------------------------
        accounts = BANKS[st.session_state.bank]
        balances = {a: random.randint(5_000, 90_000) for a in accounts}

        transactions = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.today(), periods=7),
            "Amount": np.random.randint(-4000, 6000, 7),
            "Category": random.choices(
                ["Salary", "Groceries", "Bills", "Shopping", "Investments"], k=7
            )
        })

        service = st.selectbox(
            "Select Fintech Service",
            [
                "Account Aggregation",
                "Financial Management (AI)",
                "Customer Usage Prediction (AI)",
                "Spending Safety Advisor (AI)",
                "Instant Credit Risk (AI)"
            ]
        )

        st.divider()

        # =================================================
        # ACCOUNT AGGREGATION
        # =================================================
        if service == "Account Aggregation":
            acc = st.selectbox("Select Account", list(balances.keys()))
            st.metric("Account Balance", f"${balances[acc]:,}")

            st.info(
            "🧠 **AI Insight:** Viewing real-time balances helps prevent overdrafts "
            "and improves financial awareness. Consider consolidating unused balances "
            "or allocating surplus funds to savings or investments."
        )

        # =================================================
        # FINANCIAL MANAGEMENT (AI)
        # =================================================
        elif service == "Financial Management (AI)":
            total_balance = sum(balances.values())
            avg_spend = abs(transactions[transactions["Amount"] < 0]["Amount"].mean())

            st.metric("Total Balance", f"${total_balance:,}")
            st.metric("Average Spend", f"${avg_spend:,.0f}")

            st.line_chart(transactions.set_index("Date")["Amount"])

            if avg_spend > 3000:
                st.warning(
                    "🧠 **AI Insight:** High spending trend detected. "
                    "Reduce discretionary expenses, set weekly spending limits, "
                    "and enable real-time alerts to maintain financial stability."
                )
            else:
                st.success(
                    "🧠 **AI Insight:** Spending is under control. "
                    "You are managing expenses efficiently. Consider automating savings "
                    "or allocating funds to low-risk investments."
                )

        # =================================================
        # CUSTOMER USAGE PREDICTION (AI)
        # =================================================
        elif service == "Customer Usage Prediction (AI)":
            recent_spend = abs(transactions[transactions["Amount"] < 0]["Amount"].sum())
            predicted_next_month = int(recent_spend * random.uniform(1.1, 1.3))
            total_balance = sum(balances.values())

            st.metric("Last 7 Days Spending", f"${recent_spend:,}")
            st.metric("Predicted Next Month Spending", f"${predicted_next_month:,}")
            
            projection = pd.DataFrame({
                "Day": range(1, 8),
                "Projected Balance": [
                    total_balance - (i * recent_spend / 7) for i in range(1, 8)
                ]
            })
            
            st.line_chart(projection.set_index("Day"))
        
            if predicted_next_month > total_balance * 0.7:
                st.error(
                    "🧠 **AI Insight:** Predicted spending may exceed safe balance levels. "
                    "Immediate expense control is recommended to avoid liquidity issues."
                )
            elif predicted_next_month > total_balance * 0.4:
                st.warning(
                    "🧠 **AI Insight:** Moderate spending risk detected. "
                    "Monitor expenses closely and adjust discretionary spending."
                )
            else:
                st.success(
                    "🧠 **AI Insight:** Predicted usage is within safe limits. "
                    "Your financial behavior is sustainable."
                )

        # =================================================
        # SPENDING SAFETY ADVISOR (AI)
        # =================================================
        elif service == "Spending Safety Advisor (AI)":
            total_balance = sum(balances.values())
            total_spent = abs(transactions[transactions["Amount"] < 0]["Amount"].sum())
            safe_limit = total_balance * 0.5

            st.metric("Total Balance", f"${total_balance:,}")
            st.metric("Recent Spending", f"${total_spent:,}")
            st.metric("Recommended Safe Limit", f"${safe_limit:,.0f}")

            cat_spend = (
                transactions[transactions["Amount"] < 0]
                .groupby("Category")["Amount"]
                .sum()
                .abs()
            )

            st.bar_chart(cat_spend)

            if total_spent > safe_limit:
               st.error(
                   "🧠 **AI Insight:** Spending exceeds the recommended safety threshold. "
                   "Reduce non-essential expenses, apply category budgets, "
                   "and review spending weekly."
               )
            else:
                st.success(
                    "🧠 **AI Insight:** Spending is within safe limits. "
                    "Maintain current habits and consider increasing savings."
                )

        # =================================================
        # CREDIT RISK (AI)
        # =================================================
        elif service == "Instant Credit Risk (AI)":
            score = random.randint(650, 820)
            st.metric("Credit Score", score)

            if score >= 750:
                st.success("Low Risk – High approval chance")
            elif score >= 680:
                st.warning("Medium Risk")
            else:
                st.error("High Risk")

        # =================================================
        # LOGOUT
        # =================================================
        st.divider()
        if st.button("🔓 Logout"):
            st.session_state.step = 1
            st.session_state.logged_in = False
            st.session_state.consent = False
            st.rerun()


# ====================================================
#  PAGE 6: NEWS FEED
# ====================================================
if page == "📰 Real-Time AI News Summaries":
    st.header("📰 Real-Time Fintech & Open Finance News")

    st.info("Fetching live news from Google News RSS (if available)...")

    try:
        feed_url = (
            "https://news.google.com/rss/search?q=open+finance+OR+fintech+OR+"
            "digital+banking+OR+open+banking&hl=en&gl=US&ceid=US:en"
        )
        feed = feedparser.parse(feed_url)

        if feed.entries:
            for entry in feed.entries[:5]:
                st.markdown(f"🔹 **[{entry.title}]({entry.link})**")
                if hasattr(entry, "published"):
                    st.caption(entry.published)
                st.divider()
        else:
            raise ValueError("No news entries found.")
    except Exception:
        st.warning("Could not fetch live news. Showing static headlines instead.")
        for headline in [
            "IMF: Open Finance boosts inclusion in emerging economies",
            "AI-driven fintechs reshape global banking",
            "UPI-style real-time payments expand internationally",
        ]:
            st.write(f"🗞️ {headline}")

    st.subheader("🤖 AI Summary of Current Landscape")
    summary_prompt = """
    Summarize the current global landscape of Open Finance, Open Banking, and AI in financial services.
    Focus on:
    - Innovation trends
    - Regulatory direction
    - Consumer behavior and inclusion
    - Key opportunities and risks.
    2–4 paragraphs, concise.
    """
    st.write(ask_gemini(summary_prompt))
    add_points(10)


# ====================================================
# PAGE 7: 🤖 Gemini Assistant — Chat + Voice (Updated)
# ====================================================
elif page == "🤖 Gemini Assistant":

    st.title("🤖 Gemini AI Assistant")
    st.caption("Chat using text or speak your question. Works like a mini Gemini inside Streamlit.")

    # -----------------------------
    # Session-state for memory
    # -----------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of {role: "user/assistant", content: "text"}

    if "engine" not in st.session_state:
        st.session_state.engine = pyttsx3.init()

    engine = st.session_state.engine

    # STOP speech button
    if st.button("🛑 Stop Speaking"):
        try:
            engine.stop()
            st.success("Speech stopped.")
        except:
            st.warning("Could not stop speech engine.")

    # -----------------------------
    # Chat UI display
    # -----------------------------
    st.markdown(
        """
        <style>
        .user-msg {background:#DCF8C6;padding:10px;border-radius:10px;margin:5px;max-width:70%;float:right;}
        .bot-msg {background:#ffffff;padding:10px;border-radius:10px;margin:5px;border:1px solid #ddd;max-width:70%;float:left;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-msg'>{msg['content']}</div><br><br>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-msg'>{msg['content']}</div><br><br>", unsafe_allow_html=True)

    st.markdown("---")

    # ======================================================
    # TEXT CHAT
    # ======================================================
    user_input = st.text_input("Type your message to Gemini:", "")

    if st.button("Send Message"):
        if user_input.strip():
            # Save user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                model = genai.GenerativeModel("gemini-2.5-flash")

                response = model.generate_content(
                    f"You are an AI assistant helping with finance, Open Finance, ML, and analytics. "
                    f"Answer conversationally.\n\nUser: {user_input}"
                )

                bot_reply = response.text.strip()
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

                # Speak only first sentence
                short_voice = bot_reply.split(".")[0] + "."
                engine.say(short_voice)
                engine.runAndWait()

            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Gemini Error: {e}"})

            st.rerun()

    # ======================================================
    # VOICE CHAT
    # ======================================================
    st.markdown("### 🎤 Or speak your question")
    if st.button("Start Voice Input"):
        recognizer = sr.Recognizer()

        with sr.Microphone() as src:
            st.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(src, duration=1)
            st.info("Listening… speak now.")

            try:
                audio = recognizer.listen(src, timeout=5)
                text = recognizer.recognize_google(audio)

                # Show captured speech
                st.success(f"🎙️ You said: {text}")
                st.session_state.chat_history.append({"role": "user", "content": text})

                # Generate reply
                model = genai.GenerativeModel("gemini-2.5-flash")
                reply = model.generate_content(
                    f"User asked via speech: {text}. Provide a helpful and concise answer."
                ).text.strip()

                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # Speak only one short sentence
                speak_out = reply.split(".")[0] + "."
                engine.say(speak_out)
                engine.runAndWait()

                st.rerun()

            except sr.WaitTimeoutError:
                st.error("⏳ No speech detected.")
            except Exception as e:
                st.error(f"Voice Error: {e}")


# ====================================================
#  PAGE 7: POLICY COMPARATOR + AI
# ====================================================

elif page == "⚖️ Policy Comparator":
    st.header("⚖️ Open Finance Policy Comparator")

    countries = ["India", "UK", "Brazil", "USA", "Singapore", "Kenya"]
    policy_strength = {
        "India":      {"Innovation": 8.5, "Regulation": 7.5, "Adoption": 9.0},
        "UK":         {"Innovation": 9.0, "Regulation": 8.8, "Adoption": 8.6},
        "Brazil":     {"Innovation": 7.8, "Regulation": 7.0, "Adoption": 7.4},
        "USA":        {"Innovation": 8.2, "Regulation": 7.2, "Adoption": 7.8},
        "Singapore":  {"Innovation": 9.4, "Regulation": 9.0, "Adoption": 8.9},
        "Kenya":      {"Innovation": 7.6, "Regulation": 6.8, "Adoption": 7.2},
    }

    c1, c2 = st.columns(2)
    with c1:
        country1 = st.selectbox("Country 1", countries, index=0)
    with c2:
        country2 = st.selectbox("Country 2", countries, index=1)

    df_compare = pd.DataFrame({
        "Category": ["Innovation", "Regulation", "Adoption"],
        country1: list(policy_strength[country1].values()),
        country2: list(policy_strength[country2].values()),
    })

    fig = px.bar(df_compare, x="Category", y=[country1, country2], barmode="group",
                 title=f"Open Finance Policy Strength: {country1} vs {country2}")
    st.plotly_chart(fig, use_container_width=True)
    add_points(10)

    st.subheader("🤖 AI Policy Analysis")
    p1 = policy_strength[country1]
    p2 = policy_strength[country2]

    policy_prompt = f"""
    You are a policy analyst specialising in Open Finance.

    Country 1: {country1}
    - Innovation: {p1["Innovation"]}
    - Regulation: {p1["Regulation"]}
    - Adoption: {p1["Adoption"]}

    Country 2: {country2}
    - Innovation: {p2["Innovation"]}
    - Regulation: {p2["Regulation"]}
    - Adoption: {p2["Adoption"]}

    TASK:
    - Compare these two countries in terms of Open Finance maturity.
    - Explain which country is currently ahead and why.
    - Suggest what each country can learn from the other.
    - Provide recommendations to strengthen consumer trust and adoption.
    """
    st.write(ask_gemini(policy_prompt))

    # ====================================================
    # 🤖 AI ANALYSIS OF POLICY DIFFERENCES
    # ====================================================
    st.subheader("🤖 AI Analysis of Policy Maturity")

    # Convert policy values to short summary text
    summary1 = f"""
    {country1}:
    Innovation: {policy_strength[country1]["Innovation"]}
    Regulation: {policy_strength[country1]["Regulation"]}
    Adoption: {policy_strength[country1]["Adoption"]}
    """

    summary2 = f"""
    {country2}:
    Innovation: {policy_strength[country2]["Innovation"]}
    Regulation: {policy_strength[country2]["Regulation"]}
    Adoption: {policy_strength[country2]["Adoption"]}
    """

    # AI Prompt
    policy_prompt = f"""
    You are an expert in global Open Finance regulation, policy maturity, and fintech ecosystems.

    Compare the following two countries based on their Open Finance scores:

    COUNTRY 1:
    {summary1}

    COUNTRY 2:
    {summary2}

    TASKS:
    1. Identify which country is stronger in Open Finance maturity and explain why.
    2. Analyze Innovation, Regulation, and Adoption differences.
    3. Explain what regulatory or ecosystem advantages each country has.
    4. Identify gaps and weaknesses for each.
    5. Explain how each country could improve its Open Finance ecosystem.
    6. Keep explanations data-driven based on the provided scores.
    7. Ensure clarity and provide a well-structured comparison.

    Generate a detailed policy comparison analysis:
    """

    # Gemini Response
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(policy_prompt)
        ai_policy_summary = response.text

        st.success("✨ AI Policy Comparison Generated")
        st.write(ai_policy_summary)

    except Exception as e:
        st.error(f"Gemini API Error: {e}")


# ====================================================
# 🚨 Fraud Detection (AI-Powered with Gemini)
# ====================================================
elif page == "🚨 Fraud Detection":
    st.header("🚨 AI Fraud Detection System (Simulation + Gemini Analysis)")

    st.write("Upload a CSV file containing transaction data. The system will flag suspicious transactions and AI will explain WHY they look fraudulent.")

    file = st.file_uploader("📂 Upload CSV file (Transactions Data):")

    if file:
        df = pd.read_csv(file)
        st.subheader("📘 Uploaded Data Preview")
        st.dataframe(df.head())

        # --------------------------------------------------------
        # 1️⃣ Fraud Score Simulation (replaceable with real ML)
        # --------------------------------------------------------
        df["Fraud_Score"] = np.random.uniform(0, 1, len(df))
        flagged = df[df["Fraud_Score"] > 0.8]

        st.subheader(f"⚠️ Suspicious Transactions Detected: {len(flagged)}")
        st.dataframe(flagged)

        if len(flagged) == 0:
            st.success("👍 No suspicious patterns detected in this dataset.")
            st.stop()

        # --------------------------------------------------------
        # 2️⃣ Prepare Summary for AI (Token-Optimized)
        # --------------------------------------------------------
        summary_rows = flagged.describe(include="all").to_string()
        sample_rows = flagged.head(5).to_string()

        st.markdown("### 🤖 AI Explanation of Fraudulent Patterns")
        st.info("Gemini is analyzing the dataset to detect fraud-like patterns...")

        # --------------------------------------------------------
        # 3️⃣ Gemini AI Analysis (Safe + Optimized)
        # --------------------------------------------------------
        try:
            model = genai.GenerativeModel("gemini-2.5-flash") 

            prompt = f"""
            You are an expert financial fraud analyst.

            Below is a statistical summary and sample rows of suspicious transactions flagged by a fraud model:

            -----------------------------------------------------
            SUMMARY STATISTICS:
            {summary_rows}

            SAMPLE SUSPICIOUS ROWS:
            {sample_rows}
            -----------------------------------------------------

            Your task:
            - Analyze the statistical abnormalities.
            - Explain WHY these transactions may be fraudulent.
            - Identify unusual amounts, timing, patterns, frequency, merchants, or deviations from normal behavior.
            - Provide a clear, specific explanation based only on the dataset shown.
            - Avoid generic or template responses. Use details from the summary.

            Provide your fraud analysis:
            """

            response = model.generate_content(prompt)
            ai_explanation = response.text

            st.success("🧠 AI Fraud Analysis Generated Successfully")
            st.write(ai_explanation)

        except Exception as e:
            st.error(f"Gemini API Error: {e}")

        # Award gamification points
        add_points(15)

    else:
        st.info("⬆️ Please upload a CSV file to begin AI fraud analysis.")



