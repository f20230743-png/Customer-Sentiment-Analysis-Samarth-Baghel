import streamlit as st
import sys
import os

# ==========================================
# IMPORT PREDICTION FUNCTION
# ==========================================

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.predict import predict_sentiment

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📊 Project Information")

    st.markdown("---")

    st.write("📁 **Dataset**")
    st.write("Women's Clothing Reviews")

    st.write("📈 **Total Reviews**")
    st.write("22,641")

    st.write("🤖 **Best Model**")
    st.write("Logistic Regression")

    st.write("🎯 **Accuracy**")
    st.write("82.38%")

    st.write("📝 **TF-IDF Features**")
    st.write("5000")

    st.markdown("---")

    st.success("Machine Learning Project")

# ==========================================
# MAIN TITLE
# ==========================================

st.title("😊 Customer Sentiment Analysis")

st.write(
    "Analyze customer reviews using Machine Learning and Natural Language Processing."
)

st.divider()

# ==========================================
# DASHBOARD METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📊 Reviews",
        value="22,641"
    )

with col2:
    st.metric(
        label="🎯 Accuracy",
        value="82.38%"
    )

with col3:
    st.metric(
        label="🤖 Best Model",
        value="Logistic Regression"
    )

with col4:
    st.metric(
        label="📝 Features",
        value="5000"
    )

st.divider()

# ==========================================
# INPUT
# ==========================================

st.subheader("📝 Enter Customer Review")

review = st.text_area(
    "",
    height=180,
    placeholder="Example: This dress is beautiful and fits perfectly!"
)

# ==========================================
# BUTTON
# ==========================================

if st.button("🔍 Predict Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a customer review.")

    else:

        prediction = predict_sentiment(review)

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success("😊 Positive")

        elif prediction == "Negative":
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")

st.divider()

# ==========================================
# MODEL COMPARISON
# ==========================================

st.subheader("🏆 Model Comparison")

comparison_data = {
    "Model": [
        "Logistic Regression",
        "Linear SVM",
        "Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [
        "82.38%",
        "81.72%",
        "80.21%",
        "79.68%"
    ]
}

st.table(comparison_data)

st.divider()

# ==========================================
# DATASET INFORMATION
# ==========================================

st.subheader("📂 Dataset Information")

left, right = st.columns(2)

with left:
    st.info("""
**Dataset Name**

Women's Clothing E-Commerce Reviews

**Rows**

22,641

**Columns**

10
""")

with right:
    st.info("""
**Target Classes**

😊 Positive

😐 Neutral

😞 Negative
""")

st.divider()

# ==========================================
# SAMPLE REVIEWS
# ==========================================

st.subheader("💡 Example Reviews")

st.success("This dress fits perfectly and the quality is amazing.")

st.error("Very poor quality. I am disappointed with this product.")

st.info("The parcel arrived yesterday in good condition.")

st.divider()

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
---
<center>

### Customer Sentiment Analysis

Developed using

**Python • Machine Learning • NLP • Streamlit**

© 2026 Developed by **Suji**

</center>
""",
    unsafe_allow_html=True
)