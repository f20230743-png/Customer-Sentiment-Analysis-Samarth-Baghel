# 😊 Customer Sentiment Analysis

A Machine Learning and Natural Language Processing (NLP) project that analyzes customer reviews and predicts whether the sentiment is **Positive**, **Neutral**, or **Negative**.

Built using Python, Scikit-learn, NLTK, TF-IDF, and Streamlit.

---

## 🚀 Features

- Customer review sentiment prediction
- Text preprocessing using NLP
- TF-IDF feature extraction
- Multiple Machine Learning models
- Interactive Streamlit web application
- Model comparison based on accuracy

---

## 📊 Dataset

**Women's Clothing E-Commerce Reviews**

- Total Reviews: **22,641**
- Features: **10**
- Target Classes:
  - 😊 Positive
  - 😐 Neutral
  - ☹️ Negative

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Streamlit

---

## 🧠 Machine Learning Models

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **82.38%** |
| Linear SVM | 81.72% |
| Naive Bayes | 80.21% |
| Random Forest | 79.68% |

**Best Model:** Logistic Regression

---

## 📂 Project Structure

```
Customer-Sentiment-Analysis/
│
├── app/
│   └── app.py
│
├── data/
│   └── Womens Clothing E-Commerce Reviews.csv
│
├── model/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/customer-sentiment-analysis.git
```

Go to the project folder

```bash
cd customer-sentiment-analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app.py
```

---

## 📸 Application

The application allows users to:

- Enter a customer review
- Predict the sentiment
- Compare Machine Learning models
- View dataset information
- Explore example reviews

---

## 📈 Workflow

```
Dataset
    ↓
Data Cleaning
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Train Machine Learning Models
    ↓
Model Evaluation
    ↓
Save Best Model
    ↓
Streamlit Web Application
```

---

## 🎯 Future Improvements

- Deep Learning (LSTM/BERT)
- Sentiment confidence score
- Word Cloud visualization
- Docker deployment
- Cloud deployment
- REST API integration

---

## 👨‍💻 Author

**Sujitha**

If you found this project helpful, feel free to ⭐ this repository.
