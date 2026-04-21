# 🏇 Horse Race Prediction Platform

An AI-powered web application that predicts winning probabilities of horses using machine learning and feature engineering.

---

## 📌 Features

- 🎯 Predict win probabilities for each horse
- 📊 Interactive visualization dashboard (Streamlit)
- 🧠 Machine Learning model (XGBoost + Calibration)
- ⚙️ Feature engineering (odds, jockey stats, race size, etc.)
- 🏁 Race-wise probability normalization

---

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Plotly

---

## 📂 Project Structure
Horse Race Prediction/
│── app/
│ └── app.py
│── src/
│ ├── preprocess.py
│ ├── features.py
│ └── model.py
│── data/ # (ignored - not uploaded)
│── model.pkl # trained model
│── requirements.txt
│── README.md


---

## 📊 Dataset

Dataset used from Kaggle:

👉 https://www.kaggle.com/datasets/hwaitt/horse-racing

⚠️ Dataset is not included in this repository due to size constraints.

---

## ⚙️ Setup Instructions

### 1. Clone repo

```bash
git clone https://github.com/your-username/horse-race-prediction.git
cd horse-race-prediction

2. Install dependencies
pip install -r requirements.txt
3. Add dataset

Place dataset inside:

data/
4. Train model
python src/model.py
5. Run app
streamlit run app/app.py
📈 Model Details
Algorithm: XGBoost (with class balancing)
Calibration: Probability calibration for better predictions
Features:
Horse win rate
Jockey win rate
Odds-based features
Race size
Recent performance
⚠️ Notes
Model accuracy may appear high due to class imbalance.
Probabilities reflect realistic uncertainty in horse racing.
🚀 Future Improvements
Live race data integration
Advanced feature engineering
SHAP explainability
Betting strategy insights

👩‍💻 Author

Keerthana C
