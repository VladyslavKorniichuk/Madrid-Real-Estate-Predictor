# 🏠 Madrid Real Estate Predictor

A machine learning web application that predicts apartment prices in Madrid based on property features. Built with Streamlit and trained on real Madrid housing data.

---

## 📸 Demo

<img width="660" height="822" alt="image" src="https://github.com/user-attachments/assets/a3cbe126-268a-41a1-86c3-098a4911b545" />


---

## 🤖 Models & Performance

| Model | MAE Error | RMSE Error |
|---|---|---|
| **Random Forest** ⭐ | 110,780 € | 248,478 € |
| ANN | 136,807 € | 389,466 € |
| Gradient Boosting | 139,990 € | 265,012 € |
| XGBoost | 142,270 € | 272,539 € |
| ANN 2 (SNN) | 157,531 € | 324,476 € |
| KNeighbors | 170,705 € | 332,617 € |
| Lasso | 195,813 € | 347,032 € |
| Linear Regression | 195,820 € | 347,034 € |

> ⭐ Random Forest achieved the best results with the lowest MAE

---

## 🛠️ Tech Stack

- **Python 3.10**
- **Streamlit** — web application
- **scikit-learn** — ML models
- **XGBoost** — gradient boosting
- **pandas / numpy** — data processing
- **joblib** — model serialization

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/VladyslavKorniichuk/Madrid-Real-Estate-Predictor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
cd app
streamlit run app.py
```

---

## 📁 Project Structure

```
Project/
├── app/
│   ├── app.py              # Streamlit application
│   └── models/             # Trained models (.pkl files)
│       ├── scaler.pkl
│       ├── rf_model.pkl
│       ├── lr_model.pkl
│       ├── ann_model.pkl
│       ├── ann_model_2.pkl
│       ├── model_knr.pkl
│       ├── model_lasso.pkl
│       ├── model_GBR.pkl
│       └── model_XGB.pkl
├── model.ipynb             # Model training notebook
├── data_cleaning.py        # Data preprocessing
└── README.md
```

---

## ✍️ Authors

- **Pavlo Mysiuk** — Streamlit app, 4 additional ML models
- **Vladyslav Korniichuk** — Data cleaning, initial 4 ML models
