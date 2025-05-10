# ⚽ South American Football Match Predictor

This project uses machine learning to predict outcomes of international football matches between South American teams. It features a full data pipeline, model training, and real-time prediction interface.

---

## 📊 What It Does

* Loads historical match data (2000–present)
* Computes:

  * Rolling averages (goals scored/conceded)
  * Head-to-head performance
* One-hot encodes team names and confederations
* Trains Random Forest and Logistic Regression models
* Predicts the probability of:

  * 🏠 Home Win
  * 🤝 Draw
  * 🛫 Away Win
* Displays feature importance and interprets team strengths

---

## 🧠 Tech Stack

* Python (pandas, scikit-learn, matplotlib)
* RandomForestClassifier + LogisticRegression
* Joblib model serialization
* Optional: XGBoost, CatBoost

---

## 🚀 How to Run

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models**

   ```bash
   python train_models.py
   ```

3. **Predict match outcome**

   ```bash
   python predict_match.py
   ```

---

## 🗂 Project Structure

```
futbolconu-visualization-project/
├── data/
│   ├── matches.csv
│   ├── processed_X.csv
│   └── processed_y.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── logistic_regression_feature_importance.csv
├── src/
│   └── prediction/
│       ├── match_data_preprocessor.py
│       └── football_match_predictor.py
├── train_models.py
└── predict_match.py
```

---

## 🔍 Example Output

```text
Random Forest Match Prediction Probabilities:
  Home Win:  59.6%
  Draw:      25.1%
  Away Win:  15.3%
```

---

## 📌 Future Improvements

* Add a Streamlit or Flask frontend for interactive web use
* Use Elo or FIFA rankings as additional features
* Model explainability (e.g. SHAP)
* Include tournament phase or neutral venue data

---

## 📚 Acknowledgments

Inspired by South American football and powered by data!

---

## 🧠 Author

Created by Andres, a software engineer and football enthusiast.
