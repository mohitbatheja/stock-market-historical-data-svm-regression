# Stock Market Price Prediction using Support Vector Regressor (SVR)

This repository contains a machine learning project focused on predicting stock closing prices using historical market data. The model is built using Python, `pandas`, `scikit-learn`, and Support Vector Regression (SVR) with an RBF kernel.

---

## 📌 Project Overview

The objective of this project is to process historical stock market dataset records, clean numerical price values, handle invalid dates, scale trading metrics, and predict the final **Closing Price (`Close/Last`)** using trading volume and daily price points.

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Python 3.x
- **Libraries:**
  - `pandas` – Data manipulation and cleaning
  - `numpy` – Numerical calculations
  - `scikit-learn` – Feature scaling, model training (SVR), and evaluation metrics

---

## 📊 Dataset & Preprocessing

The input dataset (`Stock Market Historical Data.csv`) contains key daily trading metrics across multiple companies (e.g., AAPL, NFLX):

1. **Data Cleaning:** 
   - Stripped `$` symbols and commas from currency columns (`Close/Last`, `Open`, `High`, `Low`) and cast them to `float64`.
   - Converted `Date` values to datetime standard (`pd.to_datetime`) with `errors='coerce'`.
   - Dropped missing values (`NaN`/`NaT`) generated during date conversion.

2. **Feature Selection:**
   - **Features (`X`):** `Volume`, `Open`, `High`, `Low`
   - **Target (`y`):** `Close/Last`

3. **Data Splitting & Scaling:**
   - **Train/Test Split:** 80% training data, 20% testing data (`random_state=24`).
   - **Standardization:** Features were normalized using `StandardScaler`.

---

## ⚙️ Model Architecture

- **Algorithm:** Support Vector Regressor (`SVR`)
- **Kernel:** Radial Basis Function (`rbf`)

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model Training
svm_r = SVR(kernel="rbf")
svm_r.fit(x_train_scaled, y_train)

```

---

## 📈 Model Performance & Results

The model performance evaluated on the 20% test subset yielded the following evaluation metrics:

| Metric | Score |
| --- | --- |
| **Mean Absolute Error (MAE)** | `5.69` |
| **Mean Squared Error (MSE)** | `969.76` |
| **Root Mean Squared Error (RMSE)** | `31.14` |
| **R² Score** | **`0.9128` (91.28%)** |

