# 🔮 FIFA World Cup 2026 Predictive Oracle & Bracket Simulator

An end-to-end quantitative sports analytics web application that forecasts match probabilities and simulates the expanded 48-team tournament trajectory using machine learning.


## 🚀 Key Architectural Features

* **Dual ML Classification Engines:** Features a high-performance **XGBoost Tree Ensemble** optimized for sharp variations alongside a **Keras/TensorFlow Deep Neural Network** acting as a smooth probabilistic baseline.
* **Leakage-Free Time Series Pipeline:** Features custom rolling engineering windows (`form` and `class` metrics) calculated purely on a chronological index loop to strictly eliminate target and data leakage.
* **Deterministic Standings Engine:** Dynamically ingests multi-class match outcome probabilities to calculate whole-number group standings (Points, Wins, Draws, Losses) on the fly.
* **1,000x Monte Carlo Simulation:** Implements a stochastic knockout execution loop utilizing model-predicted weights to smooth out bracket volatility and map true statistical tournament favorites.

## 🛠️ Tech Stack & Libraries
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn, XGBoost, TensorFlow/Keras
* **Data Processing:** Pandas, NumPy, Joblib

## 📊 Project Insights & Learnings
* **The Short-Window Momentum Dilemma:** Incorporating a 5-match rolling window highlighted how massive statistical outliers in the group stage (e.g., Scotland overperforming) can over-inflate moving metrics, serving as an excellent case study in time-series volatility dampening.