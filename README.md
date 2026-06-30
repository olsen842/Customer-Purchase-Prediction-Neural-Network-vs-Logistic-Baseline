# Steam Trading Bot
> A machine learning project that predicts whether to buy or skip Steam market items using logistic regression and a neural network — with focus on fair model comparison, threshold tuning, and a real data pipeline from the Steam Community Market.

## Problem Statement

Explores whether it's possible to build a trading bot that can decide if a Steam market item is worth investing in. Goals include learning PyTorch in practice, building custom datasets, scraping a real (rate-limited) public API, and ensuring fair evaluation through strict train/test separation.

---

## Project Structure

| File | Purpose |
|---|---|
| `collect_data.py` | Scrapes live prices/listings for ~20 CS2 skins from the Steam Community Market API on a polling loop, with rate-limit handling and momentum tracking between runs |
| `generate_data.py` | Original synthetic data generator — simulates `future_price` from momentum, volume, spread and listings |
| `generate_data_v2.py` | Improved synthetic generator — models a `true_value` per skin and a `price_ratio`, with ~55% of listings simulated as underpriced "deals" for more realistic market behavior |
| `train_and_compare.py` | Trains both models on the same data/split/threshold and prints a side-by-side comparison; saves the trained models + scaler |
| `app.py` | Flask API that loads the saved model + scaler and serves buy/skip predictions with a confidence score |
| `test.py` | Standalone logistic regression baseline run for quick comparison checks |

---

## Input Features

| Feature | Description |
|---|---|
| `price_ratio` | Relative price vs. median |
| `num_listings` | Number of sellers (supply) |
| `daily_volume` | Trading activity (liquidity) |
| `spread` | Buy/sell price gap |
| `momentum` | Price trend direction |

## Preprocessing

Train/test split 80/20 — identical for both models. Feature scaling via StandardScaler fitted on train data only. No data leakage: test set was unseen during all training.

---

## Models

### Logistic Regression *(Baseline)*

| Parameter | Value |
|---|---|
| Architecture | `5 → 1 (sigmoid)` |
| Loss | `BCE` |
| Optimizer | `Adam lr=0.001` |
| Epochs | `640` |

### Neural Network *(Best model)*

| Parameter | Value |
|---|---|
| Architecture | `5 → 16 → 1` |
| Activation | `ReLU + sigmoid` |
| Optimizer | `Adam lr=0.001` |
| Epochs | `640` |

---

## Results — Unseen Test Data

| Model | Accuracy | Precision | Recall | Test Loss |
|---|---|---|---|---|
| Logistic Regression | 75.00% | 0.74 | 0.76 | 0.3969 |
| **Neural Network** ✅ | **85.00%** | **0.85** | **0.78** | **0.2207** |

## Threshold Analysis — Neural Network

| Threshold | Accuracy | Precision | Recall | F1-Score | Buyers predicted |
|---|---|---|---|---|---|
| 0.5 | 87% | 0.86 | 0.84 | 0.85 | 44 |
| **0.6** ⭐ selected | 84% | 0.85 | 0.78 | 0.81 | 41 |
| 0.7 | 82% | 0.89 | 0.69 | 0.78 | 35 |

---

## Key Insights

- Neural network outperformed logistic regression in both accuracy and loss, suggesting it captured non-linear patterns in the data.
- Threshold 0.6 gave the best balance — ~85% precision while retaining ~78% of profitable trades.
- Identical optimizers and splits ensured the performance gap was due to architecture, not evaluation inconsistencies.
- Data leakage risk was resolved by enforcing a single train/test split (`train_and_compare.py`) before training either model.
- Real market data is rate-limited and slow to accumulate, so `generate_data_v2.py` models underpriced "deals" explicitly to produce a more realistic training signal than the original generator.

---

## Tech Stack

`Python` `PyTorch` `scikit-learn` `pandas` `NumPy` `Flask` `requests`

## How to Run

1. **Generate data** (synthetic, recommended for trying it out):
   ```bash
   python generate_data_v2.py
   ```
   Creates `fake_data.csv`. (Or run `generate_data.py` for the original, simpler version.)

   To collect real data instead, run `python collect_data.py` — this polls the Steam Community Market on a loop and writes to `market_data.csv`.

2. **Train & compare:**
   ```bash
   python train_and_compare.py
   ```
   Trains both models, prints a comparison table, and saves `neural_net_model.pth`, `logistic_model.pth` and `scaler.pkl`.

3. **Serve predictions via API:**
   ```bash
   python app.py
   ```
   Then `POST /predict` with `{"features": [price_ratio, num_listings, daily_volume, spread, momentum]}` to get back a `decision` ("buy"/"skip") and `confidence` score.
