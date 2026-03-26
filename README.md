# Steam Trading Bot
> A machine learning project that predicts whether to buy or skip Steam market items using logistic regression and a neural network — with focus on fair model comparison and threshold tuning.

## Problem Statement

Explores whether it's possible to build a trading bot that can decide if a Steam market item is worth investing in. Goals include learning PyTorch in practice, building custom datasets, and ensuring fair evaluation through strict train/test separation.

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
| Epochs | `700` |

### Neural Network *(Best model)*

| Parameter | Value |
|---|---|
| Architecture | `5 → 16 → 1` |
| Activation | `ReLU + sigmoid` |
| Optimizer | `Adam lr=0.001` |
| Epochs | `700` |

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

---

## Tech Stack

`Python` `PyTorch` `scikit-learn` `pandas` `NumPy`

## How to Run

1. **Generate data:**
   ```bash
   python generate_data.py
   ```
   Creates `fake_data.csv`.

2. **Train & compare:**
   ```bash
   python train_and_compare.py
   ```
   Trains both models and prints a comparison table.
