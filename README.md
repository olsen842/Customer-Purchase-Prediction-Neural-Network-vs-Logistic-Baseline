Steam Trading Bot

A machine learning project that predicts whether to buy or skip Steam market items using logistic regression and a neural network — with focus on fair model comparison and threshold tuning.

Problem Statement
Explores whether it's possible to build a trading bot that can decide if a Steam market item is worth investing in. Goals include learning PyTorch in practice, building custom datasets, and ensuring fair evaluation through strict train/test separation.

Input Features
FeatureDescriptionprice_ratioRelative price vs. mediannum_listingsNumber of sellers (supply)daily_volumeTrading activity (liquidity)spreadBuy/sell price gapmomentumPrice trend direction
Preprocessing
Train/test split 80/20 — identical for both models. Feature scaling via StandardScaler fitted on train data only. No data leakage: test set was unseen during all training.

Models
Logistic Regression (Baseline)
ParameterValueArchitecture5 → 1 (sigmoid)LossBCEOptimizerAdam lr=0.001Epochs500
Neural Network (Best model)
ParameterValueArchitecture5 → 16 → 1ActivationReLU + sigmoidOptimizerAdam lr=0.001Epochs500

Results — Unseen Test Data
ModelAccuracyPrecisionRecallTest LossLogistic Regression75.00%0.740.760.3969Neural Network ✅85.00%0.850.780.2207
Threshold Analysis — Neural Network
ThresholdAccuracyPrecisionRecallF1-ScoreBuyers predicted0.587%0.860.840.85440.6 ⭐ selected84%0.850.780.81410.782%0.890.690.7835

Key Insights

Neural network outperformed logistic regression in both accuracy and loss, suggesting it captured non-linear patterns in the data.
Threshold 0.6 gave the best balance — ~85% precision while retaining ~78% of profitable trades.
Identical optimizers and splits ensured the performance gap was due to architecture, not evaluation inconsistencies.
Data leakage risk was resolved by enforcing a single train/test split (train_and_compare.py) before training either model.


Tech Stack
Python PyTorch scikit-learn pandas NumPy
How to Run

Generate data:

bash   python generate_data.py
Creates fake_data.csv.

Train & compare:

bash   python train_and_compare.py
Trains both models and prints a comparison table.
