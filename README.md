📊 Steam Trading Bot (ML Experiment)
🔹 One-liner
A machine learning project that predicts whether to buy or skip Steam market items using logistic regression and a neural network.

🔹 Problem Statement
This project explores whether it is possible to build a trading bot that can decide if a Steam market item is worth investing in.
The goal was not only to attempt generating ROI, but also to:
* Learn PyTorch in practice
* Become more comfortable building machine learning models
* Create and work with a custom dataset

🔹 Data
The dataset was self-generated using fake_data.py.
* Simulates Steam market price behavior and randomness
* Built using NumPy (uniform distribution)
* Saved as a CSV file (fake_data.csv)
Features:
* 5 numerical input features
Target:
* 1 = buy
* 0 = skip

🔹 Preprocessing
* Train/test split: 80/20
* Feature scaling using StandardScaler

🔹 Models
1. Logistic Regression (PyTorch)
* Architecture:
    * Linear layer (5 → 1)
    * Sigmoid activation
* Loss: Binary Cross Entropy
* Optimizer: SGD (lr=0.01)
* Epochs: 1000
Result:
* Final Loss: 0.3285 (after scaling)

2. Neural Network (PyTorch)
* Architecture:
    * Input layer (5 features)
    * Hidden layer (16 neurons, ReLU)
    * Output layer (Sigmoid)
* Loss: Binary Cross Entropy
* Optimizer: Adam (lr=0.001)
* Epochs: 700
Result:
* Accuracy: 87%
* Final Loss: 0.2831

🔹 Results
Model	Accuracy	Final Loss
Logistic Regression	N/A	0.3285
Neural Network	87%	0.2831
⚠️ Logistic regression was not evaluated on a test split, so accuracy is not directly comparable.

🔹 Key Insight
Even though the neural network is more complex, logistic regression achieved very similar loss performance.
This suggests:
* The dataset is likely linearly separable or close to it
* Increasing model complexity leads to diminishing returns
Additionally, adjusting the classification threshold to 0.6 improved prediction balance:
* Better control over predictions (buy vs skip)
* Shows that model performance depends not only on training, but also on decision thresholds

🔹 Critical Observation
The comparison is not fully fair because the models were developed step by step.
When logistic regression was implemented:
* A train/test split had not yet been introduced
This means:
* Logistic regression results may be overly optimistic
* A proper comparison requires identical evaluation methods

🔹 Tech Stack
* Python
* PyTorch
* scikit-learn
* pandas
* NumPy

🔹 What I’d Do Next
* Integrate the Steam API for real market data
* Apply proper evaluation (train/test split) to all models
* Tune neural network hyperparameters
* Experiment with additional models

