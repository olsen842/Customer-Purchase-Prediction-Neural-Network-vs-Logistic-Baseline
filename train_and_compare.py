import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report




df = pd.read_csv('fake_data.csv')
X = df.drop('label', axis=1).values  
y = df['label'].values.reshape(-1, 1) 

scaler = StandardScaler()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)    
X_test = scaler.transform(X_test) 

X_train = th.tensor(X_train, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)
X_test = th.tensor(X_test, dtype=th.float32)
y_test = th.tensor(y_test, dtype=th.float32)



model = nn.Sequential(nn.Linear(5, 16), 
                      nn.ReLU(),
                      nn.Linear(16, 1),
                      nn.Sigmoid())


loss_fn = nn.BCELoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

#change epochs to 700 because we got overfitting after 640 epochs and loss started to increase
for epoch in range(700):
    model.train()
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with th.no_grad():
            test_preds = model(X_test)
            test_loss = loss_fn(test_preds, y_test)
            print(f'Epoch: {epoch+1:4d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')

with th.no_grad():
    model.eval()
    preds = (model(X_test) > 0.6).float()
    accuracy = (preds == y_test).float().mean()

    y_predicted_cls = preds.round()


    n_buy = preds.sum().item()
    n_skip = (preds == 0).sum().item()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")

    print(f"Number of buyers predicted: {n_buy}")
    print(f"Number of skips predicted: {n_skip}")
    
    nn_accuracy = accuracy.item()
    nn_final_loss = loss.item()

    y_test_np = y_test.cpu().numpy().ravel().astype(int)
    y_pred_np = y_predicted_cls.cpu().numpy().ravel().astype(int)

print(f"Final Loss: {loss.item():.4f}")
print(classification_report(y_test_np, y_pred_np, target_names=['Skip', 'Buy']))


# ------------------------------------------------------------------
#                  Logistic Regression 
# ------------------------------------------------------------------

logistic_model = nn.Sequential(nn.Linear(5, 1), nn.Sigmoid())
lr_optimizer = th.optim.Adam(logistic_model.parameters(), lr=0.001)

for epoch in range(700):
    logistic_model.train()
    lr_preds = logistic_model(X_train)
    lr_loss = loss_fn(lr_preds, y_train)
    lr_optimizer.zero_grad()
    lr_loss.backward()
    lr_optimizer.step()

logistic_model.eval()
with th.no_grad():
    lr_test_preds = (logistic_model(X_test) > 0.6).float()
    lr_accuracy = (lr_test_preds == y_test).float().mean()
    lr_final_loss = lr_loss.item()
    lr_y_predicted_cls = lr_test_preds.round()



    print(f"\n--- Logistic Regression Results ---")
    print(f"Accuracy: {lr_accuracy.item() * 100:.2f}%")

    lr_pred_np = lr_y_predicted_cls.cpu().numpy().ravel()
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test_np, lr_pred_np, target_names=['Skip', 'Buy']))


print("\n--- Model Comparison ---")
print(f"{'Model':<15} | {'Accuracy':<10} | {'Final Loss':<10}")
print(f"{'Neural Net':<15} | {nn_accuracy*100:.2f}%      | {nn_final_loss:.4f}")
print(f"{'Logistic Reg':<15} | {lr_accuracy.item()*100:.2f}%      | {lr_final_loss:.4f}")



#first score Final Loss: 4.0224
#second score Final Loss: 0.3285 after i added standartscaler 
#third score  Accuracy: 89.20%
#Number of buyers predicted: 209.0
#Number of skips predicted: 291
#Final Loss: 0.3218

#after training spilt addede
# Accuracy: 86.00%
#Number of buyers predicted: 43.0
#Number of skips predicted: 57
#Final Loss: 0.3069

#after i added hidden layer and relu activation function
# Accuracy: 87.00%
#Number of buyers predicted: 44.0
#Number of skips predicted: 56
#Final Loss: 0.2831

#fix 3
# i have added the Logistic Regression model back
# so i can comapre the performance of the neural network VS logistic regression.
# the results:
# Model           | Accuracy   | Final Loss
#Neural Net      | 85.00%      | 0.2207
#Logistic Reg    | 75.00%      | 0.3969
# so the neural network outperformed logistic regression in 
# both accuracy and final loss, which suggests that the added 
# complexity of the neural network is helping it capture patterns in the data that 
# logistic regression is missing.