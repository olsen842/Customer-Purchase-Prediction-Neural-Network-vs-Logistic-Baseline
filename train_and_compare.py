import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib




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

#change epochs to 640 because we got overfitting after 640 epochs and loss started to increase
for epoch in range(640):
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

    y_predicted_cls = preds


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

for epoch in range(640):
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
    lr_y_predicted_cls = lr_test_preds



    print(f"\n--- Logistic Regression Results ---")
    print(f"Accuracy: {lr_accuracy.item() * 100:.2f}%")

    lr_pred_np = lr_y_predicted_cls.cpu().numpy().ravel().astype(int)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test_np, lr_pred_np, target_names=['Skip', 'Buy']))


joblib.dump(scaler, 'scaler.pkl')

th.save(logistic_model.state_dict(), 'logistic_model.pth')

th.save(model.state_dict(), 'neural_net_model.pth')




print("\n--- Model Comparison ---")
print(f"{'Model':<15} | {'Accuracy':<10} | {'Final Loss':<10}")
print(f"{'Neural Net':<15} | {nn_accuracy*100:.2f}%      | {nn_final_loss:.4f}")
print(f"{'Logistic Reg':<15} | {lr_accuracy.item()*100:.2f}%      | {lr_final_loss:.4f}")
