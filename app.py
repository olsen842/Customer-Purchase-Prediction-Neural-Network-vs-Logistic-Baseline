import torch as th
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)


scaler = joblib.load('scaler.pkl')


model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
model.load_state_dict(th.load('neural_net_model.pth', map_location='cpu'))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    features = data.get('features')
    if features is None or len(features) != 5:
        return jsonify({"error": "Send 'features' as a list of 5 numbers"}), 400

    scaled = scaler.transform(np.array([features]))
    tensor = th.tensor(scaled, dtype=th.float32)

    with th.no_grad():
        probability = model(tensor).item()

    decision = "buy" if probability > 0.6 else "skip"

    return jsonify({
        "decision": decision,
        "confidence": round(probability, 4)
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)