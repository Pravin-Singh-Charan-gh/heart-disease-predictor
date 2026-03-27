from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[
        data['age'],      data['sex'],     data['cp'],
        data['trestbps'], data['chol'],    data['fbs'],
        data['restecg'],  data['thalach'], data['exang'],
        data['oldpeak'],  data['slope'],   data['ca'],
        data['thal']
    ]])
    scaled = scaler.transform(features)
    result = model.predict(scaled)[0]
    prob   = model.predict_proba(scaled)[0][1]
    return jsonify({
        'prediction': int(result),
        'probability': round(float(prob) * 100, 1),
        'label': 'High Risk' if result == 1 else 'Low Risk'
    })

if __name__ == '__main__':
    app.run(debug=True)