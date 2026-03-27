# Heart Disease Prediction Web App

A machine learning web application that predicts the risk of heart disease 
based on 13 clinical parameters.

## Live Demo
https://heart-disease-predictor-ru1m.onrender.com

## Problem Statement
Heart disease is the leading cause of death globally. Early prediction 
using patient data can help doctors take preventive action and save lives.

## Model Performance
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 79.51%   |
| Random Forest       | 98.54%   |
| XGBoost             | 98.54%   |

Random Forest was selected as the final model.

## Tech Stack
- Python 3.13
- Scikit-learn
- XGBoost
- Flask
- HTML, CSS, Bootstrap 5
- Deployed on Render

## Dataset
- UCI Heart Disease Dataset
- 1025 patient records
- 13 clinical features (age, cholesterol, blood pressure, etc.)

## Features
- Takes 13 clinical inputs from the user
- Predicts High Risk / Low Risk with confidence percentage
- Trained on real medical data
- Live deployed with public URL

## How to Run Locally
```bash
git clone https://github.com/Pravin-Singh-Charan-gh/heart-disease-predictor.git
cd heart-disease-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in your browser.

## Project Structure
```
heart-disease-predictor/
├── app.py              # Flask web server
├── train.py            # Model training script
├── model.pkl           # Saved trained model
├── scaler.pkl          # Saved scaler
├── heart.csv           # Dataset
├── requirements.txt    # Dependencies
├── Procfile            # Render deployment config
└── templates/
    └── index.html      # Frontend UI
```

## Author
Pravin Singh Charan
GitHub: https://github.com/Pravin-Singh-Charan-gh