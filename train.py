import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv('heart.csv')
print("Dataset loaded! Shape:", df.shape)
print(df.head())

# Step 2: Separate input features and target
X = df.drop('target', axis=1)   # all columns except target
y = df['target']                 # target column (0=no disease, 1=disease)

# Step 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Scale the data (make all numbers on same range)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Step 5: Train 3 models and compare
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

best_acc = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"{name}: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Step 6: Save the best model and scaler
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nBest model saved! Accuracy:", round(best_acc*100, 2), "%")