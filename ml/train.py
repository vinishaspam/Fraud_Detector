import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("data/creditcard.csv")

TARGET_COL = "Class"  # Fraud label
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# ===============================
# 2. Scale features
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 3. Train Random Forest
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

# ===============================
# 4. Save model + scaler + features
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump({
    "model": model,
    "features": list(X.columns),
    "scaler": scaler
}, "models/fraud_model.joblib")

print("✅ Model saved successfully!")
