from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import mysql.connector
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# ===============================
# Load ML model
# ===============================
MODEL_PATH = "models/fraud_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

data = joblib.load(MODEL_PATH)
model = data["model"]
features = data["features"]
scaler = data["scaler"]

# ===============================
# MySQL configuration
# ===============================
db_config = {
    "host": "localhost",
    "user": "your_mysql_username",
    "password": "your_mysql_password",
    "database": "fraud_db"
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# ===============================
# Home route
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

# ===============================
# Single transaction prediction
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs in the correct order
        input_values = [float(request.form.get(col)) for col in features]
        input_df = pd.DataFrame([input_values], columns=features)

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prediction = "Fraud" if pred == 1 else "OK"

        # Save to MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = """
            INSERT INTO single_checks (amount, time, hour, prediction)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (
            float(request.form.get("Amount")),
            float(request.form.get("Time")),
            int(request.form.get("Hour")),
            prediction
        ))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===============================
# Batch CSV upload prediction
# ===============================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        df = pd.read_csv(file)

        # Check all features exist
        missing_cols = [c for c in features if c not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        X = df[features]
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        df["prediction"] = ["Fraud" if p == 1 else "OK" for p in preds]

        # Save to MySQL
        conn = get_db_connection()
        cursor = conn.cursor()
        for idx, row in df.iterrows():
            txn_id = row.get("TransactionID", f"TXN{idx+1}")
            sql = """
                INSERT INTO batch_predictions (txn_id, amount, time, hour, prediction)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                txn_id,
                float(row["Amount"]),
                float(row["Time"]),
                int(row["Hour"]),
                row["prediction"]
            ))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ===============================
# Run Flask app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
