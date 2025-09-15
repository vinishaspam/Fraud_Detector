import pandas as pd
import joblib
from sklearn.metrics import classification_report
from ml.utils import preprocess_df

MODEL_PATH = "models/fraud_model.pkl"
DATA_PATH = "data/creditcard.csv"

def test():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_df(df)

    y_pred = model.predict(X)
    print("🔍 Model Evaluation Report:\n")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    test()
