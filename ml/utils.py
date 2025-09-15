import pandas as pd
import numpy as np

def preprocess_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Feature engineering
    if "Time" in df.columns:
        df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    else:
        df["hour"] = 0

    if "Amount" in df.columns:
        df["log_amount"] = np.log1p(df["Amount"])
    else:
        df["log_amount"] = 0.0

    df = df.fillna(0)

    X = df.drop(columns=["Class"], errors="ignore")
    y = df["Class"] if "Class" in df.columns else None

    return X, y
