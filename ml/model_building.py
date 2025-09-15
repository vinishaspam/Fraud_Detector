from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

def build_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced"),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "LightGBM": lgb.LGBMClassifier()
    }
    return models
