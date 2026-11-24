from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .veri import KATEGORIK, SAYISAL


def on_isleme(olcekle):
    return ColumnTransformer([
        ("kat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), KATEGORIK),
        ("say", StandardScaler() if olcekle else "passthrough", SAYISAL),
    ], verbose_feature_names_out=False)


def modelleri_hazirla():
    return {
        "Lojistik Regresyon": Pipeline([("on", on_isleme(True)),
                                        ("model", LogisticRegression(C=0.5, max_iter=2000))]),
        "Random Forest": Pipeline([("on", on_isleme(False)),
                                   ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=10,
                                                                    n_jobs=-1, random_state=42))]),
        "XGBoost": Pipeline([("on", on_isleme(False)),
                             ("model", XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3,
                                                     subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                                     random_state=42, n_jobs=-1))]),
        "LightGBM": Pipeline([("on", on_isleme(False)),
                              ("model", LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                                       min_child_samples=30, subsample=0.8, subsample_freq=1,
                                                       colsample_bytree=0.8, random_state=42, verbose=-1))]),
    }
