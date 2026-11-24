from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

KOK = Path(__file__).resolve().parent.parent
VERI_DOSYASI = KOK / "data" / "Telco-Customer-Churn.csv"

HIZMETLER = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
             "TechSupport", "StreamingTV", "StreamingMovies"]
KATEGORIK = ["gender", "SeniorCitizen", "Partner", "Dependents", "InternetService", "Contract",
             "PaperlessBilling", "PaymentMethod"] + HIZMETLER
SAYISAL = ["tenure", "MonthlyCharges", "TotalCharges", "hizmet_sayisi"]


def veri_oku(yol=VERI_DOSYASI):
    df = pd.read_csv(yol)
    # TotalCharges metin olarak geliyor; boş olanlar henüz ilk faturası kesilmemiş (tenure=0) müşteriler
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def ozellik_ekle(df):
    df = df.copy()
    df["hizmet_sayisi"] = (df[HIZMETLER] == "Yes").sum(axis=1)
    return df[KATEGORIK + SAYISAL]


def bol(df, seed=42):
    """%60 eğitim, %20 doğrulama, %20 test; churn oranı her bölümde aynı kalsın."""
    egitim, kalan = train_test_split(df, test_size=0.4, stratify=df["Churn"], random_state=seed)
    dogrulama, test = train_test_split(kalan, test_size=0.5, stratify=kalan["Churn"], random_state=seed)
    return egitim, dogrulama, test
