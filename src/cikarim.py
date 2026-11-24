"""Kayıtlı modelle müşteri puanlama. API bunu kullanıyor."""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from .acikla import Aciklayici
from .kampanya import musteri_kari
from .veri import KOK, bol, ozellik_ekle, veri_oku

MODEL_KLASORU = Path(os.environ.get("MODEL_KLASORU", KOK / "models"))
ARKA_PLAN = MODEL_KLASORU / "arka_plan.csv"


def risk_seviyesi(p):
    return "yüksek" if p >= 0.5 else "orta" if p >= 0.25 else "düşük"


class Puanlayici:
    def __init__(self):
        self.model = joblib.load(MODEL_KLASORU / "model.joblib")
        self.bilgi = json.loads((MODEL_KLASORU / "model_bilgisi.json").read_text(encoding="utf-8"))
        self.aciklayici = Aciklayici(self.model, pd.read_csv(ARKA_PLAN))

    def puanla(self, kayitlar):
        """kayitlar: ham müşteri sözlükleri (Telco sütunları, SeniorCitizen bool)."""
        df = pd.DataFrame(kayitlar)
        df["SeniorCitizen"] = df["SeniorCitizen"].map({True: "Yes", False: "No"})
        X = ozellik_ekle(df)
        p = self.model.predict_proba(X)[:, 1]
        kar = musteri_kari(p, df["MonthlyCharges"].values)
        nedenler = self.aciklayici.nedenler(X)
        return [{"churn_olasiligi": round(float(pi), 4), "risk": risk_seviyesi(pi),
                 "kampanya_onerisi": bool(ki > 0), "beklenen_kampanya_kari": round(float(ki), 2),
                 "nedenler": n} for pi, ki, n in zip(p, kar, nedenler)]


def arka_plan_kaydet(adet=200):
    """SHAP için eğitim setinden küçük bir örneklem (API veri setine ihtiyaç duymasın)."""
    egitim, _, _ = bol(veri_oku())
    ozellik_ekle(egitim).sample(adet, random_state=0).to_csv(ARKA_PLAN, index=False)


if __name__ == "__main__":
    arka_plan_kaydet()
