"""SHAP ile model açıklamaları. One-hot sütunlarının katkıları asıl özellik altında toplanıyor
(ör. Contract_Month-to-month + Contract_One year + ... -> Sözleşme).

Çalıştırma:  python -m src.acikla
"""
import joblib
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

from .veri import KATEGORIK, KOK, SAYISAL, bol, ozellik_ekle, veri_oku

TURKCE = {
    "gender": "Cinsiyet", "SeniorCitizen": "65 yaş üstü", "Partner": "Eşi var", "Dependents": "Bakmakla yükümlü",
    "InternetService": "İnternet türü", "Contract": "Sözleşme", "PaperlessBilling": "Kağıtsız fatura",
    "PaymentMethod": "Ödeme yöntemi", "PhoneService": "Telefon hattı", "MultipleLines": "Çoklu hat",
    "OnlineSecurity": "Çevrimiçi güvenlik", "OnlineBackup": "Çevrimiçi yedekleme",
    "DeviceProtection": "Cihaz koruma", "TechSupport": "Teknik destek", "StreamingTV": "TV yayını",
    "StreamingMovies": "Film yayını", "tenure": "Abonelik süresi (ay)", "MonthlyCharges": "Aylık ücret",
    "TotalCharges": "Toplam ödeme", "hizmet_sayisi": "Hizmet sayısı",
}


def ham_pipeline(model):
    """Kalibre edilmiş modelin içindeki asıl pipeline'ı çıkar."""
    e = model.estimator
    return getattr(e, "estimator", e)


class Aciklayici:
    def __init__(self, model, arka_plan):
        self.pipe = ham_pipeline(model)
        on, asil = self.pipe.named_steps["on"], self.pipe.named_steps["model"]
        Z = on.transform(arka_plan)
        if isinstance(asil, LogisticRegression):
            self.aciklayici = shap.LinearExplainer(asil, Z)
        else:
            self.aciklayici = shap.TreeExplainer(asil)
        # Dönüştürülmüş her sütunun hangi asıl özelliğe ait olduğu
        kodlayici = on.named_transformers_["kat"]
        self.gruplar = [o for o, kat in zip(KATEGORIK, kodlayici.categories_) for _ in kat] + SAYISAL

    def degerler(self, X):
        """(n, asıl özellik sayısı) SHAP değerleri (log-odds ölçeğinde)."""
        s = self.aciklayici.shap_values(self.pipe.named_steps["on"].transform(X))
        if isinstance(s, list):
            s = s[1]
        s = np.asarray(s)
        if s.ndim == 3:
            s = s[..., 1]
        ozellikler = KATEGORIK + SAYISAL
        toplam = np.zeros((len(s), len(ozellikler)))
        for j, grup in enumerate(self.gruplar):
            toplam[:, ozellikler.index(grup)] += s[:, j]
        return toplam, ozellikler

    def nedenler(self, X, adet=3):
        """Her müşteri için churn riskini en çok artıran özellikler."""
        d, ozellikler = self.degerler(X)
        sonuc = []
        for i, satir in enumerate(d):
            sira = np.argsort(-satir)[:adet]
            sonuc.append([{"ozellik": TURKCE[ozellikler[j]], "deger": str(X.iloc[i][ozellikler[j]]),
                           "etki": round(float(satir[j]), 3)} for j in sira if satir[j] > 0])
        return sonuc


def main():
    # matplotlib sadece grafik üretirken gerekiyor; API imajında kurulu değil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = joblib.load(KOK / "models" / "model.joblib")
    egitim, _, test = bol(veri_oku())
    X_e, X_t = ozellik_ekle(egitim), ozellik_ekle(test)
    a = Aciklayici(model, X_e.sample(500, random_state=0))
    d, ozellikler = a.degerler(X_t)

    ortalama = np.abs(d).mean(0)
    sira = np.argsort(ortalama)[::-1][:12][::-1]
    fig, eksen = plt.subplots(figsize=(7, 5))
    eksen.barh([TURKCE[ozellikler[j]] for j in sira], ortalama[sira], color="tab:blue")
    eksen.set_xlabel("Ortalama |SHAP| (log-odds)")
    eksen.set_title("Churn kararında en etkili özellikler (test)")
    plt.tight_layout()
    plt.savefig(KOK / "reports" / "figures" / "shap_onem.png", dpi=140)
    plt.close(fig)

    # Örnek: en riskli müşteri
    p = model.predict_proba(X_t)[:, 1]
    i = int(np.argmax(p))
    satir = d[i]
    sira = np.argsort(np.abs(satir))[::-1][:8][::-1]
    fig, eksen = plt.subplots(figsize=(7, 4.2))
    eksen.barh([f"{TURKCE[ozellikler[j]]} = {X_t.iloc[i][ozellikler[j]]}" for j in sira], satir[sira],
               color=["tab:red" if v > 0 else "tab:blue" for v in satir[sira]])
    eksen.axvline(0, color="black", lw=0.8)
    eksen.set_xlabel("Churn riskine katkı (log-odds)")
    eksen.set_title(f"Örnek müşteri: churn olasılığı %{p[i] * 100:.0f}")
    plt.tight_layout()
    plt.savefig(KOK / "reports" / "figures" / "shap_ornek_musteri.png", dpi=140)
    plt.close(fig)
    print("En etkili:", [TURKCE[ozellikler[j]] for j in np.argsort(ortalama)[::-1][:5]])


if __name__ == "__main__":
    main()
