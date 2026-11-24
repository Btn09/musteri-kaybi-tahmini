"""Modelleri karşılaştırır, en iyisini kalibre edip kaydeder, kampanya stratejilerini test setinde ölçer.

Çalıştırma:  python -m src.egit
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .kampanya import ILETISIM, IKNA_ORANI, kar_egrisi, musteri_kari, strateji_kari
from .modeller import modelleri_hazirla
from .veri import KOK, bol, ozellik_ekle, veri_oku

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_KLASORU = KOK / "models"
RAPOR = KOK / "reports"
GRAFIK = RAPOR / "figures"


def metrikler(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def main():
    for k in (MODEL_KLASORU, GRAFIK):
        k.mkdir(parents=True, exist_ok=True)
    df = veri_oku()
    egitim, dogrulama, test = bol(df)
    X_e, y_e = ozellik_ekle(egitim), egitim["Churn"].values
    X_d, y_d = ozellik_ekle(dogrulama), dogrulama["Churn"].values
    X_t, y_t = ozellik_ekle(test), test["Churn"].values
    print(f"Eğitim {len(y_e)} | Doğrulama {len(y_d)} | Test {len(y_t)} | churn oranı %{y_e.mean() * 100:.1f}")

    satirlar, modeller = [], modelleri_hazirla()
    for ad, model in modeller.items():
        model.fit(X_e, y_e)
        m = metrikler(y_d, model.predict_proba(X_d)[:, 1])
        satirlar.append({"model": ad, **{f"dogrulama_{k}": v for k, v in m.items()}})
        print(f"{ad:20s} ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.4f}")
    tablo = pd.DataFrame(satirlar).sort_values("dogrulama_roc_auc", ascending=False)
    secilen = tablo.iloc[0]["model"]
    print(f"Seçilen: {secilen}")

    # Kampanya kararları olasılığın kendisine dayandığı için olasılıkların gerçekçi olması gerekiyor.
    # Doğrulama setinde sigmoid kalibrasyonu yapıyorum.
    ham = modeller[secilen]
    kalibre = CalibratedClassifierCV(FrozenEstimator(ham), method="sigmoid").fit(X_d, y_d)

    p_ham_t = ham.predict_proba(X_t)[:, 1]
    p_t = kalibre.predict_proba(X_t)[:, 1]
    p_d = kalibre.predict_proba(X_d)[:, 1]
    for ad, model in modeller.items():
        tablo.loc[tablo.model == ad, "test_roc_auc"] = roc_auc_score(y_t, model.predict_proba(X_t)[:, 1])
    tablo.to_csv(RAPOR / "model_karsilastirma.csv", index=False, float_format="%.4f")

    aylik_t, aylik_d = test["MonthlyCharges"].values, dogrulama["MonthlyCharges"].values

    # Hedeflenecek müşteri oranını doğrulama setinde seçip testte aynısını uyguluyorum
    egri_d = kar_egrisi(p_d, y_d, aylik_d)
    oran = int(np.argmax(egri_d)) / len(y_d)
    k_t = int(round(oran * len(y_t)))
    sira = np.argsort(-p_t)
    hedef_sirali = np.zeros(len(y_t), bool)
    hedef_sirali[sira[:k_t]] = True
    hedef_beklenen = musteri_kari(p_t, aylik_t) > 0

    rng = np.random.default_rng(0)
    rastgele = np.mean([strateji_kari(rng.permutation(len(y_t)) < k_t, y_t, aylik_t) for _ in range(200)])
    stratejiler = {
        "Kimseye kampanya yapma": 0.0,
        "Herkese kampanya": strateji_kari(np.ones(len(y_t), bool), y_t, aylik_t),
        f"Rastgele {k_t} müşteri": rastgele,
        f"Model: en riskli %{oran * 100:.0f} ({k_t} müşteri)": strateji_kari(hedef_sirali, y_t, aylik_t),
        f"Model: beklenen kârı pozitif olanlar ({hedef_beklenen.sum()} müşteri)":
            strateji_kari(hedef_beklenen, y_t, aylik_t),
    }
    for ad, kar in stratejiler.items():
        print(f"  {ad:55s} {kar:10,.0f} $")

    duyarlilik = []
    for ikna in (0.10, 0.20, 0.30, 0.50):
        hedef = musteri_kari(p_t, aylik_t, ikna=ikna) > 0
        duyarlilik.append({"ikna_orani": ikna, "hedeflenen": int(hedef.sum()),
                           "model_kari": round(strateji_kari(hedef, y_t, aylik_t, ikna=ikna)),
                           "herkese_kari": round(strateji_kari(np.ones(len(y_t), bool), y_t, aylik_t, ikna=ikna))})

    sonuc = {
        "secilen_model": secilen,
        "test_ham": {k: round(v, 4) for k, v in metrikler(y_t, p_ham_t).items()},
        "test_kalibre": {k: round(v, 4) for k, v in metrikler(y_t, p_t).items()},
        "test_musteri": int(len(y_t)), "test_churn": int(y_t.sum()),
        "varsayimlar": {"iletisim_maliyeti": ILETISIM, "ikna_orani": IKNA_ORANI,
                        "tesvik": "1 aylık ücret", "korunan_gelir": "12 aylık ücret"},
        "hedef_orani_dogrulamadan": round(oran, 3),
        "stratejiler": {k: round(v) for k, v in stratejiler.items()},
        "ikna_orani_duyarlilik": duyarlilik,
    }
    with open(RAPOR / "sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    joblib.dump(kalibre, MODEL_KLASORU / "model.joblib", compress=3)
    with open(MODEL_KLASORU / "model_bilgisi.json", "w", encoding="utf-8") as f:
        json.dump({"model": secilen, "kalibrasyon": "sigmoid", **sonuc["varsayimlar"]}, f, ensure_ascii=False,
                  indent=2)

    grafikler(y_t, p_ham_t, p_t, aylik_t, secilen, k_t)


def grafikler(y_t, p_ham, p_kal, aylik, secilen, k_t):
    fig, eksen = plt.subplots(figsize=(5.5, 5))
    for ad, p in [("Kalibrasyon öncesi", p_ham), ("Kalibrasyon sonrası", p_kal)]:
        gercek, tahmin = calibration_curve(y_t, p, n_bins=10, strategy="quantile")
        eksen.plot(tahmin, gercek, "o-", label=f"{ad} (Brier {brier_score_loss(y_t, p):.4f})")
    eksen.plot([0, 1], [0, 1], "--", color="gray")
    eksen.set_xlabel("Tahmin edilen churn olasılığı")
    eksen.set_ylabel("Gerçekleşen churn oranı")
    eksen.set_title(f"{secilen}: kalibrasyon (test)")
    eksen.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAFIK / "kalibrasyon.png", dpi=140)
    plt.close(fig)

    egri = kar_egrisi(p_kal, y_t, aylik)
    x = np.arange(len(egri)) / len(y_t) * 100
    fig, eksen = plt.subplots(figsize=(8, 4.5))
    eksen.plot(x, egri, label="Modelin sıralamasıyla")
    eksen.plot([0, 100], [0, egri[-1]], "--", color="gray", label="Rastgele seçim (ortalama)")
    eksen.axvline(k_t / len(y_t) * 100, color="tab:red", ls=":", label="Doğrulamada seçilen oran")
    eksen.axhline(0, color="black", lw=0.8)
    eksen.set_xlabel("Hedeflenen müşteri oranı (%), en riskliden başlayarak")
    eksen.set_ylabel("Kampanya kârı ($)")
    eksen.set_title("Test setinde kampanya kârı")
    eksen.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK / "kar_egrisi.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
