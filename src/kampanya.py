"""Elde tutma kampanyasının kârını hesaplar.

Varsayımlar (README'de farklı değerlerle de sonuç var):
  - Kampanya: hedeflenen müşteriye 1 aylık ücretsiz kullanım teklif ediliyor, ayrıca kişi başı 5 $ iletişim maliyeti.
  - Ayrılacak bir müşteri teklifle %30 ihtimalle kalıyor; kalırsa 12 aylık gelir korunmuş oluyor.
  - Zaten kalacak bir müşteri de teklifi kabul ediyor: hediye edilen ay boşa giden maliyet.
"""
import numpy as np

ILETISIM = 5.0
IKNA_ORANI = 0.30
AY = 12


def musteri_kari(churn, aylik, ikna=IKNA_ORANI, iletisim=ILETISIM):
    """Bir müşteri hedeflenirse beklenen kâr. churn: gerçek etiket (0/1) ya da olasılık."""
    tesvik = aylik
    kazanc = ikna * (AY * aylik - tesvik)          # ayrılacak müşteri ikna olursa
    kayip = tesvik                                   # kalacak müşteri hediyeyi alıyor
    return churn * kazanc - (1 - churn) * kayip - iletisim


def strateji_kari(hedef, churn, aylik, **kw):
    """hedef: hedeflenen müşteriler için True olan maske. Gerçek etiketlerle gerçekleşen kâr."""
    return float(musteri_kari(churn[hedef], aylik[hedef], **kw).sum())


def kar_egrisi(olasilik, churn, aylik, **kw):
    """Müşteriler olasılığa göre sıralanıp en riskliden başlayarak k kişi hedeflenirse kâr (k=0..n)."""
    sira = np.argsort(-olasilik)
    katki = musteri_kari(churn[sira], aylik[sira], **kw)
    return np.concatenate([[0.0], np.cumsum(katki)])
