import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEKLER = json.loads((Path(__file__).parent / "ornek_musteriler.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_saglik(istemci):
    assert istemci.get("/saglik").json()["durum"] == "calisiyor"


def test_riskli_musteri(istemci):
    s = istemci.post("/musteri", json=ORNEKLER["riskli"]).json()
    assert s["risk"] == "yüksek"
    assert s["kampanya_onerisi"] is True
    assert s["nedenler"] and s["nedenler"][0]["etki"] > 0
    assert "Abonelik süresi (ay)" in [n["ozellik"] for n in s["nedenler"]]


def test_sadik_musteri(istemci):
    s = istemci.post("/musteri", json=ORNEKLER["sadik"]).json()
    assert s["risk"] == "düşük"
    assert s["kampanya_onerisi"] is False


def test_kampanya(istemci):
    s = istemci.post("/kampanya", json=[ORNEKLER["riskli"], ORNEKLER["sadik"]]).json()
    assert s["hedeflenecek"] == 1
    assert s["toplam_beklenen_kar"] == s["sonuclar"][0]["beklenen_kampanya_kari"]


def test_gecersiz_sozlesme(istemci):
    m = dict(ORNEKLER["riskli"], Contract="Three year")
    assert istemci.post("/musteri", json=m).status_code == 422


def test_negatif_ucret(istemci):
    m = dict(ORNEKLER["riskli"], MonthlyCharges=-5)
    assert istemci.post("/musteri", json=m).status_code == 422
