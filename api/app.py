"""Müşteri kaybı (churn) tahmin servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.cikarim import Puanlayici

EvetHayir = Literal["Yes", "No"]
InternetEk = Literal["Yes", "No", "No internet service"]


class Musteri(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: bool
    Partner: EvetHayir
    Dependents: EvetHayir
    tenure: int = Field(ge=0, le=120, description="Abonelik süresi (ay)")
    PhoneService: EvetHayir
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetEk
    OnlineBackup: InternetEk
    DeviceProtection: InternetEk
    TechSupport: InternetEk
    StreamingTV: InternetEk
    StreamingMovies: InternetEk
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: EvetHayir
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    MonthlyCharges: float = Field(gt=0)
    TotalCharges: float = Field(ge=0)


class Neden(BaseModel):
    ozellik: str
    deger: str
    etki: float


class Sonuc(BaseModel):
    churn_olasiligi: float
    risk: Literal["düşük", "orta", "yüksek"]
    kampanya_onerisi: bool
    beklenen_kampanya_kari: float
    nedenler: list[Neden]


class TopluSonuc(BaseModel):
    sonuclar: list[Sonuc]
    hedeflenecek: int
    toplam_beklenen_kar: float


durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["p"] = Puanlayici()
    yield
    durum.clear()


app = FastAPI(title="Müşteri Kaybı Tahmini", lifespan=lifespan)


@app.get("/saglik")
def saglik():
    return {"durum": "calisiyor", "model": durum["p"].bilgi["model"]}


@app.post("/musteri", response_model=Sonuc)
def musteri(m: Musteri):
    return durum["p"].puanla([m.model_dump()])[0]


@app.post("/kampanya", response_model=TopluSonuc)
def kampanya(musteriler: list[Musteri]):
    """Müşteri listesini puanlar; beklenen kârı pozitif olanlar kampanyaya önerilir."""
    sonuclar = durum["p"].puanla([m.model_dump() for m in musteriler]) if musteriler else []
    hedef = [s for s in sonuclar if s["kampanya_onerisi"]]
    return {"sonuclar": sonuclar, "hedeflenecek": len(hedef),
            "toplam_beklenen_kar": round(sum(s["beklenen_kampanya_kari"] for s in hedef), 2)}
