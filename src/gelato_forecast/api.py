from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from gelato_forecast import __version__
from gelato_forecast.config import Settings
from gelato_forecast.service import ForecastService, ModelNotReadyError

app = FastAPI(
    title="Gelato Demand Forecast API",
    version=__version__,
    description="Previsão educacional de demanda de sorvetes a partir da temperatura.",
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    temperature_c: float = Field(ge=-20, le=60, description="Temperatura em graus Celsius")


class PredictionResponse(BaseModel):
    temperature_c: float
    predicted_sales: int
    prediction_interval_95: list[int]
    model_name: str
    warning: str | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()


@lru_cache(maxsize=1)
def get_forecast_service() -> ForecastService:
    settings = get_settings()
    return ForecastService(settings.model_path, settings.metadata_path)


ServiceDependency = Annotated[ForecastService, Depends(get_forecast_service)]


@app.exception_handler(ModelNotReadyError)
def model_not_ready_handler(request: Request, error: ModelNotReadyError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "type": "https://httpstatuses.com/503",
            "title": "Model not ready",
            "status": 503,
            "detail": str(error),
            "instance": str(request.url.path),
        },
        media_type="application/problem+json",
    )


@app.get("/", tags=["service"])
def root() -> dict[str, str]:
    return {
        "name": "Gelato Demand Forecast API",
        "version": __version__,
        "documentation": "/docs",
    }


@app.get("/health", tags=["service"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    ready = settings.model_path.is_file() and settings.metadata_path.is_file()
    return {"status": "healthy" if ready else "degraded", "model_ready": ready}


@app.post("/predict", response_model=PredictionResponse, tags=["forecast"])
def predict(payload: PredictionRequest, service: ServiceDependency) -> PredictionResponse:
    forecast = service.predict(payload.temperature_c)
    return PredictionResponse(
        temperature_c=forecast.temperature_c,
        predicted_sales=forecast.predicted_sales,
        prediction_interval_95=[forecast.lower_bound, forecast.upper_bound],
        model_name=forecast.model_name,
        warning=forecast.warning,
    )
