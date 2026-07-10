"""FastAPI serving endpoint for poker-transformer ONNX inference."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Union

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from poker_transformer.serving.inference import DEFAULT_ONNX_PATH, OnnxPredictor

DEFAULT_MODEL_PATH = Path(os.environ.get("ONNX_MODEL_PATH", DEFAULT_ONNX_PATH))


class ActionHistoryItem(BaseModel):
    street: Literal["PREFLOP", "FLOP", "TURN", "RIVER"]
    position: Literal["SB", "BB"]
    action_type: Literal["FOLD", "CHECK", "CALL", "BET", "RAISE", "ALL_IN"]
    amount: float = Field(default=0.0, ge=0)
    pot_before: float = Field(gt=0)
    hero_stack: int | None = Field(default=None, ge=0)
    villain_stack: int | None = Field(default=None, ge=0)


class RaiseBounds(BaseModel):
    min: int = Field(ge=0)
    max: int = Field(ge=0)


class ValidAction(BaseModel):
    action: Literal["fold", "call", "raise"]
    amount: Union[int, RaiseBounds] = 0

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: int | RaiseBounds, info) -> int | RaiseBounds:
        action = info.data.get("action")
        if action == "raise" and not isinstance(value, RaiseBounds):
            raise ValueError("raise actions require amount {min, max}")
        if action in {"fold", "call"} and isinstance(value, RaiseBounds):
            raise ValueError("fold/call actions require integer amount")
        return value


class PredictRequest(BaseModel):
    street: Literal["preflop", "flop", "turn", "river"]
    position: Literal["SB", "BB"]
    action_history: list[ActionHistoryItem] = Field(default_factory=list)
    hole_cards: list[str] = Field(default_factory=list)
    valid_actions: list[ValidAction] = Field(min_length=1)
    pot_size: float = Field(gt=0)
    hero_stack: int = Field(ge=0)
    villain_stack: int = Field(ge=0)
    big_blind: int = Field(gt=0)
    initial_hero_stack: int | None = Field(default=None, ge=0)
    initial_villain_stack: int | None = Field(default=None, ge=0)


class ActionProbability(BaseModel):
    token: str
    action_type: str
    probability: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    action: Literal["fold", "call", "raise"]
    amount: int = Field(ge=0)
    action_probabilities: list[ActionProbability]
    win_probability: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_path: str


_predictor: OnnxPredictor | None = None


def get_predictor() -> OnnxPredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predictor


def create_app(model_path: Path | None = None) -> FastAPI:
    """Application factory (used by tests to inject a temp ONNX model path)."""
    resolved_path = model_path or DEFAULT_MODEL_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _predictor
        _predictor = OnnxPredictor(resolved_path)
        app.state.model_path = str(resolved_path)
        yield
        _predictor = None

    app = FastAPI(
        title="poker-transformer",
        description="Next-action prediction API backed by quantized ONNX",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=_predictor is not None,
            model_path=str(resolved_path),
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        request: PredictRequest,
        predictor: Annotated[OnnxPredictor, Depends(get_predictor)],
    ) -> PredictResponse:
        payload = request.model_dump()
        payload["valid_actions"] = [
            {
                "action": action.action,
                "amount": (
                    action.amount.model_dump()
                    if isinstance(action.amount, RaiseBounds)
                    else action.amount
                ),
            }
            for action in request.valid_actions
        ]
        payload["action_history"] = [item.model_dump() for item in request.action_history]

        try:
            result = predictor.predict(payload)
        except Exception as exc:  # pragma: no cover - unexpected inference errors
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return PredictResponse(
            action=result.action,  # type: ignore[arg-type]
            amount=result.amount,
            action_probabilities=[
                ActionProbability(**row) for row in result.action_probabilities
            ],
            win_probability=result.win_probability,
        )

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "poker_transformer.serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
