"""Call poker-transformer FastAPI /predict for demo bot decisions."""

from __future__ import annotations

import logging
import os

import httpx

from app.hand_history import HandHistoryTracker
from poker_transformer.serving.demo_adapter import (
    build_predict_payload,
    predict_response_to_demo_decision,
)

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORMER_API_URL = "http://localhost:8000"


class TransformerBotClient:
    """HTTP client wrapper for the quantized transformer serving API."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self.base_url = (
            base_url or os.environ.get("TRANSFORMER_API_URL", DEFAULT_TRANSFORMER_API_URL)
        ).rstrip("/")
        self.timeout = timeout

    def decide(
        self,
        *,
        bot_player,
        human_player,
        common_cards,
        dict_options,
        call_value,
        min_raise,
        max_raise,
        pot_size,
        pot_before,
        big_blind,
    ) -> list:
        state = HandHistoryTracker.build_bot_state(
            bot_player=bot_player,
            human_player=human_player,
            common_cards=common_cards,
            dict_options=dict_options,
            call_value=call_value,
            min_raise=min_raise,
            max_raise=max_raise,
            pot_size=pot_size,
            pot_before=pot_before,
            big_blind=big_blind,
        )
        payload = build_predict_payload(state)

        try:
            response = httpx.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return predict_response_to_demo_decision(response.json())
        except Exception as exc:
            logger.warning("Transformer API unavailable (%s); falling back to check/call", exc)
            if dict_options.get("check"):
                return ["check"]
            if dict_options.get("call"):
                return ["call"]
            return ["fold"]


_transformer_client = TransformerBotClient()


def transformer_decision(
    bot_player,
    human_player,
    common_cards,
    dict_options,
    call_value,
    min_raise,
    max_raise,
    pot_size,
    pot_before,
    big_blind,
) -> list:
    """Module-level helper used by ``auction.py``."""
    return _transformer_client.decide(
        bot_player=bot_player,
        human_player=human_player,
        common_cards=common_cards,
        dict_options=dict_options,
        call_value=call_value,
        min_raise=min_raise,
        max_raise=max_raise,
        pot_size=pot_size,
        pot_before=pot_before,
        big_blind=big_blind,
    )
