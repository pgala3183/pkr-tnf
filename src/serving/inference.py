"""ONNX Runtime inference for poker-transformer action prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from poker_transformer.model.transformer import load_model_config
from poker_transformer.serving.export_onnx import INT8_NAME, PROJECT_ROOT
from poker_transformer.tokenizer.encode import (
    _amount_for_size_label,
    encode_action,
    encode_hand_start,
)
from poker_transformer.tokenizer.vocab import Vocabulary

DEFAULT_ONNX_PATH = PROJECT_ROOT / "checkpoints" / "onnx" / INT8_NAME

STREET_API_TO_TOKEN = {
    "preflop": "PREFLOP",
    "flop": "FLOP",
    "turn": "TURN",
    "river": "RIVER",
}


@dataclass(frozen=True)
class ParsedActionToken:
    token_id: int
    token: str
    action_type: str
    size_bucket: str | None = None


@dataclass(frozen=True)
class PredictResult:
    action: str
    amount: int
    action_probabilities: list[dict[str, Any]]
    win_probability: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _find_valid_action(valid_actions: list[dict[str, Any]], action_name: str) -> dict[str, Any] | None:
    for action in valid_actions:
        if action["action"] == action_name:
            return action
    return None


class OnnxPredictor:
    """Loads quantized ONNX once and serves /predict requests."""

    def __init__(self, model_path: str | Path = DEFAULT_ONNX_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.vocab = Vocabulary.from_config()
        self.model_config = load_model_config()
        self.block_size = self.model_config.block_size

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def encode_hand(self, payload: dict[str, Any]) -> list[int]:
        """Build tokenizer ids from API hand-state payload."""
        initial_hero = payload.get("initial_hero_stack", payload["hero_stack"])
        initial_villain = payload.get("initial_villain_stack", payload["villain_stack"])
        big_blind = payload["big_blind"]

        token_ids = [
            encode_hand_start(
                {
                    "hero_stack": initial_hero,
                    "villain_stack": initial_villain,
                    "big_blind": big_blind,
                },
                self.vocab,
            )
        ]

        for item in payload.get("action_history", []):
            token_ids.append(
                encode_action(
                    {"action_type": item["action_type"], "amount": item["amount"]},
                    {
                        "street": item["street"],
                        "position": item["position"],
                        "pot": max(float(item["pot_before"]), 1.0),
                        "big_blind": big_blind,
                        "hero_stack": item.get("hero_stack", initial_hero),
                        "villain_stack": item.get("villain_stack", initial_villain),
                    },
                    self.vocab,
                )
            )

        return token_ids

    def legal_action_tokens(self, payload: dict[str, Any]) -> list[ParsedActionToken]:
        street = STREET_API_TO_TOKEN[payload["street"]]
        position = payload["position"]
        pot = float(max(payload["pot_size"], 1))
        hero_stack = int(payload["hero_stack"])
        valid_actions = payload["valid_actions"]

        fold_action = _find_valid_action(valid_actions, "fold")
        call_action = _find_valid_action(valid_actions, "call")
        raise_action = _find_valid_action(valid_actions, "raise")

        legal: list[ParsedActionToken] = []
        for token_id in range(self.vocab.size):
            token = self.vocab.token_for(token_id)
            parsed = self.vocab.parse_token(token)
            if parsed.get("kind") != "action":
                continue
            if parsed["street"] != street or parsed["position"] != position:
                continue

            action_type = parsed["action_type"]
            if action_type == "FOLD" and fold_action is not None:
                legal.append(ParsedActionToken(token_id, token, action_type))
            elif action_type == "CHECK" and call_action is not None and int(call_action["amount"]) == 0:
                legal.append(ParsedActionToken(token_id, token, action_type))
            elif action_type == "CALL" and call_action is not None and int(call_action["amount"]) > 0:
                legal.append(ParsedActionToken(token_id, token, action_type))
            elif action_type == "ALL_IN" and self._all_in_is_legal(call_action, raise_action, hero_stack):
                legal.append(ParsedActionToken(token_id, token, action_type))
            elif action_type in {"BET", "RAISE"} and raise_action is not None:
                size_bucket = parsed.get("size_bucket")
                if size_bucket and self._raise_bucket_is_legal(
                    size_bucket,
                    pot,
                    raise_action["amount"],
                    hero_stack,
                ):
                    legal.append(
                        ParsedActionToken(token_id, token, action_type, size_bucket=size_bucket)
                    )

        return legal

    @staticmethod
    def _all_in_is_legal(
        call_action: dict[str, Any] | None,
        raise_action: dict[str, Any] | None,
        hero_stack: int,
    ) -> bool:
        if raise_action is not None:
            return int(raise_action["amount"]["max"]) >= hero_stack > 0
        if call_action is not None:
            return int(call_action["amount"]) >= hero_stack > 0
        return False

    @staticmethod
    def _raise_bucket_is_legal(
        size_bucket: str,
        pot: float,
        raise_bounds: dict[str, int],
        hero_stack: int,
    ) -> bool:
        min_raise = int(raise_bounds["min"])
        max_raise = int(raise_bounds["max"])
        if size_bucket == "ALL_IN":
            return max_raise >= hero_stack > 0
        amount = int(round(_amount_for_size_label(size_bucket, pot)))
        amount = max(min_raise, min(amount, max_raise))
        return min_raise <= amount <= max_raise

    def token_to_engine_action(
        self,
        chosen: ParsedActionToken,
        payload: dict[str, Any],
    ) -> tuple[str, int]:
        valid_actions = payload["valid_actions"]
        fold_action = _find_valid_action(valid_actions, "fold")
        call_action = _find_valid_action(valid_actions, "call")
        raise_action = _find_valid_action(valid_actions, "raise")

        if chosen.action_type == "FOLD" and fold_action is not None:
            return "fold", 0
        if chosen.action_type == "CHECK" and call_action is not None:
            return "call", 0
        if chosen.action_type == "CALL" and call_action is not None:
            return "call", int(call_action["amount"])
        if chosen.action_type == "ALL_IN":
            if raise_action is not None:
                return "raise", int(raise_action["amount"]["max"])
            if call_action is not None:
                return "call", int(call_action["amount"])

        if chosen.action_type in {"BET", "RAISE"} and raise_action is not None:
            pot = float(max(payload["pot_size"], 1))
            min_raise = int(raise_action["amount"]["min"])
            max_raise = int(raise_action["amount"]["max"])
            if chosen.size_bucket == "ALL_IN":
                amount = max_raise
            else:
                amount = int(round(_amount_for_size_label(chosen.size_bucket or "100-150%", pot)))
            amount = max(min_raise, min(amount, max_raise))
            return "raise", amount

        call_action = _find_valid_action(valid_actions, "call")
        if call_action is not None:
            return "call", int(call_action["amount"])
        return "fold", 0

    def predict(self, payload: dict[str, Any]) -> PredictResult:
        token_ids = self.encode_hand(payload)
        if len(token_ids) > self.block_size:
            token_ids = token_ids[-self.block_size :]

        input_ids = np.array([token_ids], dtype=np.int64)
        action_logits, win_prob = self.session.run(
            self.output_names,
            {self.input_name: input_ids},
        )

        next_logits = action_logits[0, len(token_ids) - 1]
        legal_tokens = self.legal_action_tokens(payload)
        if not legal_tokens:
            action, amount = "call", int(_find_valid_action(payload["valid_actions"], "call")["amount"])
            return PredictResult(
                action=action,
                amount=amount,
                action_probabilities=[],
                win_probability=float(win_prob[0, 0]),
            )

        legal_ids = np.array([item.token_id for item in legal_tokens], dtype=np.int64)
        masked = next_logits[legal_ids]
        probs = _softmax(masked.astype(np.float64))

        best_index = int(np.argmax(probs))
        chosen = legal_tokens[best_index]
        action, amount = self.token_to_engine_action(chosen, payload)

        action_probabilities = [
            {
                "token": item.token,
                "action_type": item.action_type,
                "probability": float(probs[index]),
            }
            for index, item in enumerate(legal_tokens)
        ]
        action_probabilities.sort(key=lambda row: row["probability"], reverse=True)

        return PredictResult(
            action=action,
            amount=amount,
            action_probabilities=action_probabilities,
            win_probability=float(win_prob[0, 0]),
        )
