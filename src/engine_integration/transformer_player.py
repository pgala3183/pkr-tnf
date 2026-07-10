"""PyPokerEngine player backed by a trained poker-transformer checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from pypokerengine.players import BasePokerPlayer

from poker_transformer.model.transformer import ModelConfig, PokerTransformer
from poker_transformer.tokenizer.encode import (
    _amount_for_size_label,
    encode_action,
    encode_hand_start,
)
from poker_transformer.tokenizer.vocab import Vocabulary

STREET_MAP = {
    "preflop": "PREFLOP",
    "flop": "FLOP",
    "turn": "TURN",
    "river": "RIVER",
}

FORCED_ACTIONS = frozenset({"SMALLBLIND", "BIGBLIND", "ANTE"})
VOLUNTARY_ACTIONS = frozenset({"FOLD", "CALL", "RAISE"})

PolicyMode = Literal["greedy", "sampled"]


@dataclass
class ParsedActionToken:
    token_id: int
    action_type: str
    size_bucket: str | None = None


def _pot_size(round_state: dict[str, Any]) -> int:
    pot = round_state["pot"]
    total = int(pot["main"]["amount"])
    for side in pot.get("side", []):
        total += int(side["amount"])
    return total


def _uuid_to_position(player_uuid: str, round_state: dict[str, Any]) -> str:
    seats = round_state["seats"]
    sb_pos = round_state["small_blind_pos"]
    bb_pos = round_state["big_blind_pos"]
    for index, seat in enumerate(seats):
        if seat["uuid"] == player_uuid:
            if index == sb_pos:
                return "SB"
            if index == bb_pos:
                return "BB"
    raise ValueError(f"Could not resolve position for player {player_uuid}")


def _normalize_action(
    raw_action: str,
    amount: float,
    wager_amount: float,
    street: str,
    aggression_on_street: set[str],
    *,
    is_all_in: bool = False,
) -> str | None:
    action = raw_action.upper()
    if action in FORCED_ACTIONS:
        return "CALL"
    if action == "FOLD":
        return "FOLD"
    if action == "CALL":
        if is_all_in:
            return "ALL_IN"
        if wager_amount == 0:
            return "CHECK"
        return "CALL"
    if action == "RAISE":
        if is_all_in:
            return "ALL_IN"
        if street not in aggression_on_street:
            return "BET"
        return "RAISE"
    return None


def _find_valid_action(valid_actions: list[dict[str, Any]], action_name: str) -> dict[str, Any] | None:
    for action in valid_actions:
        if action["action"] == action_name:
            return action
    return None


def load_transformer_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[PokerTransformer, ModelConfig]:
    """Load a trained model checkpoint saved by the training loop."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**payload["model_config"])
    model = PokerTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, model_cfg


class TransformerPlayer(BasePokerPlayer):
    """
    Poker bot that predicts the next tokenizer action from hand history.

    Token reconstruction matches ``src/training/self_play.py`` so inference
    sees the same sequences the model was trained on.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        policy: PolicyMode = "greedy",
        temperature: float = 1.0,
        device: str | None = None,
        vocab: Vocabulary | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.temperature = max(temperature, 1e-6)
        self.vocab = vocab or Vocabulary.from_config()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.model_config = load_transformer_checkpoint(checkpoint_path, self.device)
        if seed is not None:
            torch.manual_seed(seed)

        self.uuid = ""
        self._big_blind = 0
        self._initial_stacks: dict[str, int] = {}
        self._player_names: dict[str, str] = {}

    def declare_action(
        self,
        valid_actions: list[dict[str, Any]],
        hole_card: list[str],
        round_state: dict[str, Any],
    ) -> tuple[str, int]:
        del hole_card  # hole cards are not part of the action tokenizer vocabulary yet.

        token_ids = self._encode_hand_so_far(round_state)
        if len(token_ids) > self.model_config.block_size:
            token_ids = token_ids[-self.model_config.block_size :]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            action_logits, _ = self.model(input_ids)

        next_logits = action_logits[0, len(token_ids) - 1]
        legal_tokens = self._legal_action_tokens(round_state, valid_actions)
        if not legal_tokens:
            return self._fallback_action(valid_actions)

        masked_logits = torch.full_like(next_logits, float("-inf"))
        for parsed in legal_tokens:
            masked_logits[parsed.token_id] = next_logits[parsed.token_id]

        chosen = self._select_token(masked_logits)
        return self._token_to_engine_action(chosen, valid_actions, round_state)

    def receive_game_start_message(self, game_info: dict[str, Any]) -> None:
        small_blind = int(game_info["rule"]["small_blind_amount"])
        self._big_blind = small_blind * 2

    def receive_round_start_message(
        self,
        round_count: int,
        hole_card: list[str],
        seats: list[dict[str, Any]],
    ) -> None:
        del round_count, hole_card
        self._initial_stacks = {seat["name"]: int(seat["stack"]) for seat in seats}
        self._player_names = {seat["uuid"]: seat["name"] for seat in seats}

    def receive_street_start_message(self, street: str, round_state: dict[str, Any]) -> None:
        del street, round_state

    def receive_game_update_message(self, new_action: dict[str, Any], round_state: dict[str, Any]) -> None:
        del new_action, round_state

    def receive_round_result_message(
        self,
        winners: list[dict[str, Any]],
        hand_info: list[dict[str, Any]],
        round_state: dict[str, Any],
    ) -> None:
        del winners, hand_info, round_state
        self._initial_stacks = {}
        self._player_names = {}

    def _encode_hand_so_far(self, round_state: dict[str, Any]) -> list[int]:
        hero_name = self._player_names.get(self.uuid)
        if hero_name is None:
            raise RuntimeError("TransformerPlayer uuid was not registered via round_start_message")

        names = list(self._initial_stacks.keys())
        if len(names) == 2:
            hero_stack = self._initial_stacks[names[0]]
            villain_stack = self._initial_stacks[names[1]]
            if hero_name == names[1]:
                hero_stack, villain_stack = villain_stack, hero_stack
        else:
            hero_stack = self._initial_stacks.get(hero_name, 1000)
            villain_stack = max(self._initial_stacks.values(), default=1000)

        token_ids = [
            encode_hand_start(
                {
                    "hero_stack": hero_stack,
                    "villain_stack": villain_stack,
                    "big_blind": self._big_blind,
                },
                self.vocab,
            )
        ]

        stacks = dict(self._initial_stacks)
        pot = 0.0
        aggression_on_street: set[str] = set()
        logged_action_keys: set[tuple[str, str, str, float]] = set()

        for street_key in ("preflop", "flop", "turn", "river"):
            street = STREET_MAP[street_key]
            histories = round_state.get("action_histories", {}).get(street_key, [])
            for history_action in histories:
                player_uuid = history_action.get("uuid")
                if player_uuid is None:
                    continue

                raw_action = str(history_action["action"]).upper()
                if raw_action not in FORCED_ACTIONS | VOLUNTARY_ACTIONS:
                    continue

                dedupe_key = (
                    street,
                    player_uuid,
                    raw_action,
                    float(history_action.get("amount", 0)),
                )
                if dedupe_key in logged_action_keys:
                    continue
                logged_action_keys.add(dedupe_key)

                wager_amount = float(
                    history_action.get(
                        "add_amount",
                        history_action.get("paid", history_action.get("amount", 0)),
                    )
                )
                pre_pot = max(pot, 1.0)
                pot += wager_amount

                player_name = self._player_names.get(player_uuid, player_uuid)
                current_stack = stacks.get(player_name, 0)
                replay_state = {
                    "street": street_key,
                    "seats": [
                        {
                            "uuid": uuid,
                            "name": name,
                            "stack": stacks.get(name, 0),
                            "state": "allin" if stacks.get(name, 0) == 0 else "participating",
                        }
                        for uuid, name in self._player_names.items()
                    ],
                    "small_blind_pos": round_state["small_blind_pos"],
                    "big_blind_pos": round_state["big_blind_pos"],
                }
                position = _uuid_to_position(player_uuid, replay_state)
                is_all_in = raw_action in {"CALL", "RAISE"} and wager_amount >= current_stack > 0

                normalized = _normalize_action(
                    raw_action,
                    float(history_action.get("amount", wager_amount)),
                    wager_amount,
                    street,
                    aggression_on_street,
                    is_all_in=is_all_in,
                )
                if normalized is None:
                    continue

                if normalized in {"BET", "RAISE"}:
                    aggression_on_street.add(street)

                hero_stack_now, villain_stack_now = self._hero_villain_stacks_from_dict(stacks, hero_name)
                token_ids.append(
                    encode_action(
                        {"action_type": normalized, "amount": wager_amount},
                        {
                            "street": street,
                            "position": position,
                            "pot": pre_pot,
                            "big_blind": self._big_blind,
                            "hero_stack": hero_stack_now,
                            "villain_stack": villain_stack_now,
                        },
                        self.vocab,
                    )
                )

                stacks[player_name] = max(stacks.get(player_name, 0) - int(wager_amount), 0)

        return token_ids

    def _hero_villain_stacks_from_dict(
        self,
        stacks: dict[str, int],
        hero_name: str,
    ) -> tuple[int, int]:
        hero_stack = stacks.get(hero_name, 0)
        villain_stack = 0
        for name, stack in stacks.items():
            if name != hero_name:
                villain_stack = stack
        return hero_stack, villain_stack

    def _hero_villain_stacks(self, round_state: dict[str, Any]) -> tuple[int, int]:
        hero_name = self._player_names[self.uuid]
        stacks = {seat["name"]: int(seat["stack"]) for seat in round_state["seats"]}
        return self._hero_villain_stacks_from_dict(stacks, hero_name)

    def _legal_action_tokens(
        self,
        round_state: dict[str, Any],
        valid_actions: list[dict[str, Any]],
    ) -> list[ParsedActionToken]:
        street = STREET_MAP[round_state["street"]]
        position = _uuid_to_position(self.uuid, round_state)
        pot = float(max(_pot_size(round_state), 1))
        hero_stack, _ = self._hero_villain_stacks(round_state)

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
                legal.append(ParsedActionToken(token_id, action_type))
            elif action_type == "CHECK" and call_action is not None and int(call_action["amount"]) == 0:
                legal.append(ParsedActionToken(token_id, action_type))
            elif action_type == "CALL" and call_action is not None and int(call_action["amount"]) > 0:
                legal.append(ParsedActionToken(token_id, action_type))
            elif action_type == "ALL_IN" and self._all_in_is_legal(call_action, raise_action, hero_stack):
                legal.append(ParsedActionToken(token_id, action_type))
            elif action_type in {"BET", "RAISE"} and raise_action is not None:
                size_bucket = parsed.get("size_bucket")
                if size_bucket and self._raise_bucket_is_legal(
                    size_bucket,
                    pot,
                    raise_action["amount"],
                    hero_stack,
                ):
                    legal.append(
                        ParsedActionToken(token_id, action_type, size_bucket=size_bucket)
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

    def _select_token(self, masked_logits: torch.Tensor) -> ParsedActionToken:
        if self.policy == "greedy":
            token_id = int(torch.argmax(masked_logits).item())
        else:
            finite = masked_logits.clone()
            finite[finite == float("-inf")] = -1e9
            probs = F.softmax(finite / self.temperature, dim=-1)
            token_id = int(torch.multinomial(probs, num_samples=1).item())

        token = self.vocab.token_for(token_id)
        parsed = self.vocab.parse_token(token)
        return ParsedActionToken(
            token_id=token_id,
            action_type=parsed["action_type"],
            size_bucket=parsed.get("size_bucket"),
        )

    def _token_to_engine_action(
        self,
        chosen: ParsedActionToken,
        valid_actions: list[dict[str, Any]],
        round_state: dict[str, Any],
    ) -> tuple[str, int]:
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
            return self._fallback_action(valid_actions)

        if chosen.action_type in {"BET", "RAISE"} and raise_action is not None:
            pot = float(max(_pot_size(round_state), 1))
            min_raise = int(raise_action["amount"]["min"])
            max_raise = int(raise_action["amount"]["max"])
            if chosen.size_bucket == "ALL_IN":
                amount = max_raise
            else:
                amount = int(round(_amount_for_size_label(chosen.size_bucket or "100-150%", pot)))
            amount = max(min_raise, min(amount, max_raise))
            return "raise", amount

        return self._fallback_action(valid_actions)

    @staticmethod
    def _fallback_action(valid_actions: list[dict[str, Any]]) -> tuple[str, int]:
        call_action = _find_valid_action(valid_actions, "call")
        if call_action is not None:
            return "call", int(call_action["amount"])
        fold_action = _find_valid_action(valid_actions, "fold")
        if fold_action is not None:
            return "fold", 0
        return valid_actions[0]["action"], int(valid_actions[0].get("amount", 0) or 0)
