"""Build and serialize the poker action tokenizer vocabulary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "tokenizer.yaml"

ACTIONS_WITHOUT_SIZE = frozenset({"FOLD", "CHECK", "CALL", "ALL_IN"})
ACTIONS_WITH_SIZE = frozenset({"BET", "RAISE"})


@dataclass(frozen=True)
class TokenizerConfig:
    action_types: tuple[str, ...]
    size_buckets: tuple[dict[str, Any], ...]
    streets: tuple[str, ...]
    positions: tuple[str, ...]
    stack_buckets: tuple[dict[str, Any], ...]
    special_tokens: tuple[str, ...]
    pad_token: str
    sizeless_action_types: frozenset[str]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TokenizerConfig":
        return cls(
            action_types=tuple(raw["action_types"]),
            size_buckets=tuple(raw["size_buckets"]),
            streets=tuple(raw["streets"]),
            positions=tuple(raw["positions"]),
            stack_buckets=tuple(raw["stack_buckets"]),
            special_tokens=tuple(raw["special_tokens"]),
            pad_token=raw["pad_token"],
            sizeless_action_types=frozenset(raw["sizeless_action_types"]),
        )


def load_config(config_path: str | Path | None = None) -> TokenizerConfig:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return TokenizerConfig.from_dict(raw)


def _hand_start_token(stack_label: str) -> str:
    return f"HAND_START|{stack_label}"


def _action_token(
    street: str,
    position: str,
    action_type: str,
    size_label: str | None = None,
) -> str:
    if size_label is None:
        return f"{street}|{position}|{action_type}"
    return f"{street}|{position}|{action_type}|{size_label}"


def build_vocabulary(config: TokenizerConfig | None = None) -> list[str]:
    """Return ordered token strings for the full flat vocabulary."""
    config = config or load_config()
    tokens: list[str] = [config.pad_token]

    for stack_bucket in config.stack_buckets:
        tokens.append(_hand_start_token(stack_bucket["label"]))

    tokens.extend(config.special_tokens)

    for street in config.streets:
        for position in config.positions:
            for action_type in config.action_types:
                if action_type in config.sizeless_action_types:
                    tokens.append(_action_token(street, position, action_type))
                elif action_type in ACTIONS_WITH_SIZE:
                    for size_bucket in config.size_buckets:
                        tokens.append(
                            _action_token(
                                street,
                                position,
                                action_type,
                                size_bucket["label"],
                            )
                        )

    return tokens


class Vocabulary:
    """Bidirectional mapping between token strings and integer ids."""

    def __init__(
        self,
        tokens: list[str],
        config: TokenizerConfig,
        *,
        version: int = 1,
    ) -> None:
        if len(tokens) != len(set(tokens)):
            duplicates = {token for token in tokens if tokens.count(token) > 1}
            raise ValueError(f"Duplicate tokens in vocabulary: {sorted(duplicates)}")

        self.version = version
        self.config = config
        self.tokens = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = dict(enumerate(self.tokens))

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> "Vocabulary":
        config = load_config(config_path)
        return cls(build_vocabulary(config), config)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.config.pad_token]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def id_for(self, token: str) -> int:
        return self.token_to_id[token]

    def token_for(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "size": self.size,
            "tokens": self.tokens,
            "token_to_id": self.token_to_id,
            "config_path": "configs/tokenizer.yaml",
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    @classmethod
    def load_json(cls, path: str | Path, config_path: str | Path | None = None) -> "Vocabulary":
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        config = load_config(config_path)
        tokens = payload["tokens"]
        expected = build_vocabulary(config)
        if tokens != expected:
            raise ValueError("Saved vocabulary does not match config/tokenizer.yaml")

        return cls(tokens, config, version=payload.get("version", 1))

    def is_hand_start(self, token: str) -> bool:
        return token.startswith("HAND_START|")

    def is_special(self, token: str) -> bool:
        return token in self.config.special_tokens or token == self.config.pad_token

    def is_action(self, token: str) -> bool:
        return not self.is_special(token) and not self.is_hand_start(token)

    def parse_token(self, token: str) -> dict[str, str]:
        if token == self.config.pad_token:
            return {"kind": "pad", "token": token}

        if token in self.config.special_tokens:
            return {"kind": "special", "token": token}

        if self.is_hand_start(token):
            return {
                "kind": "hand_start",
                "stack_bucket": token.split("|", 1)[1],
            }

        parts = token.split("|")
        if len(parts) == 3:
            street, position, action_type = parts
            return {
                "kind": "action",
                "street": street,
                "position": position,
                "action_type": action_type,
            }

        if len(parts) == 4:
            street, position, action_type, size_bucket = parts
            return {
                "kind": "action",
                "street": street,
                "position": position,
                "action_type": action_type,
                "size_bucket": size_bucket,
            }

        raise ValueError(f"Unrecognized token format: {token}")
