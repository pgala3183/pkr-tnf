"""Shared hero-centric hand sequence encoding (actions + optional cards)."""

from __future__ import annotations

from typing import Any, Sequence

from poker_transformer.tokenizer.encode import (
    encode_action,
    encode_card,
    encode_deal,
    encode_hand_start,
    encode_special,
)
from poker_transformer.tokenizer.vocab import Vocabulary

STREET_ORDER = ("PREFLOP", "FLOP", "TURN", "RIVER")
BOARD_COUNTS = {"FLOP": 3, "TURN": 4, "RIVER": 5}


def encode_hand_sequence(
    *,
    hero_stack: float,
    villain_stack: float,
    big_blind: float,
    actions: Sequence[Any],
    vocab: Vocabulary,
    hero_hole: Sequence[str] | None = None,
    community_cards: Sequence[str] | None = None,
    showdown: bool = False,
    include_cards: bool = True,
    flush_street: str | None = None,
    finalize: bool = True,
) -> list[int]:
    """
    Build a token sequence for one hand from the hero's perspective.

    When ``include_cards`` is true and the vocab has card tokens:
      HAND_START, CARD, CARD, [actions with DEAL_* + board cards at street starts], ...

    ``flush_street`` appends deal+board for the current street when no action on
    that street has been recorded yet (needed for live next-action prediction).
    Set ``finalize=False`` for live next-action context (omit SHOWDOWN/HAND_END).
    """
    token_ids = [
        encode_hand_start(
            {
                "hero_stack": hero_stack,
                "villain_stack": villain_stack,
                "big_blind": big_blind,
            },
            vocab,
        )
    ]

    use_cards = (
        include_cards
        and vocab.config.include_cards
        and bool(vocab.config.deal_tokens)
    )

    if use_cards and hero_hole:
        for card in hero_hole:
            token_ids.append(encode_card(card, vocab))

    community = list(community_cards or [])
    seen_streets: set[str] = set()

    def _append_board(street: str) -> None:
        if street not in BOARD_COUNTS or street in seen_streets:
            return
        n = BOARD_COUNTS[street]
        if len(community) < n:
            return
        seen_streets.add(street)
        token_ids.append(encode_deal(street, vocab))
        if street == "FLOP":
            new_cards = community[:3]
        elif street == "TURN":
            new_cards = community[3:4]
        else:
            new_cards = community[4:5]
        for card in new_cards:
            token_ids.append(encode_card(card, vocab))

    for action in actions:
        street = _attr(action, "street")
        if use_cards:
            _append_board(street)

        token_ids.append(
            encode_action(
                {
                    "action_type": _attr(action, "action"),
                    "amount": float(_attr(action, "amount")),
                },
                {
                    "street": street,
                    "position": _attr(action, "position"),
                    "pot": max(float(_attr(action, "pot_size")), 1.0),
                    "big_blind": big_blind,
                    "hero_stack": hero_stack,
                    "villain_stack": villain_stack,
                },
                vocab,
            )
        )

    if use_cards and flush_street:
        _append_board(flush_street)

    if finalize:
        if showdown:
            token_ids.append(encode_special("SHOWDOWN", vocab))
        token_ids.append(encode_special("HAND_END", vocab))
    return token_ids


def _attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)
