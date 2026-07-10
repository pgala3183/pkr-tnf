"""Track hand history in the demo app for transformer API requests."""

from __future__ import annotations

from poker_transformer.serving.demo_adapter import (
    DemoActionRecord,
    DemoHandState,
    demo_options_to_valid_actions,
    normalize_demo_decision,
    street_from_common_cards,
    street_token_from_common_cards,
)


class HandHistoryTracker:
    """
    Accumulates tokenizer-aligned actions during a demo hand.

    Perspective is always the transformer bot as hero (bot_a in training).
    """

    actions: list[DemoActionRecord] = []
    aggression_on_street: set[str] = set()
    sb_name: str = ""
    bb_name: str = ""
    bot_name: str = ""
    human_name: str = ""
    initial_stacks: dict[str, int] = {}
    current_street: str = "preflop"

    @classmethod
    def start_hand(
        cls,
        *,
        bot_name: str,
        human_name: str,
        sb_name: str,
        bb_name: str,
        stacks: dict[str, int],
    ) -> None:
        cls.actions = []
        cls.aggression_on_street = set()
        cls.bot_name = bot_name
        cls.human_name = human_name
        cls.sb_name = sb_name
        cls.bb_name = bb_name
        cls.initial_stacks = dict(stacks)
        cls.current_street = "preflop"

    @classmethod
    def set_street(cls, common_cards: list[str] | None) -> None:
        cls.current_street = street_from_common_cards(common_cards)
        cls.aggression_on_street = set()

    @classmethod
    def position_for(cls, player_name: str) -> str:
        return "SB" if player_name == cls.sb_name else "BB"

    @classmethod
    def _stacks_for_record(cls, bot_stack: int, human_stack: int) -> tuple[int, int]:
        return bot_stack, human_stack

    @classmethod
    def record_blind(cls, player_name: str, amount: float, pot_before: float) -> None:
        street = street_token_from_common_cards(None)
        hero_stack, villain_stack = cls._stacks_for_record(
            cls.initial_stacks.get(cls.bot_name, 0),
            cls.initial_stacks.get(cls.human_name, 0),
        )
        cls.actions.append(
            DemoActionRecord(
                street=street,
                position=cls.position_for(player_name),  # type: ignore[arg-type]
                action_type="CALL",
                amount=amount,
                pot_before=max(pot_before, 1.0),
                hero_stack=hero_stack,
                villain_stack=villain_stack,
            )
        )

    @classmethod
    def record_decision(
        cls,
        *,
        player_name: str,
        decision: str,
        chips: int,
        pot_before: float,
        bot_stack: int,
        human_stack: int,
        call_value: int,
    ) -> None:
        street = street_token_from_common_cards(
            None if cls.current_street == "preflop" else _board_from_street(cls.current_street)
        )
        normalized = normalize_demo_decision(
            decision,
            chips,
            call_value=call_value,
            player_stack=bot_stack if player_name == cls.bot_name else human_stack,
            aggression_on_street=cls.aggression_on_street,
            street=street,
        )
        if normalized is None:
            return

        if normalized in {"BET", "RAISE"}:
            cls.aggression_on_street.add(street)

        hero_stack, villain_stack = cls._stacks_for_record(bot_stack, human_stack)
        cls.actions.append(
            DemoActionRecord(
                street=street,
                position=cls.position_for(player_name),  # type: ignore[arg-type]
                action_type=normalized,
                amount=float(chips if decision in {"raise", "call", "all-in"} else 0),
                pot_before=max(pot_before, 1.0),
                hero_stack=hero_stack,
                villain_stack=villain_stack,
            )
        )

    @classmethod
    def build_bot_state(
        cls,
        *,
        bot_player,
        human_player,
        common_cards: list[str] | None,
        dict_options: dict[str, bool],
        call_value: int,
        min_raise: int,
        max_raise: int,
        pot_size: float,
        pot_before: float,
        big_blind: int,
    ) -> DemoHandState:
        del pot_before  # history already stores per-action pot_before values
        return DemoHandState(
            street=street_from_common_cards(common_cards),
            hero_position=cls.position_for(bot_player.name),  # type: ignore[arg-type]
            action_history=list(cls.actions),
            hole_cards=list(bot_player.cards or []),
            valid_actions=demo_options_to_valid_actions(
                dict_options,
                call_value,
                min_raise,
                max_raise,
            ),
            pot_size=max(float(pot_size), 1.0),
            hero_stack=int(bot_player.stack),
            villain_stack=int(human_player.stack),
            big_blind=int(big_blind),
            initial_hero_stack=int(cls.initial_stacks.get(cls.bot_name, bot_player.stack)),
            initial_villain_stack=int(cls.initial_stacks.get(cls.human_name, human_player.stack)),
        )


def _board_from_street(street: str) -> list[str]:
    """Placeholder board length marker — only count matters for street token."""
    if street == "flop":
        return ["__", "__", "__"]
    if street == "turn":
        return ["__"] * 4
    if street == "river":
        return ["__"] * 5
    return []
