"""Generate tokenized self-play training data via PyPokerEngine."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ENGINE_ROOT = PROJECT_ROOT / "src" / "engine_integration" / "pypokerengine"


def _ensure_engine_path() -> None:
    vendored_root = str(VENDORED_ENGINE_ROOT)
    if vendored_root not in sys.path:
        sys.path.insert(0, vendored_root)


_ensure_engine_path()

from examples.players.honest_player import HonestPlayer  # noqa: E402
from examples.players.random_player import RandomPlayer  # noqa: E402
from pypokerengine.api.game import setup_config, start_poker  # noqa: E402
from pypokerengine.players import BasePokerPlayer  # noqa: E402

from poker_transformer.tokenizer.vocab import Vocabulary  # noqa: E402

STREET_MAP = {
    "preflop": "PREFLOP",
    "flop": "FLOP",
    "turn": "TURN",
    "river": "RIVER",
}

FORCED_ACTIONS = frozenset({"SMALLBLIND", "BIGBLIND", "ANTE"})
VOLUNTARY_ACTIONS = frozenset({"FOLD", "CALL", "RAISE"})


class FastHonestPlayer(HonestPlayer):
    """HonestPlayer with fewer Monte Carlo simulations for data generation speed."""

    def declare_action(self, valid_actions, hole_card, round_state):
        from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards

        community_card = round_state["community_card"]
        win_rate = estimate_hole_card_win_rate(
            nb_simulation=20,
            nb_player=self.nb_player,
            hole_card=gen_cards(hole_card),
            community_card=gen_cards(community_card),
        )
        if win_rate >= 1.0 / self.nb_player:
            action = valid_actions[1]
        else:
            action = valid_actions[0]
        return action["action"], action["amount"]


@dataclass
class RecordedAction:
    street: str
    position: str
    action: str
    amount: float
    pot_size: float
    stack_sizes: dict[str, int]
    player_name: str
    player_uuid: str


@dataclass
class HandRecord:
    hand_id: int
    actions: list[RecordedAction] = field(default_factory=list)
    hole_cards: dict[str, list[str]] = field(default_factory=dict)
    player_names: dict[str, str] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    big_blind: int = 0
    initial_stacks: dict[str, int] = field(default_factory=dict)
    community_cards: list[str] = field(default_factory=list)


class HandRecorder:
    def __init__(self) -> None:
        self.current: HandRecord | None = None
        self.completed: list[HandRecord] = []
        self._aggression_on_street: set[str] = set()
        self._logged_action_keys: set[tuple[str, str, str, float]] = set()

    def start_hand(self, hand_id: int, big_blind: int) -> None:
        self.current = HandRecord(hand_id=hand_id, big_blind=big_blind)
        self._aggression_on_street = set()
        self._logged_action_keys = set()

    def finish_hand(self) -> None:
        if self.current is not None:
            self.completed.append(self.current)
            self.current = None
        self._aggression_on_street = set()
        self._logged_action_keys = set()

    def record_round_start(
        self,
        hole_cards: dict[str, list[str]],
        player_names: dict[str, str],
        initial_stacks: dict[str, int],
    ) -> None:
        if self.current is None:
            return
        self.current.hole_cards.update(hole_cards)
        self.current.player_names.update(player_names)
        self.current.initial_stacks.update(initial_stacks)

    def record_result(self, result: dict[str, Any], round_state: dict[str, Any] | None = None) -> None:
        if self.current is None:
            return
        if round_state is not None:
            self.current.community_cards = list(round_state.get("community_card", []))
            self._replay_action_histories(round_state)
        self.current.result = result

    def _replay_action_histories(self, round_state: dict[str, Any]) -> None:
        if self.current is None:
            return

        stacks = {
            name: int(self.current.initial_stacks.get(name, 0))
            for name in self.current.player_names.values()
        }
        pot = 0.0
        self.current.actions = []
        self._aggression_on_street = set()
        self._logged_action_keys = set()

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
                if dedupe_key in self._logged_action_keys:
                    continue
                self._logged_action_keys.add(dedupe_key)

                wager_amount = float(
                    history_action.get(
                        "add_amount",
                        history_action.get("paid", history_action.get("amount", 0)),
                    )
                )
                pre_pot = max(pot, 1.0)
                pot += wager_amount

                player_name = self.current.player_names.get(player_uuid, player_uuid)
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
                        for uuid, name in self.current.player_names.items()
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
                    self._aggression_on_street,
                    is_all_in=is_all_in,
                )
                if normalized is None:
                    continue

                if normalized in {"BET", "RAISE"}:
                    self._aggression_on_street.add(street)

                stack_sizes = dict(stacks)
                self.current.actions.append(
                    RecordedAction(
                        street=street,
                        position=position,
                        action=normalized,
                        amount=float(
                            wager_amount
                            if normalized in {"BET", "RAISE", "CALL", "ALL_IN"}
                            else history_action.get("amount", 0)
                        ),
                        pot_size=pre_pot,
                        stack_sizes=stack_sizes,
                        player_name=player_name,
                        player_uuid=player_uuid,
                    )
                )

                stacks[player_name] = max(stacks.get(player_name, 0) - int(wager_amount), 0)

    def record_engine_action(self, new_action: dict[str, Any], round_state: dict[str, Any]) -> None:
        """Legacy hook for live updates; full hands are rebuilt from action_histories."""
        return


class RecordingPlayer(BasePokerPlayer):
    """Wrap a heuristic bot and log every engine action broadcast."""

    def __init__(
        self,
        inner: BasePokerPlayer,
        recorder: HandRecorder,
        name: str,
        *,
        log_updates: bool = False,
    ) -> None:
        self.inner = inner
        self.recorder = recorder
        self.name = name
        self.log_updates = log_updates
        self.uuid = ""

    def set_uuid(self, uuid: str) -> None:
        self.uuid = uuid
        if hasattr(self.inner, "set_uuid"):
            self.inner.set_uuid(uuid)
        else:
            self.inner.uuid = uuid

    def declare_action(self, valid_actions, hole_card, round_state):
        return self.inner.declare_action(valid_actions, hole_card, round_state)

    def receive_game_start_message(self, game_info):
        if hasattr(self.inner, "receive_game_start_message"):
            self.inner.receive_game_start_message(game_info)

    def receive_round_start_message(self, round_count, hole_card, seats):
        hole_cards = {}
        player_names = {}
        initial_stacks = {}
        for seat in seats:
            player_names[seat["uuid"]] = seat["name"]
            initial_stacks[seat["name"]] = int(seat["stack"])
        if hole_card:
            hole_cards[self.uuid] = list(hole_card)
        self.recorder.record_round_start(hole_cards, player_names, initial_stacks)
        if hasattr(self.inner, "receive_round_start_message"):
            self.inner.receive_round_start_message(round_count, hole_card, seats)

    def receive_street_start_message(self, street, round_state):
        if hasattr(self.inner, "receive_street_start_message"):
            self.inner.receive_street_start_message(street, round_state)

    def receive_game_update_message(self, new_action, round_state):
        if self.log_updates:
            self.recorder.record_engine_action(new_action, round_state)
        if hasattr(self.inner, "receive_game_update_message"):
            self.inner.receive_game_update_message(new_action, round_state)

    def receive_round_result_message(self, winners, hand_info, round_state):
        showdown = bool(hand_info)
        winner_names = [winner["name"] for winner in winners]
        if self.log_updates:
            self.recorder.record_result(
                {
                    "winners": winner_names,
                    "showdown": showdown,
                    "hand_info": hand_info,
                },
                round_state,
            )
        else:
            self.recorder.record_result(
                {
                    "winners": winner_names,
                    "showdown": showdown,
                    "hand_info": hand_info,
                }
            )
        if hasattr(self.inner, "receive_round_result_message"):
            self.inner.receive_round_result_message(winners, hand_info, round_state)


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


def hand_to_token_ids(
    record: HandRecord,
    vocab: Vocabulary,
    *,
    hero_name: str | None = None,
    include_cards: bool = True,
) -> list[int]:
    if not record.actions:
        return []

    names = list(record.initial_stacks.keys())
    if hero_name is None:
        hero_name = names[0] if names else None
    if hero_name is None:
        return []

    villain_name = next((n for n in names if n != hero_name), None)
    hero_stack = float(record.initial_stacks.get(hero_name, 1000))
    villain_stack = float(
        record.initial_stacks.get(villain_name, hero_stack) if villain_name else hero_stack
    )
    hole_by_name = {
        record.player_names.get(uuid, uuid): cards for uuid, cards in record.hole_cards.items()
    }

    from poker_transformer.tokenizer.hand_sequence import encode_hand_sequence

    return encode_hand_sequence(
        hero_stack=hero_stack,
        villain_stack=villain_stack,
        big_blind=float(record.big_blind),
        actions=record.actions,
        vocab=vocab,
        hero_hole=hole_by_name.get(hero_name),
        community_cards=record.community_cards,
        showdown=bool(record.result.get("showdown")),
        include_cards=include_cards,
    )


def _build_bot(
    name: str,
    *,
    checkpoint: Path | None = None,
    device: str = "cpu",
    policy: str = "greedy",
) -> BasePokerPlayer:
    if name == "honest":
        return FastHonestPlayer()
    if name == "random":
        return RandomPlayer()
    if name == "transformer":
        if checkpoint is None:
            raise ValueError("bot=transformer requires --checkpoint")
        from poker_transformer.engine_integration.transformer_player import TransformerPlayer

        return TransformerPlayer(checkpoint, policy=policy, device=device)  # type: ignore[arg-type]
    raise ValueError(f"Unknown bot: {name}")


def play_hand(
    hand_id: int,
    recorder: HandRecorder,
    *,
    initial_stack: int = 1000,
    small_blind: int = 20,
    bot_a: str = "honest",
    bot_b: str = "random",
    checkpoint: Path | None = None,
    device: str = "cpu",
    policy: str = "greedy",
) -> HandRecord:
    recorder.start_hand(hand_id, big_blind=small_blind * 2)

    config = setup_config(max_round=1, initial_stack=initial_stack, small_blind_amount=small_blind)
    config.register_player(
        name="bot_a",
        algorithm=RecordingPlayer(
            _build_bot(bot_a, checkpoint=checkpoint, device=device, policy=policy),
            recorder,
            "bot_a",
            log_updates=True,
        ),
    )
    config.register_player(
        name="bot_b",
        algorithm=RecordingPlayer(
            _build_bot(bot_b, checkpoint=checkpoint, device=device, policy=policy),
            recorder,
            "bot_b",
        ),
    )

    start_poker(config, verbose=0)
    recorder.finish_hand()
    return recorder.completed[-1]


def _hand_to_training_example(record: HandRecord, vocab: Vocabulary) -> dict[str, Any]:
    names = list(record.initial_stacks.keys())
    player_perspectives: dict[str, Any] = {}
    for name in names:
        token_ids = hand_to_token_ids(record, vocab, hero_name=name, include_cards=True)
        uuid = next((u for u, n in record.player_names.items() if n == name), name)
        player_perspectives[name] = {
            "hole_cards": record.hole_cards.get(uuid, []),
            "token_ids": token_ids,
        }

    primary = names[0] if names else "bot_a"
    token_ids = player_perspectives.get(primary, {}).get("token_ids") or hand_to_token_ids(
        record, vocab, include_cards=True
    )

    return {
        "hand_id": record.hand_id,
        "token_ids": token_ids,
        "num_tokens": len(token_ids),
        "actions": [asdict(action) for action in record.actions],
        "hole_cards": {
            record.player_names.get(uuid, uuid): cards for uuid, cards in record.hole_cards.items()
        },
        "community_cards": list(record.community_cards),
        "player_perspectives": player_perspectives,
        "result": record.result,
        "big_blind": record.big_blind,
        "initial_stacks": record.initial_stacks,
    }


def generate_dataset(
    num_hands: int,
    output_dir: str | Path,
    *,
    shard_size: int = 500,
    initial_stack: int = 1000,
    small_blind: int = 20,
    bot_a: str = "honest",
    bot_b: str = "random",
    checkpoint: Path | None = None,
    device: str = "cpu",
    policy: str = "greedy",
    vocab: Vocabulary | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = vocab or Vocabulary.from_config()
    recorder = HandRecorder()
    token_counter: Counter[int] = Counter()
    sequence_lengths: list[int] = []

    shard_index = 0
    shard_examples: list[dict[str, Any]] = []

    for hand_id in tqdm(range(num_hands), desc="Self-play hands"):
        record = play_hand(
            hand_id,
            recorder,
            initial_stack=initial_stack,
            small_blind=small_blind,
            bot_a=bot_a,
            bot_b=bot_b,
            checkpoint=checkpoint,
            device=device,
            policy=policy,
        )
        example = _hand_to_training_example(record, vocab)
        token_ids = example["token_ids"]
        example["token_ids_tensor"] = torch.tensor(token_ids, dtype=torch.long)
        shard_examples.append(example)
        sequence_lengths.append(len(token_ids))
        token_counter.update(token_ids)

        if len(shard_examples) >= shard_size:
            _write_shard(output_dir, shard_index, shard_examples, vocab.size)
            shard_index += 1
            shard_examples = []

    if shard_examples:
        _write_shard(output_dir, shard_index, shard_examples, vocab.size)
        shard_index += 1

    manifest = {
        "num_hands": num_hands,
        "num_shards": shard_index,
        "shard_size": shard_size,
        "vocab_size": vocab.size,
        "output_dir": str(output_dir),
        "bot_a": bot_a,
        "bot_b": bot_b,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "include_cards": True,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    return {
        "manifest": manifest,
        "token_counter": token_counter,
        "sequence_lengths": sequence_lengths,
        "vocab": vocab,
    }


def _write_shard(output_dir: Path, shard_index: int, examples: list[dict[str, Any]], vocab_size: int) -> None:
    path = output_dir / f"self_play_{shard_index:05d}.pt"
    payload = {
        "vocab_size": vocab_size,
        "hands": [
            {
                "hand_id": ex["hand_id"],
                "token_ids": ex["token_ids_tensor"],
                "num_tokens": ex["num_tokens"],
                "hole_cards": ex["hole_cards"],
                "player_perspectives": ex["player_perspectives"],
                "result": ex["result"],
                "actions": ex["actions"],
                "big_blind": ex["big_blind"],
                "initial_stacks": ex["initial_stacks"],
            }
            for ex in examples
        ],
    }
    torch.save(payload, path)


def print_summary(stats: dict[str, Any]) -> None:
    vocab: Vocabulary = stats["vocab"]
    token_counter: Counter[int] = stats["token_counter"]
    sequence_lengths: list[int] = stats["sequence_lengths"]
    manifest = stats["manifest"]

    total_tokens = sum(sequence_lengths)
    avg_length = total_tokens / max(len(sequence_lengths), 1)

    print("\n=== Self-play dataset summary ===")
    print(f"Total hands: {manifest['num_hands']:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average sequence length: {avg_length:.2f}")
    print(f"Shards written: {manifest['num_shards']} -> {manifest['output_dir']}")
    print(f"Vocabulary size: {manifest['vocab_size']}")

    missing_tokens = [
        vocab.token_for(token_id)
        for token_id in range(vocab.size)
        if token_counter[token_id] == 0
    ]
    print(f"\nTokens never observed: {len(missing_tokens)}")
    if missing_tokens:
        print("  (may indicate dead bucketing branches or rare events)")
        for token in missing_tokens[:30]:
            print(f"  - {token}")
        if len(missing_tokens) > 30:
            print(f"  ... and {len(missing_tokens) - 30} more")

    print("\nTop 15 token frequencies:")
    for token_id, count in token_counter.most_common(15):
        print(f"  {count:8,}  {vocab.token_for(token_id)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play tokenized training data.")
    parser.add_argument("--num-hands", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "self_play")
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--initial-stack", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=20)
    parser.add_argument(
        "--bot-a",
        choices=["honest", "random", "transformer"],
        default="honest",
    )
    parser.add_argument(
        "--bot-b",
        choices=["honest", "random", "transformer"],
        default="random",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Required when bot-a or bot-b is transformer",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy", choices=["greedy", "sampled"], default="greedy")
    args = parser.parse_args()

    if "transformer" in {args.bot_a, args.bot_b} and args.checkpoint is None:
        parser.error("--checkpoint is required when using bot=transformer")

    stats = generate_dataset(
        num_hands=args.num_hands,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        initial_stack=args.initial_stack,
        small_blind=args.small_blind,
        bot_a=args.bot_a,
        bot_b=args.bot_b,
        checkpoint=args.checkpoint,
        device=args.device,
        policy=args.policy,
    )
    print_summary(stats)


if __name__ == "__main__":
    main()
