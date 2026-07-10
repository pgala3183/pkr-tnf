"""Evaluate train/val loss and perplexity from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from poker_transformer.model.transformer import ModelConfig, PokerTransformer
from poker_transformer.tokenizer.vocab import Vocabulary
from poker_transformer.training.data import load_hands, make_dataloader, split_hands
from poker_transformer.training.metrics import evaluate
from poker_transformer.training.train import TrainingConfig, load_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[PokerTransformer, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**payload["model_config"])
    model = PokerTransformer(model_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint train/val losses.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "best.pt",
        help="Path to checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "training.yaml",
        help="Training config for data paths and hyperparameters",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.from_config()

    model, payload = load_checkpoint(args.checkpoint, device)

    hands = load_hands(PROJECT_ROOT / config.data_dir)
    train_hands, val_hands = split_hands(hands, val_ratio=config.val_ratio)

    train_loader = make_dataloader(
        train_hands,
        vocab=vocab,
        batch_size=config.batch_size,
        block_size=config.block_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    val_loader = make_dataloader(
        val_hands,
        vocab=vocab,
        batch_size=config.batch_size,
        block_size=config.block_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    train_metrics = evaluate(
        model,
        train_loader,
        pad_id=vocab.pad_id,
        value_loss_weight=config.value_loss_weight,
        device=device,
    )
    val_metrics = evaluate(
        model,
        val_loader,
        pad_id=vocab.pad_id,
        value_loss_weight=config.value_loss_weight,
        device=device,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "step": payload.get("step"),
        "train_hands": len(train_hands),
        "val_hands": len(val_hands),
        "value_loss_weight": config.value_loss_weight,
        "train": train_metrics,
        "val": val_metrics,
    }

    print("\n=== Checkpoint evaluation ===")
    print(json.dumps(report, indent=2))
    print(
        f"\nVal action perplexity: {val_metrics['action_perplexity']:.4f} "
        f"(action loss {val_metrics['action_loss']:.4f}, total loss {val_metrics['total_loss']:.4f})"
    )


if __name__ == "__main__":
    main()
