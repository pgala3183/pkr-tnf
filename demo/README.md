# Demo integration (jarczano Texas Hold'em Web App)

Heads-up browser poker app wired to the poker-transformer FastAPI service.

## Setup

From the **repo root**:

```bash
pip install -e .
pip install -r demo/requirements.txt

# Export quantized ONNX model if you have not already
python -m poker_transformer.serving.export_onnx --checkpoint checkpoints/best.pt
```

## Run

Terminal 1 — transformer API:

```bash
uvicorn poker_transformer.serving.api:app --host 0.0.0.0 --port 8000
```

Terminal 2 — demo Flask app:

```bash
cd demo
python main.py
```

Open the printed URL in a browser → **Menu** → select **Transformer** → **Play**.

## Architecture

| Component | Role |
|-----------|------|
| `demo/app/hand_history.py` | Tracks actions during a hand (demo format) |
| `src/serving/demo_adapter.py` | Maps demo state ↔ `/predict` JSON (decoupled) |
| `demo/app/transformer_bot.py` | HTTP client calling `http://localhost:8000/predict` |

Set `TRANSFORMER_API_URL` to point at a remote API if needed.

## Toggle

The menu button **Transformer** selects `opponent=transformer` (vs Bob/Carol/Dave heuristics).
