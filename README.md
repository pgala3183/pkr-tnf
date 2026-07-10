# poker-transformer

Transformer-based poker agent trained on hand histories, integrated with PyPokerEngine.

## Project Status

<!-- Fill in as you go -->

## Structure

```
src/
├── tokenizer/          # action <-> token vocabulary
├── model/              # transformer architecture
├── training/           # training loop, config, checkpoints
├── engine_integration/ # PyPokerEngine player wrapper
├── eval/               # benchmarking harness
├── serving/            # ONNX export + FastAPI app
└── utils/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
