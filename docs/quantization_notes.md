# int8 Quantization Notes (poker-transformer)

This document accompanies `src/serving/quantization_demo.py` (single-layer,
hand-rolled int8) and `src/serving/export_onnx.py` (full-model ONNX dynamic
quantization from Step 8). Numbers below were measured on CPU with the
project's smoke checkpoint and validation self-play shards unless noted.

---

## What int8 quantization is doing (one linear layer)

A `nn.Linear` layer stores a weight matrix **W** (shape `out × in`) and
computes `y = x @ W.T + b`.

**Symmetric int8 quantization** compresses **W** only (activations stay fp32 in
our demo and in ORT *dynamic* quantization):

1. Find the dynamic range of **W** (global max |w| for per-tensor, or per-row
   max for per-channel).
2. Pick a **scale** so the largest magnitude maps to 127:
   `scale = max_abs / 127`.
3. Encode: `W_int8 = round(W / scale)` clipped to `[-128, 127]`.
4. At inference, **dequantize on the fly**: `Ŵ = W_int8.float() * scale`,
   then run the usual fp32 matmul.

The error comes entirely from step 3: rounding each weight to the nearest
integer in int8 space. Different rows sharing one scale (per-tensor) amplifies
that rounding for small-magnitude channels.

### Measured accuracy — `blocks[0].mlp.fc` (1024 × 256)

Evaluated on **20,000 real validation tokens** (non-pad positions from
`blocks[0].mlp.fc` inputs):

| Method | Max abs error | Mean abs error |
|--------|---------------|----------------|
| Hand per-tensor int8 | 0.0148 | 0.00265 |
| Hand per-channel int8 | 0.0098 | 0.00175 |
| ORT `quantize_dynamic` (same layer) | 0.0200 | 0.00331 |

Per-channel beats per-tensor on this layer because each output neuron gets its
own scale (range **3.49e-4 – 7.28e-4** vs a single global **7.28e-4**).
ORT's slightly higher error here is expected: it uses **asymmetric uint8**
weights (`QuantType.QUInt8`, zero-point ≠ 0) and a different rounding path,
not identical math to our symmetric int8 tutorial code.

**Weight storage** for this layer:

| Format | Size |
|--------|------|
| fp32 | 1024 KiB |
| int8 per-tensor + 1 scale | ~256 KiB |
| int8 per-channel + 1024 scales | ~260 KiB |

---

## Why int8 helps more when memory-bandwidth-bound (roofline)

The **roofline model** plots achievable performance vs **arithmetic intensity**
(FLOPs per byte moved from DRAM):

```
performance (FLOP/s)
        |     /  compute roof (peak FLOP/s)
        |    /
        |   /  ← compute-bound region (steep part)
        |  /
        | /________________  memory roof (bandwidth × intensity)
        |/  ← memory-bound region (flat, slope = bandwidth)
        +------------------ arithmetic intensity
```

- **Compute-bound** ops: dense matmuls large enough that the GPU spends most
  time in tensor cores (high intensity). Example from Step 5.5 profiling: top
  CUDA time in `mm` / `bmm` / attention when batch and sequence are large.

- **Memory-bandwidth-bound** ops: performance is capped by how fast HBM can
  deliver bytes, not by peak FLOP/s. Reading a full fp32 weight matrix for
  every token position is mostly **loads**, not math.

**int8 helps in the memory-bound case** because the same matmul reads **4×
fewer weight bytes** (1 byte vs 4). If the kernel was waiting on DRAM, smaller
weights raise effective throughput even when the multiply still happens in fp32
after dequantization.

**int8 helps less (or not at all) in the compute-bound case** because the
tensor cores are already saturated; shaving weight bytes does not increase FLOP/s
once math is the bottleneck.

For poker-transformer at short average sequence length (~6.6 tokens/hand from
self-play stats), many layers operate at **low batch × seq** → low arithmetic
intensity → often **closer to the memory-bandwidth-limited region** of the
roofline than to peak matmul throughput. That is why weight-only quantization
is a reasonable first optimization even before activation quantization.

---

## Per-tensor vs per-channel tradeoffs

| | Per-tensor | Per-channel (per output row) |
|---|------------|------------------------------|
| **Scales stored** | 1 | `out_features` |
| **Implementation** | Simplest | Slightly more metadata |
| **Accuracy** | Worse when channels have very different ranges | Better; each filter uses full int8 range |
| **Memory savings** | Best (one scale) | Still ~4× on weights; +`4 × out_features` bytes |
| **This demo** | mean err 0.00265 | mean err 0.00175 |

Rule of thumb: per-channel is the default for conv/linear **weights** in most
production quantizers; per-tensor is mainly for embeddings or when metadata
size matters.

---

## Speed: hand-rolled vs ORT (same layer)

Single-layer CPU latency (20k × 256 matmul, 300 timed iterations):

| Implementation | Mean | p95 |
|----------------|------|-----|
| fp32 PyTorch | 43.0 ms | 48.5 ms |
| Hand int8 (dequant → fp32 matmul) | 42.7–43.0 ms | ~48.5 ms |
| ORT int8 dynamic | 34.4 ms | 38.1 ms |

The **hand-rolled path does not speed up** because we still call fp32
`F.linear` after dequantizing weights — we only reduced **storage**, not the
math precision or memory traffic of the GEMM itself in PyTorch.

**ORT is faster** because its runtime uses specialized int8/uint8 GEMM paths
that multiply with quantized weights without materializing full fp32 **W** in
DRAM.

Full-model ONNX export (Step 8, entire transformer):

| Model | File size | Mean latency |
|-------|-----------|--------------|
| fp32 ONNX | 19.30 MB | 5.14 ms |
| int8 dynamic ONNX | 5.14 MB (**73% smaller**) | 4.29 ms (**1.20×**) |

End-to-end speedup is modest on CPU for this small model because attention and
elementwise ops remain fp32; the win compounds on larger models and GPU EPs
where weight bandwidth dominates.

---

## What accuracy was lost (concrete summary)

1. **Single MLP projection (`mlp.fc`)** on validation activations:
   - Typical output drift: **~0.0018–0.0033 mean** absolute error per element
   - Worst-case per element: **~0.010–0.020** max absolute error
   - Relative to typical hidden activations (order ~0.1–1.0), this is **~0.2–3%**
     mean drift at this layer's output — error **compounds** through GELU and
     five more blocks, so end-to-end action logits should be checked separately.

2. **Full model (Step 8)** fp32 ONNX vs PyTorch: validation script asserts
   `action_logits` and `win_prob` match within **atol=1e-4, rtol=1e-3** on
   random inputs — quantization was **not** applied to that check (fp32 export
   only). After int8 dynamic quantization, expect larger logit drift; re-run
   `evaluate_loss.py` or compare action argmax on a val batch to quantify task
   impact.

3. **Takeaway**: int8 weight quantization here buys **~4× smaller weights** and
   **~1.2× CPU inference** on the full ONNX model, with **sub-percent to few-
   percent per-layer output noise** on the first MLP projection. Whether that
   changes win-rate vs FishPlayer requires an engine roll-out test with the
   quantized ONNX session — not covered by the layer-local demo.

---

## Reproduce

```bash
# Single-layer educational demo (prints error + latency table)
python -m poker_transformer.serving.quantization_demo --checkpoint checkpoints/best.pt

# Full-model fp32 + int8 ONNX export
python -m poker_transformer.serving.export_onnx --checkpoint checkpoints/best.pt
```

Read the inline comments in `quantization_demo.py` for step-by-step math
(`scale`, `program_id`-style indexing, dequant-on-the-fly vs true int8 GEMM).
