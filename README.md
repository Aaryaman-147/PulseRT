# ⚡ PulseRT: GPU-Accelerated ICU Time-Series Monitoring

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-GPU_Kernels-00A67E.svg)](https://github.com/openai/triton)

**PulseRT** is a research-engineering prototype of a GPU-accelerated biomedical time-series inference system. It addresses a critical bottleneck in healthcare AI: **preprocessing latency**. By pushing irregular physiological data handling (missing values, temporal gaps) directly to the GPU via custom Triton kernels, PulseRT achieves sub-10ms end-to-end inference latency on consumer hardware.

---


## 🏗 Architecture Diagram

```text
[ MIMIC-III ICU Dataset ]
         │ (Raw Telemetry: HR, SpO2, Resp)
         ▼
[ FastAPI Streaming Simulator ] ──(WebSocket)──▶ [ Next.js Cyberpunk Dashboard ]
         │ (Batched Tensors)                               ▲ (Live Charting & Alerts)
         ▼                                                 │
[ Triton GPU Kernel ]                                      │
   ├─ NaN Mask Generation                                  │
   ├─ LOCF Index Propagation (Matrix ffill)                │
   └─ Δt (Delta T) Extraction                              │
         │ (Clean Tensor, Mask, Δt)                        │
         ▼                                                 │
[ PyTorch GRU-D Engine ]                                   │
   ├─ Temporal Decay Factor (γ)                            │
   ├─ Empirical Mean Reversion                             │
   └─ Sequence Prediction                                  │
         │ (Reconstruction Error / MSE)                    │
         └─────────────────────────────────────────────────┘

```

---
## 🚀 Technical Highlights

- Zero-Copy Pipeline: The raw data stream is converted to a PyTorch tensor immediately. Triton preprocessing and GRU inference occur contiguously in VRAM, eliminating expensive Host-to-Device CPU transfers.
- Triton Preprocessing: Standard pandas/numpy interpolation strategies destroy low-latency budgets. PulseRT utilizes custom Triton C++ level kernels via Python to perform massively parallel data masking.
- Matrix LOCF: Replaced standard iterative forward-fill with highly optimized GPU index propagation using cummax and gather.
- Dynamic Temporal Decay: Implements the GRU-D architecture to mathematically decay missing sensor readings toward empirical baselines based on the time since the last valid observation ($\Delta t$).

---

## 📊 Performance Benchmarks

---

## 🕳️ Missing Data Handling & Triton Kernel Explanation

Medical streams frequently drop packets due to sensor disconnection or recalibration. To solve this without CPU overhead, our custom Triton Kernel loads raw telemetry into SRAM, instantly detects -999.0 (NaN) flags, and generates a binary observation mask. Next, we execute Last Observation Carried Forward (LOCF) via Index Propagation. Instead of looping through the tensor to carry values forward, we construct an index matrix, zero out the indices of missing values, and use PyTorch's cummax (cumulative maximum) to instantly drag the last valid index down the column. Finally, we calculate the time gap ($\Delta t$) by subtracting the LOCF index from the current step index.

---
