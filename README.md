# ⚡ PulseRT: GPU-Accelerated ICU Time-Series Monitoring

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-GPU_Kernels-00A67E.svg)](https://github.com/openai/triton)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000.svg)](https://nextjs.org/)

**PulseRT** is a research-engineering prototype of a GPU-accelerated biomedical time-series inference system. It addresses a critical bottleneck in healthcare AI: **preprocessing latency**. By pushing irregular physiological data handling (missing values, temporal gaps) directly to the GPU via custom Triton kernels, PulseRT achieves sub-10ms end-to-end inference latency on consumer hardware.

---
