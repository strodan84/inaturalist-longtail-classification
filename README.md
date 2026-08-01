# 🌿 iNaturalist Long-Tailed Fine-Grained Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![timm](https://img.shields.io/badge/timm-0.9.x-green?style=for-the-badge)](https://github.com/huggingface/pytorch-image-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> Fine-grained species identification benchmark evaluating modern PyTorch backbones (`timm`) and dynamic gradient loss functions on extreme long-tailed biodiversity datasets.

---

## 📌 Overview

This repository provides a modular, production-minded PyTorch framework for fine-grained computer vision on long-tailed species distributions. It features dynamic gradient balancing via **Seesaw Loss**, automated experiment tracking, and model export utilities (ONNX/TorchScript) targeting real-world deployments similar to the iNaturalist vision pipeline.

### Key Features
* 🧬 **Long-Tail Balancing:** Custom `SeesawLoss` implementation mitigating rare-class gradient suppression.
* ⚡ **Modern Backbones:** Seamless swapping across `ConvNeXt`, `EVA-02`, and `Swin Transformer V2` via `timm`.
* 📊 **Evaluations:** Split accuracy tracking for head, medium, and tail (rare taxa) classes.
* 📦 **Export Ready:** Scripted ONNX runtime conversion pipeline with INT8 quantization.

---

## 📑 Table of Contents

- [Quickstart](#-quickstart)
- [Architecture & Loss](#-architecture--loss)
- [Benchmarking Results](#-benchmarking-results)
- [Repository Structure](#-repository-structure)
- [License](#-license)

---

## 🚀 Quickstart

### Prerequisites

```bash
git clone [https://github.com/your-username/inaturalist-longtail-vision.git](https://github.com/your-username/inaturalist-longtail-vision.git)
cd inaturalist-longtail-vision
pip install -r requirements.txt
