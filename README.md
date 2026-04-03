# CLICKER: Cross-Lingual Knowledge Editing via In-Context Learning with Adaptive Stepwise Reasoning

This repository accompanies the EACL 2026 Findings paper *CLICKER: Cross-Lingual Knowledge Editing via In-Context Learning with Adaptive Stepwise Reasoning*. The release provides the main research framework components used for CLICKER, including the core knowledge-editing modules, retrieval and cache construction code, evaluation code, and bundled benchmark data files.

## Features / Released Components

- Core CLICKER implementation for retrieval-augmented, in-context cross-lingual knowledge editing.
- Modular code for dataset loading, context construction, demo construction, and evaluation.
- Embedding cache generation and retrieval precomputation utilities.
- Evaluation code for reliability, generalization, and locality-style metrics.
- Bundled multilingual benchmark files for `Multi-CounterFact` and `MzsRE` under `datasets/`.
- For the full `Multi-CounterFact` dataset, refer to [Multi-CounterFact](https://huggingface.co/datasets/KazeJiang/Multi-CounterFact).

## What Is Included in This Repository

- Main framework code:
  - `main.py`
  - `knowledge_editor.py`
  - `context_builder.py`
  - `dataset_loader.py`
  - `precompute_retrieval.py`
  - `evaluate.py`
  - `f1_em_metrics.py`
- Dataset files included in this release:
  - `datasets/Multi-CounterFact/` for English, German, French, Japanese, and Chinese
  - `datasets/MzsRE/` for English, German, French, and Chinese
- A pinned dependency file: `requirements.txt`

## Installation

The repository ships with a `requirements.txt` file from the research environment used for development.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Please note:

- `requirements.txt` is the only environment specification included in this release.
- It contains a research-oriented dependency set with pinned versions and GPU-specific packages such as `faiss-gpu`; users may need to adapt the environment to their own platform.
- External model access is not bundled. The current code supports either a local Hugging Face model path or an OpenAI API-based path, but both require user-side configuration.

## Basic Usage

This release exposes the main framework code, but it is not packaged as a fully configurable command-line tool. The primary entry point is `main.py`.

Before running it, inspect and adapt the hard-coded paths and model settings in the source code to match your environment:

- In `main.py`, dataset paths are currently written as absolute `/datasets/...` paths and model paths as absolute `/models/...` paths.
- In `knowledge_editor.py`, the OpenAI client path uses a placeholder API key string and requires user-side configuration if that branch is used.
- Optional `use_remake=True` execution expects an auxiliary classifier checkpoint that is not included in this release.

After local configuration, the main script can be executed from the repository root:

```bash
python main.py
```

As released, `main.py` is configured in code for a specific experimental setting rather than through CLI arguments. The script:

1. loads the selected dataset pair,
2. creates embedding and retrieval caches if they do not already exist,
3. initializes the CLICKER editor, and
4. runs evaluation with the selected settings.

The reusable framework components are available directly in the Python modules if you want to adapt the code for your own research workflows. However, this repository does not currently provide a cleaned public pipeline for full experiment reproduction across all paper settings.

## Citation

If you use this repository or the accompanying paper, please cite:

```bibtex
@inproceedings{jiang-etal-2026-clicker,
  title = {CLICKER: Cross-Lingual Knowledge Editing via In-Context Learning with Adaptive Stepwise Reasoning},
  author = {Jiang, Zehui and Zhao, Xin and Kumadaki, Yuta and Yoshinaga, Naoki},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2026},
  pages = {5007--5022},
  year = {2026}
}
```
