# SinLlama Continual Pretraining and Finetuning

This repository contains scripts and tools for continual pretraining the SinLlama model and finetuning the continual pretrained version for downstream tasks such as classification, generation, and evaluation.

The base SinLlama model can be found on Hugging Face: [polyglots/SinLlama_v01](https://huggingface.co/polyglots/SinLlama_v01).

## Overview

The codebase is organized into several key directories:

- **continual_pretraining/**: Scripts for continual pretraining the SinLlama model on additional datasets.
- **finetune/**: Scripts for finetuning the continual pretrained model on downstream tasks like news classification, sentiment analysis, and writing style detection.
- **data/**: Data processing and evaluation scripts, including clustering, translation, and dataset preparation.
- **data_preprocessing/**: Tools for analyzing, cleaning, and preparing datasets.
- **model_analysis/**: Scripts for analyzing model weights, activations, and performance metrics.
- **projects/**: Miscellaneous scripts for model merging, inference, pruning, and evaluation.

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers library
- Other dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Continual Pretraining

Navigate to the `continual_pretraining/` directory and run the appropriate script. For example:

```bash
python train_lora_cpt.py
```

Refer to `train_lora_cpt.md` for detailed instructions.

## Finetuning

For downstream tasks, use scripts in the `finetune/` directory. Examples include:

- `train_lora_classification.py` for classification tasks.
- `prepare_news_dataset.py`, `prepare_sentiment_dataset.py`, etc., for dataset preparation.

See `train_lora_classification.md` and `evaluate.md` for more details.

## Data Processing

Use scripts in `data/` and `data_preprocessing/` to prepare and evaluate datasets. For instance:

- `clustering.py` for clustering data.
- `merge_jsonl.py` for merging datasets.

## Model Analysis

Analyze the model using scripts in `model_analysis/`, such as `weights_distribution_visualization.py` or `activation_magnitude_visualization.py`.

## Evaluation

Run evaluation scripts like `eval_ppl.py` in the `projects/` directory to measure perplexity and other metrics.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)