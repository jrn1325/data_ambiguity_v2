# README

## Project Overview

This project fine-tunes Microsoft's CodeBERT model using adapters to classify JSON schema data. It utilizes PyTorch, Hugging Face's Transformers, and AdapterHub to train and evaluate the model. The project is configured to log training and evaluation metrics using Weights & Biases (wandb).

## Features

- Fine-tunes CodeBERT using adapters.
- Implements early stopping to prevent overfitting.
- Uses gradient accumulation for efficient training.
- Supports multi-GPU training.
- Logs training progress and evaluation metrics with wandb.
- Provides functions for training, testing, and evaluating the model.

## Requirements

Ensure the following dependencies are installed:

```sh
pip install torch transformers adapter-transformers scikit-learn pandas numpy wandb tqdm
```

## Directory Structure

- `converted_processed_schemas/` - Folder containing processed schema files.
- `processed_jsons/` - Folder containing processed JSON files.
- `adapter-model/` - Directory where trained adapters are saved.

## Model Training

To train the model, call:

```python
train_model(train_df, test_df)
```

where `train_df` and `test_df` are Pandas DataFrames containing schema data and labels.

## Model Evaluation

To evaluate a trained model:

```python
evaluate_model(test_df)
```

This function loads the trained adapter and computes evaluation metrics.

## Key Components

### Early Stopping

The `EarlyStopper` class monitors validation loss and stops training if no improvement is observed after a specified number of epochs.

### Dataset and Tokenization

The `CustomDataset` class handles tokenizing JSON schema data for CodeBERT, ensuring it fits within the model's maximum token length.

### Training Process

- Initializes CodeBERT with an adapter (`data_ambiguity`).
- Freezes the base model and trains only adapter layers.
- Uses the AdamW optimizer and linear learning rate scheduler.
- Implements gradient accumulation for stable training.
- Logs losses and metrics using wandb.

### Testing

- Evaluates the trained model on the test dataset.
- Computes accuracy, precision, recall, and F1-score.
- Logs metrics in wandb.

## Logging with Weights & Biases

wandb is used to track training progress and evaluation metrics. Initialize it with:

```python
wandb.init(project="custom-codebert_all_files_25")
```

Make sure to set up a wandb account and log in before running training.

## Saving and Loading Model

- The trained adapter is saved in `adapter-model/`.
- The `load_model_and_adapter()` function loads the saved model for evaluation.

## Hyperparameters

- **Batch Size:** 64
- **Max Token Length:** 512
- **Learning Rate:** 2e-5
- **Gradient Accumulation Steps:** 4
- **Epochs:** 25

## Running on Multiple GPUs

If multiple GPUs are available, the script automatically distributes training across them using `nn.DataParallel`.

## Notes

- Ensure `torch.cuda.is_available()` returns `True` to leverage GPU acceleration.
- The dataset should contain `schema` (JSON object) and `label` (integer) columns.

## Citation

If using CodeBERT, please cite:

```bibtex
@article{feng2020codebert,
  title={CodeBERT: A Pre-Trained Model for Programming and Natural Languages},
  author={Feng, Zhangyin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
}
```

