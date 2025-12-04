# README

## Project Overview

This project fine-tunes Microsoft's CodeBERT model using adapters to classify JSON data as static or dynamic. It utilizes PyTorch, Hugging Face's Transformers, and AdapterHub to train and evaluate the model. The project is configured to log training and evaluation metrics using Weights & Biases (wandb).

## File: `get_data.py`

**Purpose:**  
This script processes raw JSON datasets and their corresponding JSON Schemas to prepare them for downstream tasks, such as static vs. dynamic key classification. It performs several preprocessing steps to ensure schemas and JSON documents are fully dereferenced, valid, and structured consistently. The output is saved in designated directories for further analysis or model input.

Key functionalities include:

### 1. Schema Loading and Dereferencing
- `load_schema(schema_path)`:
  - Loads a JSON schema from a file.
  - Returns the schema as a Python dictionary or `None` if loading fails.

- `load_and_dereference_schema(schema_path)`:
  - Fully dereferences all `$ref` references in a JSON schema using `jsonref`.
  - Converts `JsonRef` objects to plain Python `dict`/`list` structures.
  - Ensures the schema is usable for static and dynamic key extraction.

- `deref_to_dict(obj)`:
  - Recursive helper to convert `JsonRef` objects to plain Python types.

- `save_schema(dereferenced_schema, dataset_name)`:
  - Saves the dereferenced schema to the `processed_schemas` folder.

### 2. JSON Dataset Processing
- `process_documents(dataset_name, dataset_path)`:
  - Reads JSON lines from a dataset file.
  - Validates that each line is an object or array.
  - Writes valid documents to `processed_jsons` folder.

- `process_single_dataset(dataset_name)`:
  - Processes both the schema and dataset for a single dataset.
  - Performs existence check, emptiness check, schema loading, dereferencing, saving, and document processing.
  - Returns success flags for each step.

- `process_datasets(max_workers=8)`:
  - Processes all datasets concurrently using a `ThreadPoolExecutor`.
  - Tracks metrics for existence, non-empty datasets, successful schema loading, dereferencing, and overall success.
  - Prints a summary of dataset processing statistics.

### 3. Directory Management
- `recreate_directory(directory_path)`:
  - Deletes and recreates a directory to ensure a clean output folder.

### 4. Main Execution
- Processes all schemas and JSON datasets in the predefined directories (`SCHEMA_FOLDER` and `JSON_FOLDER`).
- Saves processed schemas and datasets in `processed_schemas` and `processed_jsons`.
- Measures and prints total processing time.

**How to run:**  
```bash
python get_data.py


```

## File: `convert_schemas.js`

**Purpose:**  
This script standardizes JSON Schemas to the **draft-2020-12** version using the [AlterSchema](https://github.com/sourcemeta-research/alterschema) tool.  
It ensures that all schemas, regardless of their original draft (draft-03, draft-04, draft-06, draft-07, 2019-09), are converted to a common format for downstream processing.  
If a schema is already in draft-2020-12, it is simply copied to the output directory.

**Key features:**
- Reads schemas from a specified directory (`./processed_schemas` by default).  
- Detects the current draft version using the `$schema` field.  
- Converts schemas to draft-2020-12 using AlterSchema CLI, if needed.  
- Copies schemas already in draft-2020-12 to the output directory.   

**How to run:**
1. Install AlterSchema (if not installed):
```bash
npm install -g @sourcemeta/alterschema
```
2. Run the script:
```bash
node convert_schemas.js
```

**Directories used/created:**
- `schemaDir` (`./processed_schemas`): Input folder containing JSON schemas to convert.  
- `outputDir` (`./converted_processed_schemas`): Output folder where converted schemas are saved.

**Notes:**
- Only `.json` files in the input directory are processed.  
- Unrecognized or unsupported draft versions are skipped with a log message.


## File: `process_data.py`

**Purpose:**  
Preprocess JSON documents and schemas to generate labeled datasets for static vs. dynamic key classification. Extracts paths, computes features, labels paths, and balances classes for training/testing.

**Key Functions:**

- `split_data(train_ratio, random_value)` – Split schemas into train/test sets.  
- `load_schema(schema_path)` – Load a JSON schema.  
- `get_static_paths(schema)` – Extract static paths where `additionalProperties` is `False`.  
- `process_document(doc, path_types, path_freqs)` – Extract paths and track frequency/type information.  
- `create_dataframe(path_values, path_freqs, dataset)` – Convert extracted paths into a DataFrame with features.  
- `label_paths(df, static_paths)` – Label paths as static (`0`) or dynamic (`1`).  
- `resample_data(df, random_value)` – Balance classes using oversampling.  
- `process_dataset(dataset)` / `preprocess_data(schema_list)` – Process datasets and generate final DataFrames.  

**Output:**  
- `train_data.csv` – Labeled and resampled training set.  
- `test_data.csv` – Labeled testing set.  
- Optional: `train_jsons/` and `test_jsons/` containing valid documents.

**Usage:**  
```bash
python process_data.py <train_size> <random_value>


```

## File: `model.py`

## Purpose
Classifies JSON schema keys as **static** or **dynamic** using a CodeBERT model with numeric features.

## Training Modes
1. **Adapter**: Fine-tunes only an adapter added to CodeBERT.
2. **Full**: Fully fine-tunes CodeBERT.
3. **Jxplain**: Rule-based baseline using entropy thresholds.

## Running
- **Train**: `train_model(train_df, test_df, mode="adapter"|"full")`
- **Evaluate**: `evaluate_model(test_df, eval_mode="adapter"|"full")`
- **Jxplain**: `run_jxplain(test_df, eval_mode="jxplain")`

### CLI Usage
```bash
python model.py <train_data.csv> <test_data.csv> <mode> [adapter|full|jxplain]

```

## File: `transform_schema.py`

## Purpose
Transforms JSON schemas inferred from datasets (e.g., Baazizi-style) by **removing dynamic keys** and merging their definitions into `additionalProperties`. This reduces redundancy while preserving schema constraints. Can operate based on:

- Ground truth dynamic keys (`gt`)
- Predictions from trained models (`adapter` or `full`)
- Rule-based method (`jxplain`)

## Key Functions

### 1. Path & Schema Utilities
- `normalize_dynamic_paths`: Converts string paths to tuples and sorts them bottom-up.
- `normalize_additional_properties`: Ensures `additionalProperties` is a dict.
- `merge_dynamic_schema`: Merges dynamic key schemas into `additionalProperties`.
- `resolve_ref`: Follows internal `$ref` pointers in a schema.
- `process_path`: Recursively traverses a schema to remove dynamic keys, handling:
  - Objects (`properties`)
  - Arrays (`items`, `prefixItems`)
  - Schema combinators (`anyOf`, `oneOf`, `allOf`)

### 2. Schema Transformation
- `transform_schema_with_dynamic_keys(schema, dynamic_paths)`: Applies `process_path` for all dynamic paths, producing a transformed schema.

### 3. Schema Metrics
- `get_schema_size`: Returns size in bytes (without whitespace).
- `compare_schema_sizes`: Returns size difference between original and transformed schemas.

### 4. File I/O
- `load_schema` / `save_schema`: Read/write JSON schemas.
- `process_single_dataset`: Loads a schema, gets dynamic paths (from predictions or ground truth), transforms the schema, compares sizes, and saves it.

## Running the Script

- **CLI Usage**
```bash
python transform_schema.py <inferred_schemas_dir> <eval_input_dir> <mode>
```


## Dependencies

- **Python 3.10+**  
- **NumPy** (`numpy`)  
- **Pandas** (`pandas`)  
- **PyTorch** (`torch`)  
- **Huggingface Transformers** (`transformers`)  
- **tqdm** (`tqdm`)  
- **Scikit-learn** (`scikit-learn`)  
- **Accelerate** (`accelerate`)  
- **Weights & Biases** (`wandb`)  
- **Adapters library** (`adapters`)  
- **JSON Reference handling** (`jsonref`)  

**Standard library modules used:**  
- `argparse`, `ast`, `os`, `sys`, `time`, `math`, `shutil`, `copy` (`deepcopy`)  
- `collections` (`OrderedDict`)  
- `itertools` (`combinations`)  
- `torch.multiprocessing` as `mp`  
- `torch.nn` and `torch.nn.functional` (`nn`, `F`)  
- `torch.optim` (`AdamW`)  
- `torch.utils.data` (`DataLoader`, `Dataset`)  

### Install packages
```bash
uv install numpy pandas torch transformers tqdm scikit-learn accelerate wandb adapters jsonref


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

