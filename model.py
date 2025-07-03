import ast
import json
import numpy as np
import os
import pandas as pd
import random
import sys
import torch
import torch.nn as nn
import tqdm
import wandb
from adapters import AutoAdapterModel
from collections import defaultdict
from copy import deepcopy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  precision_recall_fscore_support, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings("ignore")

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
BATCH_SIZE = 64
MAX_TOKEN_LEN = 512
ADAPTER_NAME = "data_ambiguity"
MODEL_NAME = "microsoft/codebert-base"
PATH = "./adapter-model"
# Use os.path.expanduser to expand '~' to the full home directory path
SCHEMA_FOLDER = "converted_processed_schemas"
JSON_FOLDER = "processed_jsons"
TEMP_FOLDER = "temp_jsons"

RELEVANT_KEYS = {"type", "properties", "items", "required", "additionalProperties", "oneOf", "$ref", "$defs"}


# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_TOKEN_LEN):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        schema = self.data.iloc[idx]["schema"]
        label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long)

        tokenized_schema = self.tokenizer(
            json.dumps(schema),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length
        )

        return {
            "input_ids": tokenized_schema["input_ids"].squeeze(0),
            "attention_mask": tokenized_schema["attention_mask"].squeeze(0),
            "label": label
        }

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }

class CustomEvalDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_TOKEN_LEN):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]["path"]
        schema = self.data.iloc[idx]["schema"]
        label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long)

        tokenized_schema = self.tokenizer(
            json.dumps(schema),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length
        )

        return {
            "input_ids": tokenized_schema["input_ids"].squeeze(0),
            "attention_mask": tokenized_schema["attention_mask"].squeeze(0),
            "label": label,
            "path": path
        }

def collate_eval_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    paths = [item["path"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "path": paths
    }

def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(MODEL_NAME)
    model.add_adapter(ADAPTER_NAME, config="seq_bn")
    model.add_classification_head(ADAPTER_NAME, num_labels=2)

    # Activate the adapter
    model.set_active_adapters(ADAPTER_NAME)
    model.train_adapter(ADAPTER_NAME)

    # Enable wandb logging
    wandb.watch(model)

    return model, tokenizer

def train_model(train_df, test_df):
    accumulation_steps = 4
    learning_rate = 2e-5
    num_epochs = 25

    # Start wandb logging
    wandb.init(
        project="custom-codebert_all_files_25",
        config={
            "accumulation_steps": accumulation_steps,
            "batch_size": BATCH_SIZE,
            "dataset": "json-schemas",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_name": MODEL_NAME,
        }
    )

    # Initialize model and tokenizer
    model, tokenizer = initialize_model()

    # Freeze the base model (CodeBERT)
    for param in model.parameters():
        param.requires_grad = False

    # Only train adapter parameters
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True
            
    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load dataset
    train_dataset = CustomDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x))

    # Learning rate scheduler
    num_training_steps = (num_epochs * len(train_dataloader) // accumulation_steps)
    num_warmup_steps = int(0.1 * num_training_steps) 
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Early stopping
    early_stopper = EarlyStopper(patience=5, min_delta=0.005) 
    step = 0

     # Train the model
    model.train()
    pbar = tqdm.tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, position=1, leave=False, total=len(train_dataloader))):
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            training_loss = outputs.loss
            training_loss = training_loss.mean()

            # Normalize loss for gradient accumulation
            (training_loss / accumulation_steps).backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                step += 1

            total_loss += training_loss.item()
        
        average_loss = total_loss / len(train_dataloader)

        # Test the model
        testing_loss = test_model(test_df, tokenizer, model, device, wandb)

        wandb.log({
            "training_loss": average_loss,
            "testing_loss": testing_loss,
            "learning_rate": lr_scheduler.get_last_lr()[-1], 
            "step": step,
            "epoch": epoch + 1
        })

        # early stopping
        early_stop = early_stopper.early_stop(testing_loss)
        if early_stop:
            print("Early stopping")
            break
        
    # Save the adapter
    save_model_and_adapter(model)
    wandb.save(f"{PATH}/*")
    wandb.finish()

def test_model(test_df, tokenizer, model, device, wandb):
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x))

    model.eval()
    total_loss = 0.0

    total_actual_labels = []
    total_predicted_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Get the probabilities
            logits = outputs.logits

            # Calculate the testing loss
            testing_loss = outputs.loss.mean()
            total_loss += testing_loss.item()

            # Get the actual and predicted labels
            actual_labels = labels.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            total_actual_labels.extend(actual_labels)
            total_predicted_labels.extend(predicted_labels)

    average_loss = total_loss / len(test_loader)


   # Overall accuracy (computed on all labels)
    accuracy = accuracy_score(total_actual_labels, total_predicted_labels)

    # Metrics for the positive class (1)
    dynamic_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1)

    # Metrics for the negative class (0)
    static_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=0)

    # Log metrics to wandb
    metrics = {
        "accuracy": accuracy,
        "dynamic precision": dynamic_precision,
        "dynamic recall": dynamic_recall,
        "dynamic F1": dynamic_f1,
        "static precision": static_precision,
        "static recall": static_recall,
        "static F1": static_f1,
    }

    # Log metrics to wandb
    wandb.log(metrics)
    return average_loss
    
def filter_labels_positive(true_labels, predicted_labels):
    """
    Filter true and predicted labels for the positive class.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        tuple: Tuple containing filtered true labels and filtered predicted labels for the positive class.
    """

    # Get indices where true labels are 1
    positive_indices = [i for i, label in enumerate(true_labels) if label == 1]

    # Filter true labels for positive class
    true_labels_positive = [true_labels[i] for i in positive_indices]

    # Filter predicted labels for positive class
    predicted_labels_positive = [predicted_labels[i] for i in positive_indices]

    return true_labels_positive, predicted_labels_positive

def filter_labels_negative(true_labels, predicted_labels):
    """
    Filter true and predicted labels for the negative class.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        tuple: Tuple containing filtered true labels and filtered predicted labels for the negative class.
    """

    # Get indices where true labels are 0 (negative class)
    negative_indices = [i for i, label in enumerate(true_labels) if label == 0]

    # Filter true labels for negative class
    true_labels_negative = [true_labels[i] for i in negative_indices]

    # Filter predicted labels for negative class
    predicted_labels_negative = [predicted_labels[i] for i in negative_indices]

    return true_labels_negative, predicted_labels_negative

def save_model_and_adapter(model):
    """
    Save the model's adapter and log it as a WandB artifact.

    Args:
        model: The model with the adapter to save.
    """

    path = os.path.join(os.getcwd(), "adapter-model")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module    
    # Save the entire model
    model.save_pretrained(path)

    # Save the adapter
    model.save_adapter(path, ADAPTER_NAME)
    
def load_model_and_adapter():
    """
    Load the model and adapter from the specified path.

    Returns:
        PreTrainedModel: The model with the loaded adapter.
    """
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoAdapterModel.from_pretrained(PATH)

    # Load the adapter from the saved path and activate it
    adapter_name = model.load_adapter(PATH)
    model.set_active_adapters(adapter_name)
    print(f"Loaded and activated adapter: {adapter_name}", flush=True)
    
    return model, tokenizer

def run_jxplain(test_df):
    """
    Perform the Jxplain method to classify dynamic keys based on datatype entropy 
    and key entropy. The method calculates and prints performance metrics 
    (accuracy, precision, recall, and F1 score) for both classes (dynamic and static keys) 
    as well as combined metrics.

    Args:
        test_df (pd.DataFrame): DataFrame containing test data.

    Returns:
        None: Prints accuracy, precision, recall, and F1 score for both classes.
    """
    
    # Perform Jxplain: Predict if a key is dynamic (1) based on entropy conditions
    y_pred = ((test_df["datatype_entropy"] == 1) & (test_df["key_entropy"] > 1)).astype(int)
    y_test = test_df["label"]

    # Calculate per-class metrics (returns arrays: [class_0, class_1])
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

    precision_0, precision_1 = precision
    recall_0, recall_1 = recall
    f1_0, f1_1 = f1

    # Calculate overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)

    # Print per-class metrics
    print(f"Class 0 (Static) - Precision: {precision_0:.4f}, Recall: {recall_0:.4f}, F1 Score: {f1_0:.4f}")
    print(f"Class 1 (Dynamic) - Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, F1 Score: {f1_1:.4f}")

    # Print overall metrics
    print(f"Overall (Weighted) - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1 Score: {f1_macro:.4f}, Accuracy: {accuracy:.4f}")


def evaluate_model(test_df):
    """
    Evaluate the model on the test data and collect correctly predicted dynamic paths.

    Args:
        test_df (pd.DataFrame): DataFrame with test examples.

    Returns:
        dynamic_paths (list): Paths where the model correctly predicted dynamic (label=1).
    """
    model, tokenizer = load_model_and_adapter()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataset = CustomEvalDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_eval_fn(x))

    dynamic_paths = []
    index = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            for i in range(len(preds)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                if pred_label == true_label == 1:
                    dynamic_paths.append(ast.literal_eval(test_df.iloc[index]["path"]))
                index += 1

    return dynamic_paths

def strip_schema(schema):
    """
    Strip schema to keep only type and nested properties.
    
    Args:
        schema (dict or set): The JSON Schema to be stripped.   
        
    Returns:
        dict: A simplified version of the schema containing only the type and properties.
    """
    if isinstance(schema, set):
        return list(schema)
    if not isinstance(schema, dict):
        return schema

    new_schema = {}
    for key, value in schema.items():
        if key == "type":
            new_schema["type"] = list(value) if isinstance(value, set) else value
        elif key == "properties":
            new_schema["properties"] = {
                k: strip_schema(v)
                for k, v in value.items()
            }
    return new_schema


def nest_schema(group_df, dynamic_paths=None, abstract_dynamic=True):
    """
    Builds a nested JSON Schema from path-level schema fragments.

    Args:
        group_df (pd.DataFrame): DataFrame with 'path' and 'schema' columns.
        dynamic_paths (list of tuples): Path *prefixes* to treat as dynamic (excluding final key).
        abstract_dynamic (bool): Whether to abstract dynamic paths using `additionalProperties`.

    Returns:
        dict: A nested JSON Schema.
    """
    if dynamic_paths is None:
        dynamic_paths = []

    root = {"type": "object", "properties": {}}
    dynamic_collected = defaultdict(list)

    for _, row in group_df.iterrows():
        path = ast.literal_eval(row["path"])
        schema = strip_schema(ast.literal_eval(row["schema"]))

        prefix = path[:-1]
        key = path[-1]
        matched_dynamic = None

        for dp in dynamic_paths:
            if prefix == dp:
                matched_dynamic = dp
                break  # Just find match for now; collect type later if needed

        node = root
        for i, part in enumerate(path):
            is_last = (i == len(path) - 1)
            subpath = path[:i + 1]

            if abstract_dynamic and matched_dynamic and subpath == matched_dynamic + (key,):
                node.setdefault("properties", {})
                node["properties"][key] = {
                    "additionalProperties": {}
                }

                # Collect the type here, since we’re not going to the leaf
                dynamic_collected[matched_dynamic].append(schema["properties"].get(key, {}).get("type", "object"))
                break

            node = node.setdefault("properties", {}).setdefault(part, {"type": "object"})

            if is_last:
                node.update(schema)

    # Post-process to assign correct type unions to each dynamic key abstraction
    for dp, types in dynamic_collected.items():
        node = root
        for part in dp:
            node = node.get("properties", {}).get(part, {})

        for key, prop in node.get("properties", {}).items():
            if "additionalProperties" in prop:
                all_types = set()
                for t in types:
                    if isinstance(t, list):
                        all_types.update(t)
                    else:
                        all_types.add(t)
                prop["additionalProperties"]["type"] = sorted(all_types)

    return root



def compare_json_schemas(original_schema, enhanced_schema):
    """
    Compare the size of two JSON Schemas in kilobytes (KB).

    Args:
        original_schema (dict): The original JSON Schema.
        enhanced_schema (dict): The enhanced JSON Schema.

    Returns:
        dict: A dictionary containing the size in KB of both schemas.
    """
    original_schema_str = json.dumps(original_schema, indent=2)
    enhanced_schema_str = json.dumps(enhanced_schema, indent=2)
    #original_schema_str = json.dumps(original_schema, separators=(',', ':'))
    #enhanced_schema_str = json.dumps(enhanced_schema, separators=(',', ':'))

    comparison = {
        "kilobytes": {
            "original_schema": round(len(original_schema_str.encode("utf-8")) / 1024, 2),
            "enhanced_schema": round(len(enhanced_schema_str.encode("utf-8")) / 1024, 2),
        }
    }

    return comparison

def eval_dataset(test_df):
    results = {}
    total_original = 0
    total_enhanced = 0
    total_reduction = 0
    count = 0

    for filename, group_df in tqdm.tqdm(
        test_df.groupby("filename"),
        desc="Evaluating datasets",
        total=len(test_df["filename"].unique())
    ):

        print(f"Evaluating model on: {filename} with {len(group_df)} unique paths", flush=True)
        dynamic_paths = evaluate_model(group_df)

        original_schema = nest_schema(group_df, dynamic_paths, abstract_dynamic=False)
        abstracted_schema = nest_schema(group_df, dynamic_paths, abstract_dynamic=True)
        
        # Save the original and abstracted schemas to files
        original_schema_path = os.path.join(TEMP_FOLDER, f"{filename}_original.json")
        abstracted_schema_path = os.path.join(TEMP_FOLDER, f"{filename}_abstracted.json")
        with open(original_schema_path, "w") as f:
            json.dump(original_schema, f, indent=2)
        with open(abstracted_schema_path, "w") as f:
            json.dump(abstracted_schema, f, indent=2)

        stats = compare_json_schemas(original_schema, abstracted_schema)
        results[filename] = stats

        original_kb = stats["kilobytes"]["original_schema"]
        enhanced_kb = stats["kilobytes"]["enhanced_schema"]

        total_original += original_kb
        total_enhanced += enhanced_kb

        # Calculate reduction if original size > 0 to avoid division by zero
        if original_kb > 0:
            reduction = (original_kb - enhanced_kb) / original_kb
        else:
            reduction = 0
        total_reduction += reduction

        count += 1

    average_stats = {
        "average_kilobytes": {
            "original_schema": round(total_original / count, 2) if count > 0 else 0,
            "enhanced_schema": round(total_enhanced / count, 2) if count > 0 else 0,
        },
        "average_reduction": round(total_reduction / count, 4) if count > 0 else 0
    }

    results["summary"] = average_stats

    print(json.dumps(results, indent=2))




def main():
    try:
        # Parse command-line arguments
        train_data, test_data, mode = sys.argv[-3:]

        # Ensure mode is valid
        if mode not in {"train", "eval", "jxplain"}:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")
        
        if mode == "train":
            train_df = pd.read_csv(train_data, sep=";")
            test_df = pd.read_csv(test_data, sep=";")
            train_model(train_df, test_df)
        elif mode == "eval":
            test_df = pd.read_csv(test_data, sep=";")
            eval_dataset(test_df)
        elif mode == "jxplain":
            test_df = pd.read_csv(test_data, sep=";")
            run_jxplain(test_df)

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_data> <test_data> <mode>")
        sys.exit(1)


if __name__ == "__main__":
    main()
