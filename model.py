import ast
import json
import numpy as np
import os
import pandas as pd
import shutil
import sys
import torch
import torch.nn as nn
import tqdm
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from adapters import AutoAdapterModel, SeqBnConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")

# -------------------- Constants --------------------
MODEL_NAME = "microsoft/codebert-base"
ADAPTER_PATH = "./adapter-model/adapter"
FULL_PATH = "./adapter-model/full"
ADAPTER_NAME = "data_ambiguity"
BATCH_SIZE = 64
MAX_TOK_LEN = 512
ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-5
NUM_EPOCHS = 25
SEED = 101
SCHEMA_KEYWORDS = ["definitions", "$defs", "properties", "additionalProperties", "patternProperties", "oneOf", "allOf", "anyOf", "items", "type", "not"]
DISTINCT_SUBKEYS_UPPER_BOUND = 1000


# -------------------- Early Stopper --------------------
# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.001):
        """
        patience: number of validations without improvement before stopping
        min_delta: minimum change in val loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def early_stop(self, val_loss):
        """
        Returns True if training should stop early.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        # No improvement → increase counter
        self.counter += 1

        # Patience exceeded
        return self.counter >= self.patience


# -------------------- Dataset --------------------
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_TOK_LEN):
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.schemas = dataframe["schema"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        schema = ast.literal_eval(self.schemas[idx])

        # Tokenize schema text (optional)
        encoding = self.tokenizer(
            json.dumps(schema["properties"]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Extract numeric features from schema
        numeric_feats = torch.tensor([
            schema.get("datatype_entropy", 0.0),
            schema.get("key_entropy", 0.0),
            schema.get("parent_frequency", 0.0),
            schema.get("num_nested_keys", 0.0),
            float(schema.get("semantic_similarity", 0.0)),
            float(schema.get("additionalProperties", False))
        ], dtype=torch.float)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "numeric_feats": numeric_feats,
            "labels": self.labels[idx]
        }

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    numeric_feats = torch.stack([b["numeric_feats"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_feats": numeric_feats,
        "labels": labels
    }


# -------------------- Model Initialization & Training & Testing --------------------
class CustomCodeBERT(nn.Module):
    def __init__(self, 
                 model_name=MODEL_NAME, 
                 num_numeric_features=6, 
                 num_labels=2, 
                 dropout=0.3, 
                 training_mode="adapter"):
        super().__init__()
        self.training_mode = training_mode
        self.num_numeric_features = num_numeric_features

        # Load base model
        if training_mode == "adapter":
            self.base_model = AutoAdapterModel.from_pretrained(model_name)
            self.base_model.config.output_hidden_states = True

            config = SeqBnConfig(
                mh_adapter=False,
                output_adapter=True,
                reduction_factor=16,
                non_linearity="relu",
                dropout=dropout
            )
            self.base_model.add_adapter(ADAPTER_NAME, config=config)
            self.base_model.add_classification_head(ADAPTER_NAME, num_labels=num_labels)
            self.base_model.set_active_adapters(ADAPTER_NAME)
            self.base_model.train_adapter(ADAPTER_NAME)

        else:
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                output_hidden_states=True 
            )
            self.base_model.config.output_hidden_states = True

            # Replace classifier
            if hasattr(self.base_model, "classifier") and isinstance(self.base_model.classifier, nn.Linear):
                in_features = self.base_model.classifier.in_features
                out_features = self.base_model.classifier.out_features
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, out_features)
                )

        # Hidden dimension
        hidden_size = self.base_model.config.hidden_size

        # Numeric projection
        self.numeric_proj = nn.Linear(num_numeric_features, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Final classifier after concatenation
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask, numeric_feats, labels=None):
        # Encode inputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # CLS embedding from final layer
        cls_emb = outputs.hidden_states[-1][:, 0, :]

        # Project numeric features
        numeric_emb = self.numeric_proj(numeric_feats)

        # Concatenate embeddings
        combined = torch.cat([cls_emb, numeric_emb], dim=-1)
        combined = self.dropout(combined)

        # Prediction
        logits = self.classifier(combined)

        # Loss
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"logits": logits, "loss": loss, "cls_emb": cls_emb}

def initialize_model(training_mode="adapter"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CustomCodeBERT(training_mode=training_mode)
    print(f"Initialized {MODEL_NAME} in {training_mode} mode")
    return model, tokenizer

def train_model(train_df, test_df, training_mode="adapter"):
    """
    Train the model on the training data and evaluate on the test data.

    Args:
        train_df (pd.DataFrame)
        test_df (pd.DataFrame)
        training_mode (str): "adapter" or "full"
    """

    # Initialize W&B
    wandb.init(
        project="custom-codebert_all_files_25",
        config={
            "accumulation_steps": ACCUMULATION_STEPS,
            "batch_size": BATCH_SIZE,
            "dataset": "json-schemas",
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "model_name": MODEL_NAME,
            "training_mode": training_mode,
            "adapter_name": ADAPTER_NAME,
        }
    )

    # --- Accelerator ---
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    set_seed(SEED)

    # --- Model + Tokenizer ---
    model, tokenizer = initialize_model(training_mode)

    # --- Dataset / DataLoader
    train_dataset = CustomDataset(train_df, tokenizer)
    test_dataset = CustomDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config.learning_rate)
    num_training_steps = wandb.config.epochs * len(train_loader) // wandb.config.accumulation_steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)

    # --- Prepare with accelerator ---
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, scheduler)

    # --- Early stopper ---
    early_stopper = EarlyStopper(patience=2, min_delta=0.001)

    # --- Training loop ---
    for epoch in range(wandb.config.epochs):
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            outputs = model(**batch)
            loss = outputs["loss"]
            loss = loss / wandb.config.accumulation_steps
            accelerator.backward(loss)
            total_loss += loss.item() * wandb.config.accumulation_steps

            if (i + 1) % wandb.config.accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item() * wandb.config.accumulation_steps

        avg_train_loss = total_loss / len(train_loader)

        # --- Evaluation ---
        model.eval()
        all_labels, all_preds = [], []
        total_eval_loss = 0.0
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc="Testing"):
                outputs = model(**batch)
                logits = outputs["logits"]
                loss = outputs["loss"]

                total_eval_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(accelerator.gather(preds).cpu().numpy())
                all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())

        avg_eval_loss = total_eval_loss / len(test_loader)

        # --- Compute per-class metrics ---
        accuracy = accuracy_score(all_labels, all_preds)
        precision_per_class = precision_score(all_labels, all_preds, labels=[0,1], average=None)
        recall_per_class = recall_score(all_labels, all_preds, labels=[0,1], average=None)
        f1_per_class = f1_score(all_labels, all_preds, labels=[0,1], average=None)

        # --- W&B logging ---
        wandb.log({
            "epoch": epoch+1,
            "training_loss": avg_train_loss,
            "testing_loss": avg_eval_loss,
            "accuracy": accuracy,
            "static precision": precision_per_class[0],
            "dynamic precision": precision_per_class[1],
            "static recall": recall_per_class[0],
            "dynamic recall": recall_per_class[1],
            "static F1": f1_per_class[0],
            "dynamic F1": f1_per_class[1],
            "learning_rate": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")

        # --- Early stopping ---
        if early_stopper.early_stop(avg_eval_loss):
            print("Early stopping triggered!")
            break

    # --- Save adapter only ---
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    save_model_and_adapter(unwrapped_model, training_mode)
    wandb.finish()

def evaluate_model(test_df, eval_mode="adapter", output_dir="evaluation_results"):
    """
    Evaluate the model on the test data, save per-file correct/incorrect predictions to CSVs,
    and return metrics.
    """
    output_dir = output_dir + '_' + eval_mode
    
    # delete existing output directory if exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model and tokenizer ---
    model, tokenizer = load_model_and_adapter(eval_mode)

    # --- Accelerator ---
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    set_seed(SEED)

    # --- Dataset / DataLoader ---
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # --- Prepare model & loader ---
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    total_loss = 0.0
    all_labels, all_preds = [], []

    # --- Evaluation loop ---
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
            outputs = model(**batch)
            logits = outputs["logits"]
            loss = outputs["loss"]
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(accelerator.gather(preds).cpu().numpy())
            all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy())

    # --- Convert predictions to numpy arrays ---
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- Compute metrics ---
    average_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision_per_class = precision_score(all_labels, all_preds, labels=[0, 1], average=None)
    recall_per_class = recall_score(all_labels, all_preds, labels=[0, 1], average=None)
    f1_per_class = f1_score(all_labels, all_preds, labels=[0, 1], average=None)

    print(f"\n--- Evaluation Results ---")
    print(f"Test Loss: {average_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Static (0) -> Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}")
    print(f"Dynamic (1) -> Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}")

    # --- Match predictions back to DataFrame rows ---
    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = all_preds
    test_df["correct"] = test_df["pred"] == test_df["label"]

    # Keep only essential columns
    result_df = test_df[["filename", "path", "label", "pred", "correct"]]

    # --- Save per-file CSVs ---
    for filename, group in result_df.groupby("filename"):
        file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_results.csv")
        group.to_csv(file_path, index=False, sep=';')
        print(f"Saved: {file_path} ({len(group)} rows)")

    print(f"\nTotal files processed: {result_df['filename'].nunique()}")

    # --- Return metrics and filtered dataframes for further analysis ---
    correct_df = result_df[result_df["correct"]]
    incorrect_df = result_df[~result_df["correct"]]

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "correct_df": correct_df,
        "incorrect_df": incorrect_df,
    }

def save_model_and_adapter(model, training_mode="adapter"):
    if isinstance(model, nn.DataParallel):
        model = model.module

    if training_mode == "adapter":
        os.makedirs(ADAPTER_PATH, exist_ok=True)

        # 1. Save adapter
        model.base_model.save_adapter(ADAPTER_PATH, ADAPTER_NAME)

        # 2. Save pretrained backbone
        model.base_model.save_pretrained(ADAPTER_PATH)

        # 3. Save custom layers (numeric_proj + classifier + wrapper architecture)
        torch.save(model.state_dict(), os.path.join(ADAPTER_PATH, "custom_model_weights.pt"))

        print(f"Saved adapter + custom model weights to {ADAPTER_PATH}")

    else:
        os.makedirs(FULL_PATH, exist_ok=True)
        model.base_model.save_pretrained(FULL_PATH)
        torch.save(model.state_dict(), os.path.join(FULL_PATH, "custom_model_weights.pt"))
        print(f"Saved full fine-tuned model to {FULL_PATH}")

def load_model_and_adapter(training_mode="adapter"):
    # 1. Recreate architecture
    model = CustomCodeBERT(training_mode=training_mode)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if training_mode == "adapter":
        # 2. Load base model (CodeBERT backbone)
        model.base_model = AutoAdapterModel.from_pretrained(MODEL_NAME)
        model.base_model.config.output_hidden_states = True

        # 3. Load the saved adapter into the base model
        adapter_name = model.base_model.load_adapter(ADAPTER_PATH)
        model.base_model.set_active_adapters(adapter_name)

        # 4. Load your custom layers (numeric_proj + classifier)
        state = torch.load(os.path.join(ADAPTER_PATH, "custom_model_weights.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)

        print("Loaded CustomCodeBERT with adapters from", ADAPTER_PATH)

    else:
        # Full fine-tuned model
        model.base_model = AutoModelForSequenceClassification.from_pretrained(FULL_PATH)
        state = torch.load(os.path.join(FULL_PATH, "custom_model_weights.pt"), map_location="cpu")
        model.load_state_dict(state)
        print("Loaded full CustomCodeBERT model.")

    model.eval()
    return model, tokenizer



def resolve_ref(ref, root_schema):
    """
    Resolves an internal $ref like '#/definitions/User'.
    Returns a pointer directly to the referenced dict.
    """
    assert ref.startswith("#/"), "Only internal refs supported."
    path = ref[2:].split("/")  # e.g., ['definitions', 'User']
    target = root_schema
    for p in path:
        if p in target:
            target = target[p]
        else:
            return None
    return target

def is_static_path(schema, keys, root_schema):
    """
    Determines if the given path (list of keys) is static (0) or dynamic (1)
    according to the provided JSON schema.

    Args:
        schema (dict): Current JSON schema node.
        keys (list): List of keys representing the path.
        root_schema (dict): The root JSON schema for resolving $refs.
    Returns:
        int: 0 if static, 1 if dynamic.
    """

    # Strip root marker
    if keys and keys[0] == "$":
        return is_static_path(schema, keys[1:], root_schema)

    # Resolve $ref
    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], root_schema)
        return 1 if resolved is None else is_static_path(resolved, keys, root_schema)

    # No more keys to process
    if not keys:
        return 0 if schema.get("additionalProperties") is False else 1

    key = keys[0]
    remaining = keys[1:]

    # Handle combinators
    for combiner in ("anyOf", "oneOf", "allOf"):
        if combiner in schema:
            for subschema in schema[combiner]:
                if is_static_path(subschema, keys, root_schema) == 1:
                    return 1
            return 0

    # Handle object
    if schema.get("type") == "object":
        props = schema.get("properties", {})

        # Object contains the final key
        if len(keys) == 1:
            return 0 if schema.get("additionalProperties") is False else 1

        # Descend through declared properties
        if key in props:
            return is_static_path(props[key], remaining, root_schema)

        # Key not declared
        return 0 if schema.get("additionalProperties") is False else 1

    # Handle arrays
    if schema.get("type") == "array":
        if "items" in schema:
            return is_static_path(schema["items"], keys, root_schema)
        if "prefixItems" in schema:
            for subschema in schema["prefixItems"]:
                if is_static_path(subschema, keys, root_schema) == 1:
                    return 1
            return 0

    return 1


def run_recg(test_df, schemas_dir, eval_mode="recg", output_dir="evaluation_results"):
    """
    Predict static/dynamic using the correct JSON schema per file.
    """

    # Load all schemas into a dictionary: filename -> schema
    schemas = {}
    for schema_file in os.listdir(schemas_dir):
        full_path = os.path.join("ReCG_schemas", schema_file)
        with open(full_path, 'r') as f:
            schema_data = json.load(f)
        filename = os.path.splitext(schema_file)[0]
        schemas[filename] = schema_data

    # Compute predictions per row using the correct schema
    y_pred = []
    for row in test_df.itertuples(index=False):
        schema = schemas.get(os.path.splitext(row.filename)[0])
        y_pred.append(is_static_path(schema, ast.literal_eval(row.path), schema))
        print(f"Processed {row.filename} with path {row.path} -> Prediction: {y_pred[-1]}")
   
    y_test = test_df["label"]

    # --- Metrics ---
    overall_accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0,1])
    combined_precision, combined_recall, combined_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    positive_accuracy = accuracy_score(y_test[y_test==1], [p for p,t in zip(y_pred, y_test) if t==1])
    negative_accuracy = accuracy_score(y_test[y_test==0], [p for p,t in zip(y_pred, y_test) if t==0])

    print(f"Class 0 (Static) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1_score[0]:.4f}, Accuracy: {negative_accuracy:.4f}")
    print(f"Class 1 (Dynamic) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1_score[1]:.4f}, Accuracy: {positive_accuracy:.4f}")
    print(f"Both Classes (Overall) - Precision: {combined_precision:.4f}, Recall: {combined_recall:.4f}, F1 Score: {combined_f1:.4f}, Accuracy: {overall_accuracy:.4f}")

    # --- Save per-file results ---
    result_df = test_df.copy()
    result_df["pred"] = y_pred
    result_df["correct"] = result_df["pred"] == result_df["label"]

    output_dir = output_dir + '_' + eval_mode
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, group in result_df.groupby("filename"):
        file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_results.csv")
        group.to_csv(file_path, index=False, sep=';')
        print(f"Saved: {file_path} ({len(group)} rows)")

def run_jxplain(test_df, eval_mode="jxplain", output_dir="evaluation_results"):
    """
    Perform the Jxplain method to classify dynamic keys based on datatype entropy 
    and key entropy.

    Args:
        test_df (pd.DataFrame): DataFrame containing test data.
        eval_mode (str): Evaluation mode ('jxplain').
        output_dir (str): Directory to save evaluation results.
    """
    
    # Perform Jxplain: Predict if a key is dynamic (1) based on entropy conditions
    y_pred = ((test_df["datatype_entropy"] == 0) & (test_df["key_entropy"] > 1)).astype(int)
    y_test = test_df["label"]

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)

    # Calculate precision, recall, and F1-score for both classes
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

    # Calculate combined metrics (macro average)
    combined_precision, combined_recall, combined_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Calculate accuracy for positive and negative classes
    positive_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
    negative_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])

    print(f"Class 0 (Static) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1_score[0]:.4f}, Accuracy: {negative_accuracy:.4f}")
    print(f"Class 1 (Dynamic) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1_score[1]:.4f}, Accuracy: {positive_accuracy:.4f}")
    print(f"Both Classes (Overall) - Precision: {combined_precision:.4f}, Recall: {combined_recall:.4f}, F1 Score: {combined_f1:.4f}, Accuracy: {overall_accuracy:.4f}")


    # --- Match predictions back to DataFrame rows ---
    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = y_pred
    test_df["correct"] = test_df["pred"] == test_df["label"]

    # Keep only essential columns
    result_df = test_df[["filename", "path", "label", "pred", "correct"]]

    output_dir = output_dir + '_' + eval_mode
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Save per-file CSVs ---
    for filename, group in result_df.groupby("filename"):
        file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_results.csv")
        group.to_csv(file_path, index=False, sep=';')
        print(f"Saved: {file_path} ({len(group)} rows)")


def main():
    try:
        train_data, test_data, mode, *extra = sys.argv[-4:]
        version = extra[0].lower() if extra else "adapter"
        if version not in {"adapter", "full", "jxplain", "recg"}:
            raise ValueError("Invalid training mode. Use 'adapter', 'full', 'jxplain', or 'recg'.")
        if mode not in {"train", "eval", "jxplain", "recg"}:
            raise ValueError("Invalid mode. Use 'train', 'eval', 'jxplain', or 'recg'.")

        if mode == "train":
            train_df = pd.read_csv(train_data, delimiter=';')
            test_df = pd.read_csv(test_data, delimiter=';')
            train_model(train_df, test_df, version)
        elif mode == "eval":
            test_df = pd.read_csv(test_data, delimiter=';')
            model, _ = load_model_and_adapter(version)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)
            evaluate_model(test_df, eval_mode=version)
        elif mode == "jxplain":
            test_df = pd.read_csv(test_data, delimiter=';')
            run_jxplain(test_df, eval_mode=version, output_dir="evaluation_results")
        elif mode == "recg":
            schemas_dir = "test_jsons"
            test_df = pd.read_csv(test_data, delimiter=';')
            run_recg(test_df, schemas_dir, eval_mode=version, output_dir="evaluation_results")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_data> <test_data> <mode> [adapter|full|jxplain|recg]")
        sys.exit(1)

if __name__ == "__main__":
    main()