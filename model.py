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
    

def evaluate_model(test_df):
    """
    Evaluate the model on the test data.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test data.

    Returns:
        float: Average testing loss.
    """
    # Load model adapter
    model, tokenizer = load_model_and_adapter()

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x))
    
    total_loss = 0.0
    total_actual_labels = []
    total_predicted_labels = []

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Get the probabilities
            logits = outputs.logits

            # Calculate the testing loss
            testing_loss = outputs.loss

            # Need to average the loss if we are using DataParallel
            if testing_loss.dim() > 0:
                testing_loss = testing_loss.mean()
            total_loss += testing_loss.item()

            # Get the actual and predicted labels
            actual_labels = labels.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            total_actual_labels.extend(actual_labels)
            total_predicted_labels.extend(predicted_labels)

        average_loss = total_loss / len(test_loader)

    # Calculate the accuracy, precision, recall, f1 score of the positive class
    true_labels_positive, predicted_labels_positive = filter_labels_positive(total_actual_labels, total_predicted_labels)
    dynamic_accuracy = accuracy_score(true_labels_positive, predicted_labels_positive)
    dynamic_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1)

    # Calculate the accuracy, precision, recall, f1 score of the negative class
    true_labels_negative, predicted_labels_negative = filter_labels_negative(total_actual_labels, total_predicted_labels)
    static_accuracy = accuracy_score(true_labels_negative, predicted_labels_negative)
    static_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=0)
    
    print(f"dynamic accuracy: {dynamic_accuracy}, testing_loss: {average_loss}, dynamic precision: {dynamic_precision}, dynamic recall: {dynamic_recall}, dynamic F1: {dynamic_f1}")
    print(f"static accuracy: {static_accuracy}, testing_loss: {average_loss}, static precision: {static_precision}, static recall: {static_recall}, static F1: {static_f1}")

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
    print(f"Loaded and activated adapter: {adapter_name}")
    
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
    y_pred = ((test_df["datatype_entropy"] == 0) & (test_df["key_entropy"] > 1)).astype(int)
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




def main():
    try:
        # Parse command-line arguments
        train_data, test_data, mode = sys.argv[-3:]

        # Ensure mode is valid
        if mode not in {"train", "test", "jxplain"}:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")
        
        if mode == "train":
            train_df = pd.read_csv(train_data, sep=";")
            test_df = pd.read_csv(test_data, sep=";")
            train_model(train_df, test_df)
        elif mode == "test":
            test_df = pd.read_csv(test_data, sep=";")
            evaluate_model(test_df)
        else:
            test_df = pd.read_csv(test_data, sep=";")
            run_jxplain(test_df)

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_data> <test_data> <mode>")
        sys.exit(1)


if __name__ == "__main__":
    main()
