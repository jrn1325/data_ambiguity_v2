import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import tqdm
import wandb
from adapters import AutoAdapterModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")

# Create constant variables
SCHEMA_KEYWORDS = ["definitions", "$defs", "properties", "additionalProperties", "patternProperties", "oneOf", "allOf", "anyOf", "items", "type", "not"]
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
BATCH_SIZE = 120
MAX_TOKEN_LEN = 512
ADAPTER_NAME = "data_ambiguity"
MODEL_NAME = "microsoft/codebert-base"
PATH = "./adapter-model"
# Use os.path.expanduser to expand '~' to the full home directory path
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"


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
        distinct_subkeys = self.data.iloc[idx]["schema"]
        label = torch.tensor(self.data.iloc[idx]["label"], dtype=torch.long)

        tokenized_distinct_subkeys = self.tokenizer(
            distinct_subkeys,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_length
        )

        return {
            "input_ids": tokenized_distinct_subkeys["input_ids"].squeeze(0),
            "attention_mask": tokenized_distinct_subkeys["attention_mask"].squeeze(0),
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
    model.add_classification_head(ADAPTER_NAME, num_labels=2)
    model.add_adapter(ADAPTER_NAME, config="seq_bn")
    model.set_active_adapters(ADAPTER_NAME)
    model.train_adapter(ADAPTER_NAME)
    wandb.watch(model)
    return model, tokenizer
   

def train_model(train_df, test_df):

    accumulation_steps = 4
    learning_rate = 1e-6
    num_epochs = 75

    # Start a new wandb run to track this script
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

    # Initialize tokenizer, model with adapter and classification head
    model, tokenizer = initialize_model()

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Use all available GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = CustomDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x))

    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    early_stopper = EarlyStopper(patience=1, min_delta=0.005) # training steps if the validation loss does not decrease by at least 0.005 for 1 consecutive epochs

    # Train the model
    model.train()
    pbar = tqdm.tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, position=1, leave=False, total=len(train_dataloader))):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
  
            # Calculate the training loss
            training_loss = outputs.loss

            # Need to average the loss if we are using DataParallel
            if training_loss.dim() > 0:
                training_loss = training_loss.mean()

            training_loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        
            total_loss += training_loss.item()
        
        average_loss = total_loss / len(train_dataloader)
        wandb.log({"training_loss": average_loss})

        # Test the model
        testing_loss = test_model(test_df, tokenizer, model, device, wandb)

        if early_stopper.early_stop(testing_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    # Save the adapter
    save_model_and_adapter(model.module)
    wandb.save(f"{PATH}/*")

    
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
    
    #print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {f1}')
    wandb.log({"dynamic accuracy": dynamic_accuracy, "testing_loss": average_loss, "dynamic precision": dynamic_precision, "dynamic recall": dynamic_recall, "dynamic F1": dynamic_f1})
    wandb.log({"static accuracy": static_accuracy, "static precision": static_precision, "static recall": static_recall, "static F1": static_f1})


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

    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)

    # Calculate precision, recall, and F1-score for both classes
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1])

    # Calculate combined metrics (macro average)
    combined_precision, combined_recall, combined_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # Calculate accuracy for positive and negative classes
    positive_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])  # Accuracy for positive class
    negative_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])  # Accuracy for negative class

    # Print performance metrics for the negative class (static)
    print(f"Class 0 (Static) - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1 Score: {f1_score[0]:.4f}, Accuracy: {negative_accuracy:.4f}")

    # Print performance metrics for the positive class (dynamic)
    print(f"Class 1 (Dynamic) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1 Score: {f1_score[1]:.4f}, Accuracy: {positive_accuracy:.4f}")

    # Print combined metrics
    print(f"Both Classes (Overall) - Precision: {combined_precision:.4f}, Recall: {combined_recall:.4f}, F1 Score: {combined_f1:.4f}, Accuracy: {overall_accuracy:.4f}")



def main():
    try:
        # Parse command-line arguments
        train_data, test_data, mode = sys.argv[-3:]

        # Ensure mode is valid
        if mode not in {"train", "test", "jxplain"}:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")
        
        if mode == "train":
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)
            train_model(train_df, test_df)
        elif mode == "test":
            test_df = pd.read_csv(test_data)
            evaluate_model(test_df)
        else:
            test_df = pd.read_csv(test_data)
            run_jxplain(test_df)

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_size> <random_value> <mode>")
        sys.exit(1)


if __name__ == "__main__":
    main()
