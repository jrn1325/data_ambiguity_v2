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
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
BATCH_SIZE = 120
MAX_TOKEN_LEN = 512
ADAPTER_NAME = "data_ambiguity"
MODEL_NAME = "microsoft/codebert-base"
PATH = "./adapter-model"
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
    """
    Trains a CodeBERT model using the provided training and testing datasets.

    Args:
        train_df (pd.DataFrame): Training dataset containing input text and labels.
        test_df (pd.DataFrame): Testing dataset containing input text and labels.

    Returns:
        None
    """
    accumulation_steps = 4
    learning_rate = 1e-6
    num_epochs = 75

    # Initialize Weights and Biases (wandb) to track the training process
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

    # Initialize the tokenizer and model with an adapter and classification head
    model, tokenizer = initialize_model()

    # Set up the optimizer with AdamW optimization
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # If multiple GPUs are available, use DataParallel to distribute the model across them
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set the device for model training (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the custom dataset and DataLoader for training
    train_dataset = CustomDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x))

    # Set up a learning rate scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Initialize an early stopper to stop training if the validation loss plateaus
    early_stopper = EarlyStopper(patience=1, min_delta=0.005)  # Stops if no improvement in 1 epoch

    # Train the model over multiple epochs
    model.train()
    pbar = tqdm.tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        
        # Iterate over the training data
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, position=1, leave=False, total=len(train_dataloader))):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            
            # Forward pass to calculate the model outputs and training loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            training_loss = outputs.loss

            # If using DataParallel, average the loss across multiple GPUs
            if training_loss.dim() > 0:
                training_loss = training_loss.mean()

            # Backpropagate the loss
            training_loss.backward()

            # Accumulate gradients and perform an optimizer step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                lr_scheduler.step()  # Update the learning rate
                optimizer.zero_grad()  # Clear gradients
        
            total_loss += training_loss.item()  # Track total loss
        
        # Calculate and log the average training loss for the epoch
        average_loss = total_loss / len(train_dataloader)
        wandb.log({"training_loss": average_loss})

        # Evaluate the model on the test dataset after each epoch
        testing_loss = test_model(test_df, tokenizer, model, device, wandb)

        # Check if early stopping criteria are met
        if early_stopper.early_stop(testing_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    # Save the model and its adapter for future use
    save_model_and_adapter(model.module)
    wandb.save(f"{PATH}/*")

    
def test_model(test_df, tokenizer, model, device, wandb):
    """
    Evaluates the trained CodeBERT model on a given test dataset and calculates
    performance metrics such as accuracy, precision, recall, and F1 score for both
    dynamic (positive) and static (negative) classifications.

    Args:
        test_df (pd.DataFrame): Test dataset containing input text and labels.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing input text.
        model (nn.Module): The trained model to be evaluated.
        device (torch.device): The device on which to perform the evaluation (CPU or GPU).
        wandb (wandb): Weights and Biases object for logging performance metrics.

    Returns:
        average_loss (float): The average loss calculated over the test dataset.
    """

    # Prepare the test dataset and DataLoader for evaluation
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x))

    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0

    # Lists to store actual and predicted labels
    total_actual_labels = []
    total_predicted_labels = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            
            # Forward pass to get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Extract logits (raw model predictions before applying softmax)
            logits = outputs.logits

            # Calculate the loss for this batch
            testing_loss = outputs.loss

            # Average the loss across multiple GPUs if DataParallel is used
            if testing_loss.dim() > 0:
                testing_loss = testing_loss.mean()
            total_loss += testing_loss.item()

            # Get the actual labels and predicted labels from the logits
            actual_labels = labels.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            total_actual_labels.extend(actual_labels)
            total_predicted_labels.extend(predicted_labels)

    # Calculate the average loss across the entire test dataset
    average_loss = total_loss / len(test_loader)

    # Calculate accuracy, precision, recall, and F1 score for the positive class (dynamic)
    true_labels_positive, predicted_labels_positive = filter_labels_positive(total_actual_labels, total_predicted_labels)
    dynamic_accuracy = accuracy_score(true_labels_positive, predicted_labels_positive)
    dynamic_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1, average="weighted")
    dynamic_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1, average="weighted")
    dynamic_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1, average="weighted")


    # Calculate accuracy, precision, recall, and F1 score for the negative class (static)
    true_labels_negative, predicted_labels_negative = filter_labels_negative(total_actual_labels, total_predicted_labels)
    static_accuracy = accuracy_score(true_labels_negative, predicted_labels_negative, average="weighted")
    static_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=0, average="weighted")
    static_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=0, average="weighted")
    static_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=0, average="weighted")

    # Log the performance metrics for dynamic and static classes to Weights and Biases (wandb)
    wandb.log({
        "dynamic accuracy": dynamic_accuracy,
        "testing_loss": average_loss,
        "dynamic precision": dynamic_precision,
        "dynamic recall": dynamic_recall,
        "dynamic F1": dynamic_f1
    })
    wandb.log({
        "static accuracy": static_accuracy,
        "static precision": static_precision,
        "static recall": static_recall,
        "static F1": static_f1
    })

    # Return the average test loss for this evaluation run
    return average_loss
    

def evaluate_model(test_df):
    """
    Evaluates the model on a given test dataset, calculates performance metrics 
    (accuracy, precision, recall, F1 score) for both dynamic (positive) and static 
    (negative) classifications, and prints the results.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test data with input text and labels.

    Returns:
        float: The average testing loss calculated over the test dataset.
    """

    # Load the pre-trained model and tokenizer, along with any attached adapters
    model, tokenizer = load_model_and_adapter()

    # If multiple GPUs are available, use DataParallel for distributing the model across GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the test dataset and DataLoader
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x))

    total_loss = 0.0  # To accumulate the total loss over all batches
    total_actual_labels = []  # To store the actual labels from the test set
    total_predicted_labels = []  # To store the predicted labels from the model

    # Set the model to evaluation mode (disabling dropout, etc.)
    model.eval()

    # Disable gradient calculation for evaluation (saves memory and computation)
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            # Forward pass through the model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Extract the logits (raw model predictions before applying softmax)
            logits = outputs.logits

            # Calculate the loss for the current batch
            testing_loss = outputs.loss

            # If using DataParallel, average the loss across GPUs
            if testing_loss.dim() > 0:
                testing_loss = testing_loss.mean()
            total_loss += testing_loss.item()

            # Extract the actual labels and predicted labels for this batch
            actual_labels = labels.cpu().numpy()
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            total_actual_labels.extend(actual_labels)
            total_predicted_labels.extend(predicted_labels)

        # Calculate the average loss across the entire test dataset
        average_loss = total_loss / len(test_loader)

    # Calculate accuracy, precision, recall, and F1 score for the positive class (dynamic)
    true_labels_positive, predicted_labels_positive = filter_labels_positive(total_actual_labels, total_predicted_labels)
    dynamic_accuracy = accuracy_score(true_labels_positive, predicted_labels_positive)
    dynamic_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1)
    dynamic_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1)

    # Calculate accuracy, precision, recall, and F1 score for the negative class (static)
    true_labels_negative, predicted_labels_negative = filter_labels_negative(total_actual_labels, total_predicted_labels)
    static_accuracy = accuracy_score(true_labels_negative, predicted_labels_negative)
    static_precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=0)
    static_f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=0)

    # Print the evaluation metrics for both dynamic and static classes
    print(f"dynamic accuracy: {dynamic_accuracy}, testing_loss: {average_loss}, dynamic precision: {dynamic_precision}, dynamic recall: {dynamic_recall}, dynamic F1: {dynamic_f1}")
    print(f"static accuracy: {static_accuracy}, testing_loss: {average_loss}, static precision: {static_precision}, static recall: {static_recall}, static F1: {static_f1}")

    # Return the average testing loss
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
    combined_precision, combined_recall, combined_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

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
