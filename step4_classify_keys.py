import json
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from adapters import AutoAdapterModel
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer, get_scheduler

import warnings
warnings.filterwarnings("ignore")

# Create constant variables
KEYS_TO_REMOVE = ["definitions", "$defs", "properties", "patternProperties", "oneOf", "allOf", "anyOf", "items"]
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
RANDOM_VALUE = 101
BATCH_SIZE = 64
ADAPTER_NAME = "data_ambiguity"
MODEL_NAME = "microsoft/codebert-base"
PATH = "./adapter-model"


def extract_paths(doc, path = ('$',), values = []):
    """Get the path of each key and its value from the json documents

    Args:
        doc (dict): JSON document
        path (tuple, optional): list of keys full path. Defaults to ('$',).
        values (list, optional): list of keys' values. Defaults to [].

    Raises:
        ValueError: Returns an error if the json object is not a dict or list

    Yields:
        dict: list of JSON object key value pairs
    """
    if isinstance(doc, dict):
        iterator = doc.items()
    elif isinstance(doc, list):
        if len(doc) > 0:
            #iterator = [('*', doc[0])]
            iterator = [] # We will handle lists later
        else:
            iterator = []
    else:
        raise ValueError('Invalid type')
  
    for key, value in iterator:
        yield path + (key,), value
        if isinstance(value, (dict, list)):
            yield from extract_paths(value, path + (key,), values)
    

def process_document(doc, prefix_paths_dict, path_types_dict): 
    """
    Extracts paths from the given JSON document and stores them in a dictionary,
    grouping paths that share the same prefix.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        prefix_paths_dict (dict): A dictionary where the keys are path prefixes 
                                  (tuples of path segments, excluding the last segment),
                                  and the values are sets of full paths that share the same prefix.
        path_types_dict, p): (dict): A dictionary where the keys are path prefixes and the values are lists of types of the nested keys.
    """
    
    for path, value in extract_paths(doc):
        if len(path) > 1:
            prefix = path[:-1]
            prefix_paths_dict[prefix].add(path)
        path_types_dict[path].add(type(value).__name__)


def create_dataframe(prefix_paths_dict, dataset):
    """
    Create a DataFrame from dictionaries containing path types and path frequencies.

    Args:
        path_types_dict (dict): A dictionary where keys are paths (tuples) and values are sets of types.
        dataset (str): The name of the dataset.

    Returns:
        pd.DataFrame: A DataFrame with 'path', 'distinct_subkeys', and 'filename' columns.
    """
    data = []

    # Calculate the number of distinct subkeys under each parent key
    for path, subpaths in prefix_paths_dict.items():
        distinct_subkeys = {k[-1] for k in subpaths}
        
        # Limit the number of distinct subkeys collected to the upper bound
        if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
            distinct_subkeys = set(list(distinct_subkeys)[:DISTINCT_SUBKEYS_UPPER_BOUND])

        # Append the path and the distinct subkeys to the data list
        data.append({
            "path": path,
            "distinct_subkeys": json.dumps(list(distinct_subkeys))
        })
    
    # Create the DataFrame from the data list
    df = pd.DataFrame(data)
    df["filename"] = dataset
    
    return df


def compare_tuples(tuple1, tuple2):
    if len(tuple1) != len(tuple2):
        return False

    for item1, item2 in zip(tuple1, tuple2):
        if isinstance(item2, str) and re.match(item2, item1):
            continue
        elif item1 != item2:
            return False
    return True


def label_paths(dataset, df, dynamic_paths):
    df["label"] = 0

    for key_path in dynamic_paths:
        if "items" in key_path:
            continue
        key_path = tuple([i for i in key_path if i not in KEYS_TO_REMOVE])

        for index, row in df.iterrows():
            if compare_tuples(row["path"], key_path) and row["filename"] == dataset:
                df.at[index, "label"] = 1
    return df


def extract_pattern_properties_parents(schema, path = ('$',)):
    """Get the keys under "patternProperties" in JSON schemas

    Args:
        schema (dict): JSON schema
        path (tuple, optional): path to key. Defaults to ().

    Yields:
        tuple: key path
    """
    if isinstance(schema, dict):
        if "patternProperties" in schema:
            yield path
        elif "properties" not in schema and schema.get("additionalProperties", False) is not False:
            yield path

        for key, value in schema.items():
            yield from extract_pattern_properties_parents(value, path + (key,))
                
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            yield from extract_pattern_properties_parents(item, path)


def preprocess_data(files_folder):
    frames = []

    datasets = os.listdir(files_folder)
    for dataset in tqdm.tqdm(datasets):
        schema_path = os.path.join("valid_schemas", dataset)
        with open(schema_path, 'r') as schema_file:
            json_schema = json.load(schema_file)
            dynamic_paths = list(extract_pattern_properties_parents(json_schema))
            if not dynamic_paths:
                continue
        prefix_paths_dict = defaultdict(set)
        path_types_dict = defaultdict(set)
    

        with open(os.path.join(files_folder, dataset), 'r') as file:
            for count, line in enumerate(file):
                doc = json.loads(line)
                process_document(doc, prefix_paths_dict, path_types_dict)

        #sys.stderr.write(f"Creating a dataframe for {dataset}\n")
        df = create_dataframe(prefix_paths_dict, dataset)
            
        #sys.stderr.write(f"Labeling data for {dataset}\n")
        df = label_paths(dataset, df, dynamic_paths)
        frames.append(df)

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index = True)
    df = df.reset_index(drop = True)
    
    return df



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
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distinct_subkeys = self.data.iloc[idx]["distinct_subkeys"]
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


def split_data(df, test_size):
    # Split the data into training and testing sets based on filenames
    filenames = df["filename"].tolist()
    train_filenames, test_filenames = train_test_split(filenames, test_size=test_size, random_state=RANDOM_VALUE)
    train_df = df[df["filename"].isin(train_filenames)]
    test_df = df[df["filename"].isin(test_filenames)]
    return train_df, test_df
   

def train_model(train_df, test_df):

    accumulation_steps = 4
    learning_rate = 2e-5
    num_epochs = 25

    # Start a new wandb run to track this script
    wandb.init(
        project="custom-codebert_all_files_" + str(num_epochs),
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

    early_stopper = EarlyStopper(patience=5, min_delta=0.005)


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
    save_model_and_adapter(model)

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
    true_labels_positive, predicted_labels_positive = filter_labels(total_actual_labels, total_predicted_labels)
    accuracy = accuracy_score(true_labels_positive, predicted_labels_positive)
    precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1)
    recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1)
    f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1)
    
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {f1}')
    wandb.log({"accuracy": accuracy, "testing_loss": average_loss, "precision": precision, "recall": recall, "F1": f1})

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
        true_labels_positive, predicted_labels_positive = filter_labels(total_actual_labels, total_predicted_labels)
        accuracy = accuracy_score(true_labels_positive, predicted_labels_positive)
        precision = precision_score(total_actual_labels, total_predicted_labels, pos_label=1)
        recall = recall_score(total_actual_labels, total_predicted_labels, pos_label=1)
        f1 = f1_score(total_actual_labels, total_predicted_labels, pos_label=1)
        
        print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1: {f1}')

    return average_loss


def filter_labels(true_labels, predicted_labels):
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


def main():
    dataset_folder, test_size, mode = sys.argv[-3:]
    df = preprocess_data(dataset_folder)
    train_df, test_df = split_data(df, float(test_size))
   
    if mode == "train":
        train_model(train_df, test_df)
    elif mode == "test":
        evaluate_model(test_df)
    
 
if __name__ == "__main__":
    main()
