import json
import math
import numpy as np
import os
import pandas as pd
import re
import sys
import tqdm

from adapters import AutoAdapterModel
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoTokenizer, get_scheduler

import torch
import torch.nn.functional as F
import wandb
import warnings
warnings.filterwarnings("ignore")

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000

def extract_paths(doc, paths = [], types = [], num_nested_keys = []):
    """Get the path of each key and its value from the json documents

    Args:
        doc (dict): JSON document
        paths (list, optional): list of keys full path. Defaults to [].
        types (list, optional): list of keys' datatypes. Defaults to [].
        num_nested_keys (list, optional): number of paths nested keys. Defaults to [].

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
            iterator = []
        else:
            iterator = []
    else:
        raise ValueError('Invalid type')
  
    for (key, value) in iterator:
        yield paths + [key], value, len(value) if isinstance(value, (dict, list))  else 0
        if isinstance(value, (dict, list)):
            yield from extract_paths(value, paths + [key], types, num_nested_keys)
    

def process_document(doc, freq_paths_dict, prefix_paths_dict, prefix_types_dict): 
    """Extract information from each JSON document

    Args:
        doc (dict): JSON document
        freq_paths_dict (dict): stores all paths and their frequencies
        prefix_paths_dict (dict): stores paths with same prefixes
        prefix_types_dict (dict): stores paths' values datatypes
    """
    new_obj = {"$": doc}
    for (path, value, nested_keys_count) in extract_paths(new_obj):
        path = tuple(path)
        freq_paths_dict[path] += 1
        
        if len(path) > 1:
            prefix = path[:-1]
            prefix_paths_dict[prefix].add(path)
            
        prefix_types_dict[path].add(type(value).__name__)
              

def create_dataframe(num_docs, dataset_name, freq_paths_dict, prefix_paths_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict, prefix_types_dict):
    """Create a dataframe

    Args:
        num_docs (int): total number of JSON documents in a dataset
        dataset_name (str): JSON dataset name
        freq_paths_dict (dict): dictionary of paths and their frequencies
        prefix_paths_dict (dict): dictionary of paths and their prefixes
        prefix_freqs_dict (dict): dictionary of prefixes and their frequencies
        prefix_nested_keys_freq_dict (dict): dictionary of paths and their nested keys frequencies
        prefix_types_dict (dict): dictionary of paths' datatype and their prefixes 

    Returns:
        2d list: dataframe
    """
   
    for(prefix, p_list) in prefix_paths_dict.items(): 
        prefix_freqs_dict[prefix] = [freq_paths_dict[p] for p in p_list]

    # Loop through the frequency path dictionary and record the frequency of each 
    for (path, freq) in freq_paths_dict.items():
        # Check if the delimiter is in a path
        if len(path) > 1:
            prefix = path[:-1]
            prefix_nested_keys_freq_dict[prefix].append(freq)
    
    paths = list(freq_paths_dict.keys())
    df = pd.DataFrame({"Path": paths})
    # Calculate the Jxplain entropy
    df = is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs)
    #df = populate_dataframe(num_docs, df, prefix_paths_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict, prefix_types_dict)
    
    df["Filename"] = dataset_name
    
    return df


def populate_dataframe(df, prefix_paths_dict, prefix_freqs_dict):
    """Create a dataframe of all the features

    Args:
        df (2d list): dataframe with JSON paths
        prefix_paths_dict (dict): dictionary of paths and their prefixes
        prefix_freqs_dict (dict): dictionary of paths and their frequencies

    Returns:
        2d list: dataframe
    """
    
    # Create a new column for distinct subkeys
    df["Distinct_subkeys"] = None

    # Calculate complex descriptive statistics
    for path in prefix_freqs_dict.keys():
        # Get the location of the key within the dataframe
        key_loc = df.loc[df.Path == path].index[0]

        # Get the number of unique subkeys under a parent key
        distinct_subkeys = set()
        for k in prefix_paths_dict[path]:
            distinct_subkeys.add(k[-1])
            if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
                break
            
        df.at[key_loc, "Distinct_subkeys"] = list(distinct_subkeys)

    
    # Remove rows of keys that have no nested keys
    df = df[df.Distinct_subkeys.notnull()]
    df.dropna(inplace=True)
    return df


def find_dynamic_paths(obj, path=""):
    """Use the schemas associated with the JSON datasets to find the labels

    Args:
        obj (dict, list): JSON object
        path (str, optional): path of JSON object. Defaults to "".

    Yields:
        tuple: path of key
    """
    if isinstance(obj, dict):
        if "properties" in obj and "patternProperties" in obj:
            pass
        elif "properties" in obj:
            for (k, v) in obj["properties"].items():
                yield from find_dynamic_paths(v, path + "." + k)
        elif "patternProperties" in obj:
            if len(obj["patternProperties"]) == 1:
                yield path
                yield from find_dynamic_paths(next(iter(obj["patternProperties"].values())), path + ".*")
        
    elif isinstance(obj, list):
        for (i, v) in enumerate(obj):
            yield from find_dynamic_paths(v, path + "[" + str(i) + "]")


def compare_tuples(tuple1, tuple2):
    if len(tuple1) != len(tuple2):
        return False

    for item1, item2 in zip(tuple1, tuple2):
        if isinstance(item2, str) and re.match(item2, item1):
            continue
        elif item1 != item2:
            return False
    return True


def label_paths(dataset, df, dynamic_paths, keys_to_remove):
    df["Category"] = 0

    for key_path in dynamic_paths:
        key_path = ["$"] + key_path
        if "items" in key_path:
            continue
        #print(key_path)
        key_path = tuple([i for i in key_path if i not in keys_to_remove])
        #print(key_path)

        for index, row in df.iterrows():
            if compare_tuples(row["Path"], key_path) and row["Filename"] == dataset:
                df.at[index, "Category"] = 1
    return df


def extract_pattern_properties_parents(schema, path=[]):
    """Get the keys under "patternProperties" in JSON schemas

    Args:
        schema (dict): JSON schema
        path (list, optional): path to key. Defaults to [].

    Yields:
        tuple: key path
    """
    if isinstance(schema, dict):
        if "patternProperties" in schema:
            yield path
        elif "properties" not in schema and schema.get("additionalProperties", False) is not False:
            yield path

        for key, value in schema.items():
            yield from extract_pattern_properties_parents(value, path + [key])
                
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            yield from extract_pattern_properties_parents(item, path)


def initialize_dicts():
    return (
        defaultdict(lambda: 0),  # freq_paths_dict
        defaultdict(set),  # prefix_paths_dict
        defaultdict(set),  # prefix_types_dict
        defaultdict(lambda: 0),  # prefix_freqs_dict
        defaultdict(list)  #prefix_nested_keys_freq_dict
    )


def preprocess_data(files_folder):
    frames = []
    keys_to_remove = ["definitions", "$defs", "properties", "patternProperties", "oneOf", "allOf", "anyOf", "items"]

    datasets = os.listdir(files_folder)
    for dataset in tqdm.tqdm(datasets):
        schema_path = os.path.join("valid_schemas", dataset)
        with open(schema_path, 'r') as schema_file:
            json_schema = json.load(schema_file)
            dynamic_paths = list(extract_pattern_properties_parents(json_schema))
            if not dynamic_paths:
                continue

        freq_paths_dict, prefix_paths_dict, prefix_types_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict = initialize_dicts()
        num_docs = 0

        with open(os.path.join(files_folder, dataset), 'r') as file:
            #lines = islice(file, 10)
            for count, line in enumerate(file):
                json_doc = json.loads(line)
                process_document(json_doc, freq_paths_dict, prefix_paths_dict, prefix_types_dict)
                num_docs += count

        #sys.stderr.write(f"Creating a dataframe for {dataset}\n")
        df = create_dataframe(num_docs, dataset, freq_paths_dict, prefix_paths_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict, prefix_types_dict)
            
        #sys.stderr.write(f"Labeling data for {dataset}\n")
        df = label_paths(dataset, df, dynamic_paths, keys_to_remove)
        frames.append(df)

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index = True)
    
    return df



    """
    Input: Merged dataframe, writer object
    Output: Classifier results and summary
    Purpose: Perform ML classifier model to identify dynamic keys
    """
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    
    #df[["Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]] = df[["Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]].astype(float)
    #training_features = ["Percentage", "Nesting_level", "Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]
    #training_features = ["Datatype_entropy", "Key_entropy"]
    training_features = df.columns.difference(["Path", "Filename", "Category"])

    X = df.loc[:, training_features].values
    y = df.Category
    
    random_value = 101
   
    # Perform classification using the best features/predictors
    classify(X, y, df, training_features, writer, testing_size, random_value, use_logo=False)


def is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs):
    df["Datatype_entropy"] = np.nan
    df["Key_entropy"] = np.nan

    # Calculate datatype entropy
    for (key, value) in prefix_types_dict.items():
        # Check if values of the nested keys of a set have the same datatype
        #result = all(isinstance(nested_key, type(value[0])) for nested_key in value[1:])
        df.loc[df.Path == key, "Datatype_entropy"] = 0 if all(value) else 1
            
    # Calculate key entropy and add as a new column
    for (key, value) in prefix_nested_keys_freq_dict.items():
        key_entropy = 0
        for freq in value:
            key_entropy += (freq/num_docs) * math.log(freq/num_docs)
        df.loc[df.Path == key, "Key_entropy"] = -key_entropy
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
    def __init__(self, dataframe, tokenizer, max_length=1000):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distinct_subkeys = self.data.iloc[idx]["Distinct_subkeys"]
        label = torch.tensor(self.data.iloc[idx]["Category"], dtype=torch.float32)

        # Tokenize each element in the list separately with dynamic max_length
        tokenized_distinct_subkeys = self.tokenizer(
            distinct_subkeys,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return {
            "input_ids": tokenized_distinct_subkeys["input_ids"],
            "attention_mask": tokenized_distinct_subkeys["attention_mask"],
            "label": label
        }

def collate_fn(batch, tokenizer):
    # Get maximum sequence length in the batch
    max_len = max(len(item["input_ids"][0]) for item in batch)

    # Pad all values to the maximum length in the batch
    input_ids_padded = pad_sequence(
        [F.pad(item["input_ids"][0], pad=(0, max_len - len(item["input_ids"][0])), value=tokenizer.pad_token_id) for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    attention_mask_padded = pad_sequence(
        [F.pad(item["attention_mask"][0], pad=(0, max_len - len(item["attention_mask"][0])), value=0) for item in batch],
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack([item["label"] for item in batch])

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "label": labels
    }

def custom_codebert(df, test_size, random_value):
    accumulation_steps = 4
    batch_size = 8
    learning_rate = 5e-5
    num_epochs = 25

    # Start a new wandb run to track this script
    wandb.init(
        project="custom-codebert_all_files " + str(num_epochs),
        config={
            "accumulation_steps": accumulation_steps,
            "batch_size": batch_size,
            "dataset": "json-schemas",
            "epochs": num_epochs,
            "learning_rate": learning_rate,
        }
    )
    wandb.define_metric("accuracy", summary="max")
    wandb.define_metric("precision", summary="max")
    wandb.define_metric("recall", summary="max")
    wandb.define_metric("F1", summary="max")
    wandb.define_metric("training_loss", summary="min")
    wandb.define_metric("testing_loss", summary="min")

    # Initialize tokenizer, model with adapter and classification head
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoAdapterModel.from_pretrained("microsoft/codebert-base")
    model.add_classification_head("mrpc", num_labels=2)
    model.add_adapter("mrpc", config="seq_bn")
    model.set_active_adapters("mrpc")
    wandb.watch(model)

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_df, test_df = split_data(df, test_size, random_value)
    train_dataset = CustomDataset(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer))

    # Set up scheduler to adjust the learning rate during training
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    early_stopper = EarlyStopper(patience=5, min_delta=10)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train the model
    model.train()
    pbar = tqdm.tqdm(range(num_epochs), position=0, desc="Epoch")
    for epoch in pbar:
        total_loss = 0
        for i, batch in enumerate(tqdm.tqdm(train_dataloader, position=1, leave=False, total=len(train_dataloader))):
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Modify the loss calculation
            training_loss = loss_fn(outputs.logits[:, 0], labels)
            training_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += training_loss.item()
            torch.cuda.empty_cache()

        average_loss = total_loss / len(train_dataloader)
        wandb.log({"training_loss": average_loss})

        # Evaluate the model
        testing_loss = evaluate_data(test_df, tokenizer, batch_size, model, device, wandb)

        if early_stopper.early_stop(testing_loss):
            save_model_and_adapter(model)
            break

    # Save the adapter
    model.save_adapter("./adapter", "mrpc", with_head=True)
    
def split_data(df, test_size, random_value):
    # Split the data into training and testing sets based on filenames
    filenames = df["Filename"].tolist()
    train_filenames, test_filenames = train_test_split(filenames, test_size=test_size, random_state=random_value)
    train_df = df[df["Filename"].isin(train_filenames)]
    test_df = df[df["Filename"].isin(test_filenames)]
    return train_df, test_df

def evaluate_data(test_df, tokenizer, batch_size, model, device, wandb):
    test_dataset = CustomDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer))

    model.eval()
    total_loss = 0.0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    true_labels_predictions = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits[:, 0]

            # Calculate the testing loss
            testing_loss = loss_fn(logits, batch["label"])
            total_loss += testing_loss.item()

            # Get the actual and predicted labels
            actual_labels = batch["label"].cpu().numpy()
            predictions = torch.round(torch.sigmoid(outputs.logits[:, 0]))

            # Collect both actual and predicted labels only for instances where actual label is 1
            true_labels_actuals.extend(actual_labels[actual_labels == 1].tolist())
            true_labels_predictions.extend(predictions[actual_labels == 1].cpu().numpy())


    average_loss = total_loss / len(test_loader)

    # Convert to PyTorch tensors
    true_labels_actuals = torch.tensor(actual_labels)
    true_labels_predictions = torch.tensor(true_labels_predictions)
    
    accuracy = accuracy_score(true_labels_actuals, true_labels_predictions)
    precision = precision_score(true_labels_actuals, true_labels_predictions)
    recall = recall_score(true_labels_actuals, true_labels_predictions)
    f1 = f1_score(true_labels_actuals, true_labels_predictions)

    wandb.log({"testing_loss": average_loss, "precision": precision, "recall": recall, "F1": f1, "accuracy": accuracy})
    return average_loss

def save_model_and_adapter(model):
    # Save the entire model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)

    # Save the adapter
    adapter_path = "./adapter"
    model.save_adapter(adapter_path, "mrpc", with_head=True)

    # Log the model artifact
    artifact = wandb.Artifact("customized_codebert", type="model")
    artifact.add_file(model_path)
    artifact.add_dir(adapter_path)
    wandb.log_artifact(artifact)


def main():
    dataset_folder, test_size = sys.argv[-2:]
    df = preprocess_data(dataset_folder)
    df = df.reset_index(drop = True)
    '''
    train_df, test_df = split_data(df, test_size, 101)
    # Perform Jxplain1
    y_pred = ((test_df["Datatype_entropy"] == 0) & (test_df["Key_entropy"] > 1)).astype(int)
    print(len(y_pred), len(y_test))
    # Calculate the precision, recall, f1-score, and support of dynamic keys
    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
    '''
    #df.to_csv("data.csv")
    #df = pd.read_csv("data.csv")
    custom_codebert(df, float(test_size), 101)

 
if __name__ == '__main__':
    main()

