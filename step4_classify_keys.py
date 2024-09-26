import json
import jsonref
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
import torch.nn as nn
import tqdm
import wandb
from adapters import AutoAdapterModel
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
#from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
SCHEMA_FOLDER = os.path.expanduser("~/Desktop/schemas")
JSON_FOLDER = os.path.expanduser("~/Desktop/jsons")


def split_data(train_ratio=0.8, random_value=101):
    """
    Split the list of schemas into training and testing sets making sure that schemas from the same source are grouped together.

    Args:
        train_ratio (float, optional): The ratio of schemas to use for training. Defaults to 0.8.
        random_value (int, optional): The random seed value. Defaults to 101.

    Returns:
        tuple: A tuple containing the training set and testing set.
    """

    # Get the list of schema filenames
    schemas = os.listdir(SCHEMA_FOLDER)

    # Use GroupShuffleSplit to split the schemas into train and test sets
    gss = GroupShuffleSplit(train_size=train_ratio, random_state=random_value)

    # Make sure that schema names with the same first 3 letters are grouped together because they are likely from the same source
    train_idx, test_idx = next(gss.split(schemas, groups=[s[:4] for s in schemas]))

    # Create lists of filenames for the train and test sets
    train_set = [schemas[i] for i in train_idx]
    test_set = [schemas[i] for i in test_idx]

    return train_set, test_set


def load_and_dereference_schema(schema_path):
    """
    Load the JSON schema from the specified path and dereference any $ref pointers.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict: The dereferenced JSON schema.
    """
    
    try:
        with open(schema_path, 'r') as schema_file:
            schema = jsonref.load(schema_file)
            return schema
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}")
    return {}
    

def has_pattern_properties(schema):
    """
    Recursively checks if patternProperties exist in the schema.

    Args:
        schema (dict): The JSON schema to check.

    Returns:
        bool: True if patternProperties is found, False otherwise.
    """
    if not isinstance(schema, dict):
        return False

    if "patternProperties" in schema:
        return True

    # Recursively check for patternProperties in nested schemas
    for key in ["properties", "items", "additionalProperties", "prefixItems", "anyOf", "oneOf", "allOf", "then"]:
        if key in schema:
            value = schema[key]
            if isinstance(value, dict):
                if has_pattern_properties(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if has_pattern_properties(item):
                        return True
    return False


def handle_additional_properties(schema, current_path):
    """
    Handle additionalProperties in the JSON schema.
    
    Args:
        schema (dict): The JSON schema containing additionalProperties.
        current_path (tuple): The current path in the schema hierarchy.
        
    Yields:
        tuple: Paths based on the additionalProperties settings.
    """
    additional_props = schema.get("additionalProperties")
    
    if additional_props is False:
        yield current_path 
    elif additional_props is True:
        yield current_path + ("wildpath",) 
    elif isinstance(additional_props, dict):
        if "additionalProperties" in additional_props and additional_props["additionalProperties"] is False:
            yield current_path 
        else:
            yield current_path + ("wildpath",) 
        yield from extract_static_schema_paths(additional_props, current_path)


def extract_static_schema_paths(schema, current_path=("$",)):
    """
    Extracts paths from a JSON schema where properties are explicitly defined.
    Handles `additionalProperties` and `properties` correctly.
    
    Args:
        schema (dict): The JSON schema to extract paths from.
        current_path (tuple): The current path in the schema hierarchy, defaults to ("$",).
        
    Yields:
        tuple: Explicitly defined property paths whose values are of type dict.
    """

    if not isinstance(schema, dict):
        return

    # Process defined properties
    if "properties" in schema:
        for key, value in schema["properties"].items():
            new_path = current_path + (key,)

            if isinstance(value, dict):
                if value.get("type") == "object":
                    yield from handle_additional_properties(value, new_path) 

                yield from extract_static_schema_paths(value, new_path)

    # Handle additionalProperties
    if "additionalProperties" in schema:
        yield from handle_additional_properties(schema, current_path)

    # Handle array-related keywords
    if schema.get("type") == "array":
        if "items" in schema:
            if isinstance(schema["items"], dict):
                yield from extract_static_schema_paths(schema["items"], current_path + ("*",))

        for item in schema.get("prefixItems", []):
            if isinstance(item, dict):
                yield from extract_static_schema_paths(item, current_path + ("*",))

    # Process other schema keywords like `anyOf`, `oneOf`, `allOf`, and `then`
    for applicator in ["anyOf", "oneOf", "allOf"]:
        for item in schema.get(applicator, []):
            if isinstance(item, dict):
                yield from extract_static_schema_paths(item, current_path)

    if "then" in schema and isinstance(schema["then"], dict):
        yield from extract_static_schema_paths(schema["then"], current_path)


def extract_implicit_schema_paths(schema, current_path=("$",)):
    """
    Extracts paths from a JSON schema where properties are not explicitly defined,
    yielding a path and a binary flag based on the structure of additionalProperties.

    Args:
        schema (dict or list): The JSON schema to extract paths from.
        current_path (tuple): The current path in the schema hierarchy.
        
    Yields:
        tuple: A tuple containing the path and a binary flag.
               1 if additionalProperties is a structured object,
               2 if additionalProperties is not structured or is not declared.
    """
    
    if not isinstance(schema, dict):
        return
    
    # Only check additionalProperties and patternProperties if the schema is an object
    if schema.get("type") == "object":
        additional_props = schema.get("additionalProperties", True)
        if additional_props is True or isinstance(additional_props, dict) or "patternProperties" in schema:
            yield current_path, 1
   
    # Process properties
    for key, value in schema.get("properties", {}).items():
        yield from extract_implicit_schema_paths(value, current_path + (key,))
    
    # Process schema-level keywords like anyOf, oneOf, allOf
    for applicator in ["anyOf", "oneOf", "allOf"]:
        for item in schema.get(applicator, []):
            yield from extract_implicit_schema_paths(item, current_path)

    # Process array-related schema keywords
    if "items" in schema and schema.get("type") == "array":
        yield from extract_implicit_schema_paths(schema["items"], current_path + ("*",))
    
    for item in schema.get("prefixItems", []):
        yield from extract_implicit_schema_paths(item, current_path + ("*",))

    # Process conditional schema keywords like then
    if "then" in schema:
        yield from extract_implicit_schema_paths(schema["then"], current_path)
  

def extract_paths(doc, path = ("$",), values = []):
    """Get the path of each key and its value from the json documents.

    Args:
        doc (dict): JSON document.
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
        iterator = [('*', item) for item in doc] if doc else []
    else:
        raise ValueError("Expected dict or list, got {}".format(type(doc).__name__))
  
    for key, value in iterator:
        yield path + (key,), value
        if isinstance(value, (dict, list)):
            yield from extract_paths(value, path + (key,), values)


def match_properties(schema, document):
    """Check if there is an intersection between the schema (properties or patternProperties) and the document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is a match, False otherwise.
    """
    # Check if the schema has 'properties' or 'patternProperties'
    schema_properties = schema.get("properties", {})
    #pattern_properties = schema.get("patternProperties", {})

    # Check if schema has additionalProperties set to true
    #additional_properties = schema.get("additionalProperties", False)

    # Check for matching properties from 'properties' or 'patternProperties'
    matching_properties_count = sum(1 for key in document if key in schema_properties)
    '''
    # If patternProperties exist, match keys using regex patterns
    for pattern, _ in pattern_properties.items():
        for key in document:
            if re.fullmatch(pattern, key):
                matching_properties_count += 1
    '''
    # If there are matching properties, return True
    if matching_properties_count > 0:
        return True

    # If no properties are defined but additionalProperties is allowed, consider it a match
    #if additional_properties is True:
    #    return True

    # Return False if no match is found
    return False


def process_document(doc, prefix_paths_dict, path_types_dict): 
    """
    Extracts paths from the given JSON document and stores them in a dictionary,
    grouping paths that share the same prefix.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        prefix_paths_dict (dict): A dictionary where the keys are path prefixes 
                                  (tuples of path segments, excluding the last segment),
                                  and the values are sets of full paths that share the same prefix.
        path_types_dict: (dict): A dictionary where the keys are path prefixes and the values are lists of types of the nested keys.
    """
    for path, value in extract_paths(doc):
        if len(path) > 1:
            prefix = path[:-1]
            prefix_paths_dict[prefix].add(path)
        #path_types_dict[path].add(type(value).__name__)


def create_dataframe(prefix_paths_dict, dataset):
    """
    Create a DataFrame from dictionaries containing path types and path prefixes.

    Args:
        prefix_paths_dict (dict): A dictionary where keys are path prefixes and values are sets of paths.
        dataset (str): The name of the dataset.

    Returns:
        pd.DataFrame: A DataFrame with 'path', 'distinct_subkeys', and 'filename' columns.
    """
    data = []

    # Get the distinct subkeys under each path
    for path, subpaths in prefix_paths_dict.items():
        distinct_subkeys = {k[-1] for k in subpaths}
        
        # Limit the number of distinct subkeys
        if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
            distinct_subkeys = set(list(distinct_subkeys)[:DISTINCT_SUBKEYS_UPPER_BOUND])

        if len(distinct_subkeys) == 1 and list(distinct_subkeys)[0] == '*':
            continue
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
    """
    Compare two tuples element-wise, allowing for regular expression matching on elements in tuple2.
    If item2 is a valid regex pattern (string), it tries to match item1 with item2 as a regex.
    Otherwise, it performs direct equality comparison.

    Args:
        tuple1 (tuple): The first tuple, contains actual values.
        tuple2 (tuple): The second tuple, which may contain regex patterns or direct values.

    Returns:
        bool: True if tuples are considered equal, False otherwise.
    """
    if len(tuple1) != len(tuple2):
        return False

    for item1, item2 in zip(tuple1, tuple2):
        if isinstance(item2, str):
            try:
                pattern = re.compile(item2)
                if pattern.fullmatch(item1):
                    continue
            except re.error:
                pass
        if item1 != item2:
            return False 
    return True


def is_descendant(path, prefix_path):
    """
    Check if a given path is a descendant of a prefix path.
    
    Args:
        path (tuple): The tuple-based path to check.
        prefix_path (tuple): A prefix of the path ending with 'wildpath' to compare against.
    
    Returns:
        bool: True if the path is a descendant of the static path, False otherwise.
    """
    # Check if the path starts with the prefix path and has additional elements
    return path[:len(prefix_path)] == prefix_path and len(path) > len(prefix_path)


def label_paths(df, static_paths):
    """
    Label rows in the DataFrame as:
    - 0 for static paths and their descendants
    - 1 for the prefix path of static paths containing 'wildpath'

    Args:
        df (pd.DataFrame): The DataFrame containing the paths as tuples.
        static_paths (set): A set of static paths to compare against.
    """
    # Initialize the label column with default value of 1
    df["label"] = 1

    for static_path in static_paths:
        # Label the static path itself as 0
        df.loc[df["path"] == static_path, "label"] = 0

        # Check if the static path ends with 'wildpath' for labeling its prefix
        if static_path[-1] == "wildpath":
            prefix_path = static_path[:-1]

            # Label the prefix path as 1 if it exists in the DataFrame
            if prefix_path in df["path"].values:
                df.loc[df["path"] == prefix_path, "label"] = 1

            # Label all descendants of this prefix path as 0
            for index, row in df.iterrows():
                if is_descendant(row["path"], prefix_path):
                    df.at[index, "label"] = 0


def process_single_dataset(dataset, files_folder):
    """
    Process a single dataset to create and label a DataFrame.

    Args:
        dataset (str): The name of the dataset file.
        files_folder (str): The folder where the dataset files are stored.

    Returns:
        pd.DataFrame or None: A DataFrame for the dataset with labeled paths, or None if the dataset cannot be processed.
    """

    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_and_dereference_schema(schema_path)
    
    try:
        if schema:
            if has_pattern_properties(schema):
                print(f"Skipping {dataset} due to patternProperties in the schema.")
                return None
            #dynamic_paths = set(extract_implicit_schema_paths(schema))
            static_paths = set(extract_static_schema_paths(schema))
            #print(static_paths)
            
            if static_paths:
                prefix_paths_dict = defaultdict(set)
                path_types_dict = defaultdict(set)

                with open(os.path.join(files_folder, dataset), 'r') as file:
                    for line in file:
                        try:
                            doc = json.loads(line)
                            if match_properties(schema, doc):
                                process_document(doc, prefix_paths_dict, path_types_dict)
                        except Exception as e:
                            continue
                df = create_dataframe(prefix_paths_dict, dataset)
                label_paths(df, static_paths)
                #df.to_csv(f"{dataset}_data.csv", index=False)
                return df
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error processing {dataset}: {e}")
        return None


def resample_data(df):
    """
    Resample the data to balance the classes by oversampling the minority class 
    to 50% of the size of the majority class.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to resample.

    Returns:
        pd.DataFrame: The resampled DataFrame.
    """
    # Split the DataFrame into features and labels
    X = df.drop(columns=["label"]) 
    y = df["label"]

    # Initialize the oversampler
    oversample = RandomOverSampler(sampling_strategy=0.5, random_state=101)

    # Apply the oversampler
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["label"])], axis=1)
    return df_resampled


def preprocess_data(schema_list, files_folder):
    """
    Preprocess all dataset files in a folder, parallelizing the task across multiple CPUs.

    Args:
        schema_list (list): List of schema files to use for processing.
        files_folder (str): The folder containing the dataset files.

    Returns:
        pd.DataFrame: A merged DataFrame containing labeled data from all datasets.
    """
    datasets = os.listdir(files_folder)

    # Filter datasets to only include files that match those in schema_list
    datasets = [dataset for dataset in datasets if dataset in schema_list]
    ''' 
    for i, dataset in enumerate(datasets):
        if dataset != "all-contributors-configuration-file.json":#ecosystem-json.json":
            continue
        df = process_single_dataset(dataset, files_folder)
        xxx
    '''
    
    # Process each dataset in parallel
    with ProcessPoolExecutor() as executor:
        frames = list(tqdm.tqdm(executor.map(process_single_dataset, datasets, [files_folder] * len(datasets)), total=len(datasets)))

    # Filter out any datasets that returned None
    frames = [df for df in frames if df is not None]

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index=True)

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
    def __init__(self, dataframe, tokenizer, max_length=MAX_TOKEN_LEN):
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
   

def train_model(train_df, test_df):

    accumulation_steps = 4
    learning_rate = 2e-5
    num_epochs = 25

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

    early_stopper = EarlyStopper(patience=5, min_delta=0.005) # training stips if the validation loss does not decrease by at least 0.005 for 5 consecutive epochs

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
    try:
        # Parse command-line arguments
        files_folder, train_size, random_value, mode = sys.argv[-4:]
        train_ratio = float(train_size)
        random_value = int(random_value)

        # Ensure mode is valid
        if mode not in {"train", "test"}:
            raise ValueError("Invalid mode. Use 'train' or 'test'.")
        
        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)

        # Preprocess and oversample the training data
        train_df = preprocess_data(train_set, files_folder)
        #train_df = resample_data(train_df)
        train_df = train_df.sort_values(by=["filename", "path"])
        train_df.to_csv("train_data.csv", index=False)
    
        # Preprocess the testing data
        test_df = preprocess_data(test_set, files_folder)
        test_df.to_csv("test_data.csv", index=False)
        
        #train_df = pd.read_csv("train_data.csv")
        #test_df = pd.read_csv("test_data.csv")
        
        # Execute the chosen mode
        if mode == "train":
            train_model(train_df, test_df)
        else:  # mode == "test"
            evaluate_model(test_df)

    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <files_folder> <train_size> <random_value> <mode>")
        sys.exit(1)
    
 
if __name__ == "__main__":
    main()
