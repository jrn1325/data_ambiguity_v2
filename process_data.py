import json
import jsonref
import math
import numpy as np
import os
import pandas as pd
import re
import sys
import torch
import torch.multiprocessing as mp
import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 


import warnings
warnings.filterwarnings("ignore")

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
JSON_FOLDER = "processed_jsons"
SCHEMA_FOLDER = "converted_processed_schemas"

def split_data(train_ratio, random_value):
    """
    Split the list of schemas into training and testing sets making sure that schemas from the same source are grouped together.

    Args:
        train_ratio (float, optional): The ratio of schemas to use for training.
        random_value (int, optional): The random seed value.

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
    

def load_schema(schema_path):
    """
    Load the schema from the path and resolve $ref pointers.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict or None: The loaded schema, or None if loading fails.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
            return schema
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {schema_path}: {e}", flush=True)
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}", flush=True)
    
    return None


def resolve_ref(schema, ref_path):
    """
    Resolve a $ref within the given schema.

    Args:
        schema (dict): The JSON schema.
        ref_path (str): The path to the reference.

    Returns:
        dict or None: The resolved schema, or None if the reference
                     is invalid or cannot be resolved.
    
    """
    parts = ref_path.lstrip("#/").split("/")
    ref_schema = schema
    for part in parts:
        if part in ref_schema:
            ref_schema = ref_schema[part]
        else:
            return None
    return ref_schema


def get_static_paths(schema, parent_path=("$",), is_data_level=False, root_schema=None):
    """
    Recursively traverse a JSON schema and collect full paths of properties where
    'additionalProperties' is explicitly set to False.

    Args:
        schema (dict): The JSON schema.
        parent_path (tuple): The current path accumulated as a tuple (default is the root).
        is_data_level (bool, optional): Whether we are currently in the data level.
        root_schema (dict, optional): The root schema to resolve $ref references.

    Returns:
        generator: A generator yielding full paths of object-type properties as tuples.
    """
    if root_schema is None:
        root_schema = schema

    if not isinstance(schema, dict):
        return

    # Resolve $ref if present
    if "$ref" in schema:
        ref_schema = resolve_ref(root_schema, schema["$ref"])
        if ref_schema is not None:
            yield from get_static_paths(ref_schema, parent_path, is_data_level, root_schema)
        return

    # Check 'additionalProperties' at the current level
    if "additionalProperties" in schema:
        additional_properties = schema["additionalProperties"]
        if additional_properties is False:
            yield parent_path
        elif isinstance(additional_properties, dict):
            yield from get_static_paths(additional_properties, parent_path + ("additional_key",), True, root_schema)

    # Handle 'properties'
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop, prop_schema in schema["properties"].items():
            full_path = parent_path + (prop,)
            yield from get_static_paths(prop_schema, full_path, True, root_schema)

    # Handle 'patternProperties'
    if "patternProperties" in schema and isinstance(schema["patternProperties"], dict):
        for pattern, pattern_schema in schema["patternProperties"].items():
            full_path = parent_path + ("pattern_key" + pattern,)
            yield from get_static_paths(pattern_schema, full_path, True, root_schema)

    # Handle 'items' for arrays
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            yield from get_static_paths(items, parent_path + ("*",), is_data_level, root_schema)
        elif isinstance(items, list):
            for sub_schema in items:
                yield from get_static_paths(sub_schema, parent_path + ("*",), True, root_schema)

    # Handle 'prefixItems' for arrays
    if "prefixItems" in schema:
        prefix_items = schema["prefixItems"]
        if isinstance(prefix_items, list):
            for sub_schema in prefix_items:
                yield from get_static_paths(sub_schema, parent_path + ("*",), True, root_schema)

    # Handle schema combiners: allOf, anyOf, oneOf
    for combiner in ["allOf", "anyOf", "oneOf"]:
        if combiner in schema and isinstance(schema[combiner], list):
            for sub_schema in schema[combiner]:
                yield from get_static_paths(sub_schema, parent_path, is_data_level, root_schema)

    # Handle 'if', 'then', and 'else'
    if "if" in schema and isinstance(schema["if"], dict):
        if "then" in schema and isinstance(schema["then"], dict):
            yield from get_static_paths(schema["then"], parent_path, is_data_level, root_schema)
        if "else" in schema and isinstance(schema["else"], dict):
            yield from get_static_paths(schema["else"], parent_path, is_data_level, root_schema)


def get_json_format(value):
    """
    Determine the JSON data type based on the input value's type.

    Args:
        value: The value to check.

    Returns:
        str: The corresponding JSON data type.
    """
    # Mapping from Python types to JSON types
    python_to_json_type = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        set: "array",
        dict: "object",
        type(None): "null"
    }

    # Return mapped JSON type or "object" for unknown types
    return python_to_json_type.get(type(value), "object")


def match_properties(schema, document):
    """
    Check if there is an intersection between the top-level properties in the schema and the document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is an intersection, False otherwise.
    """

    # Extract top-level schema properties
    schema_properties = set(schema.get("properties", {}).keys())

    # If there are no top-level properties, return True
    #if not schema_properties:
    #    return True

    # Extract top-level document properties
    document_properties = set(document.keys())

    # Check if there is an intersection
    return bool(schema_properties & document_properties)


def extract_paths(doc, path=("$",)):
    """
    Get the path of each key and its value from the JSON document.

    Args:
        doc (dict or list): JSON document to traverse.
        path (tuple, optional): The current path of keys. Defaults to ("$",).

    Yields:
        tuple: A tuple containing the path and the corresponding value.

    Raises:
        ValueError: If the input document is not a dict or list.
    """
    if isinstance(doc, dict):
        for key, value in doc.items():
            current_path = path + (key,)
            yield current_path, value
            if isinstance(value, (dict, list)):
                yield from extract_paths(value, current_path)

    elif isinstance(doc, list):
        for index, item in enumerate(doc):
            current_path = path + ("*",)
            yield current_path, item
            if isinstance(item, (dict, list)):
                yield from extract_paths(item, current_path)

    else:
        raise ValueError(f"Expected dict or list, got {type(doc).__name__}")


def process_document(doc, path_types_dict, parent_frequency_dict):
    """
    Extracts paths from the given JSON document and stores them in dictionaries,
    grouping paths that share the same prefix and capturing the frequency and data type of nested keys.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        path_types_dict (dict): A dictionary where the keys are path prefixes and 
                                the values are dictionaries with nested key information 
                                (frequency and data type).
        parent_frequency_dict (dict): A dictionary to track the frequency of parent keys.
    """
    
    for path, value in extract_paths(doc):
        if len(path) > 1:
            nested_key = path[-1]
            if nested_key == "*":
                continue

            prefix = path[:-1] 
            value_type = get_json_format(value)

            # Ensure path entry exists
            if prefix not in path_types_dict:
                path_types_dict[prefix] = {}

            if nested_key not in path_types_dict[prefix]:
                path_types_dict[prefix][nested_key] = {"frequency": 0, "type": set()}

            # Update frequency and type for the nested key
            path_types_dict[prefix][nested_key]["frequency"] += 1
            path_types_dict[prefix][nested_key]["type"].add(value_type)

            # Update parent frequency
            parent_frequency_dict[prefix] = parent_frequency_dict.get(prefix, 0) + 1


def get_embeddings(nested_keys):
    """
    Get the embeddings for the given nested keys using a pre-trained model.

    Args:
        nested_keys (list): A list of nested keys to get embeddings for.

    Returns:
        np.ndarray: An array of embeddings for the nested keys.
    """
    inputs = tokenizer(nested_keys, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()


def calc_semantic_similarity(nested_keys):
    """
    Calculate the average semantic similarity of nested keys based on their embeddings

    Args:
        nested_keys (dict): A dictionary where keys are nested keys and values are their types.

    Returns:
        float: The average cosine similarity between the embeddings of the nested keys.
        
    """
    if len(nested_keys) < 2:
        return 1.0
    embeddings = get_embeddings(list(nested_keys.keys()))
    similarity_matrix = cosine_similarity(embeddings)
    avg_similarity = np.mean(similarity_matrix)
    return avg_similarity


def create_dataframe(path_types_dict, parent_frequency_dict, dataset, num_docs):
    """
    Create a DataFrame from dictionaries containing path and their nested key information.

    Args:
        path_types_dict (dict): A dictionary containing frequency and type information for each nested key's value.
        parent_frequency_dict (dict): A dictionary with the frequency of parent keys.
        dataset (str): The name of the dataset.
        num_docs (int): The number of documents in the dataset.

    Returns:
        pd.DataFrame: A DataFrame with 'path', 'schema', and 'filename' columns, with additional columns for entropy.
    """
    data = []

    for path, nested_keys in path_types_dict.items():
        #if len(nested_keys) < 2:
        #    continue
        schema_info = {"properties": {}}
        values_types = set()
        frequencies = []

        parent_frequency = parent_frequency_dict.get(path, 0)

        for nested_key, nested_key_info in nested_keys.items():
            frequency = nested_key_info["frequency"] / parent_frequency if parent_frequency else 0
            value_type = nested_key_info["type"]
            values_types.add(json.dumps(list(value_type)))
            frequencies.append(frequency)

            schema_info["properties"][nested_key] = {
                "occurrence": frequency,
                "type": value_type
            }

        # Compute raw nesting depth
        raw_nesting_depth = len(path) - 1

        # Categorize nesting depth
        if raw_nesting_depth <= 3:
            schema_info["depth_level"] = "shallow"
        elif raw_nesting_depth <= 6:
            schema_info["depth_level"] = "moderate"
        else:
            schema_info["depth_level"] = "deep"

        # Type homogeneity (Homogeneous if all types are the same, Heterogeneous otherwise)
        schema_info["datatype_entropy"] = "homogeneous" if len(values_types) == 1 else "heterogeneous"
        schema_info["num_nested_keys"] = len(nested_keys)
        schema_info["semantic_similarity"] = calc_semantic_similarity(nested_keys)

        # Calculate entropy measures
        datatype_entropy = 0 if len(values_types) == 1 else 1
        key_entropy = -sum((freq / num_docs) * math.log(freq / num_docs) for freq in frequencies if freq > 0)

        # Append a JSON object (dictionary) for this path
        data.append({
            "path": path,
            "schema": schema_info,
            "datatype_entropy": datatype_entropy,
            "key_entropy": key_entropy,
            "parent_frequency": parent_frequency,
            "filename": dataset
        })

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df



def compare_paths(json_path, static_path):
    """
    Check if a schema path correctly matches a JSON path, considering
    'additional_key' and 'pattern_key'.

    Args:
        static_path (tuple): Path extracted from the schema, possibly containing 'additional_key' or 'pattern_key'.
        json_path (tuple): Path extracted from a JSON document.

    Returns:
        bool: True if the schema path matches the JSON path, considering regex patterns.
    """
    if len(static_path) != len(json_path):
        return False

    for schema_part, json_part in zip(static_path, json_path):
        if schema_part == "additional_key":  # Matches any additional key
            continue
        elif schema_part.startswith("pattern_key"):  # Matches any key that matches the regex pattern
            pattern = schema_part[11:]  # Strip 'pattern_key' prefix
            if not re.fullmatch(pattern, json_part):
                return False
        elif schema_part != json_part:  # Direct match of parts
            return False

    return True


def label_paths(df, static_paths):
    """
    Label paths in a DataFrame based on their presence in a set of static paths.

    Args:
        df (pd.DataFrame): The DataFrame containing paths to be labeled.
        static_paths (list): A list of static paths to compare against.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'label' column.
    """
    # Initialize 'label' column to 1
    df["label"] = 1
    
    # Iterate through each path in the DataFrame
    for idx, row in df.iterrows():
        json_path = row["path"]
        
        # Check if the path matches any static path
        for static_path in static_paths:
            if compare_paths(json_path, static_path):
                df.at[idx, "label"] = 0
                break 
    
    return df


def process_dataset(dataset):
    """
    Process and extract data from the documents, and return a DataFrame.
    
    Args:
        dataset (str): The name of the dataset file.
    
    Returns:
        pd.DataFrame or None: A DataFrame for the dataset, or None if no data is processed.
    """

    path_types_dict = defaultdict(lambda: defaultdict(lambda: {"frequency": 0, "type": set()}))
    parent_frequency_dict = defaultdict(int)
    
    num_docs = 0  
    matched_document_count = 0 

    # Load the schema
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_schema(schema_path)

    # Get static paths from the schema
    static_paths = set(get_static_paths(schema))
    #if len(static_paths) == 0:
    #    print(f"No static paths extracted from {dataset}.")
    #    return None

    # Load and process the dataset
    dataset_path = os.path.join(JSON_FOLDER, dataset)
    with open(dataset_path, 'r') as file:
        for line in file:
            doc = json.loads(line)
            try: 
                if isinstance(doc, dict) and match_properties(schema, doc):
                    matched_document_count += 1 
                    process_document(doc, path_types_dict, parent_frequency_dict)
                    num_docs += 1
                elif isinstance(doc, list):
                    for item in doc:
                        if isinstance(item, dict) and match_properties(schema, item):
                            matched_document_count += 1 
                            process_document(item, path_types_dict, parent_frequency_dict)
                            num_docs += 1
            except Exception as e:
                print(f"Error processing line in {dataset}: {e}")
                continue

    if len(path_types_dict) == 0:
        print(f"No paths of type object extracted from {dataset}.")
        return None
    
    df = create_dataframe(path_types_dict, parent_frequency_dict, dataset, num_docs)
    print(f"Dataset: {dataset}, Total Paths: {len(df)}, Static Paths: {len(static_paths)}", flush=True)
    df = label_paths(df, static_paths)
        
    return df


def preprocess_data(schema_list):
    """
    Preprocess all dataset files in a folder, processing them sequentially.

    Args:
        schema_list (list): List of schema files to use for processing.

    Returns:
        pd.DataFrame: A merged DataFrame containing labeled data from all datasets.
    """
    datasets = os.listdir(JSON_FOLDER)

    # Filter datasets to only include files that match those in schema_list
    datasets = [dataset for dataset in datasets if dataset in schema_list]

    # Process each dataset sequentially
    frames = []
    sys.stderr.write('Processing datasets sequentially...\n')
    
    for dataset in tqdm.tqdm(datasets, total=len(datasets)):
        df = process_dataset(dataset)  # Process each dataset sequentially
        if df is not None:
            frames.append(df)

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index=True)

    return df


def resample_data(df, random_value):
    """
    Resample the data to balance the classes by oversampling the minority class 
    to 50% of the size of the majority class.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to resample.
        random_value (int): The random seed value.

    Returns:
        pd.DataFrame: The resampled DataFrame.
    """
    # Split the DataFrame into features and labels
    X = df.drop(columns=["label"]) 
    y = df["label"]

    # Initialize the oversampler
    oversample = RandomOverSampler(sampling_strategy=0.5, random_state=random_value)

    # Apply the oversampler
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["label"])], axis=1)

    return df_resampled


def main():
    try:
        # Parse command-line arguments
        train_size, random_value = sys.argv[-2:]

    except ValueError:
        print("Usage: script.py <train_size> <random_value>")
        sys.exit(1)

    train_ratio = float(train_size)
    random_value = int(random_value)
    
    # Split the data into training and testing sets
    train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)
    
    # Preprocess and oversample the training data
    train_df = preprocess_data(train_set)
    train_df = resample_data(train_df, random_value)
    train_df.to_csv("train_data.csv", index=False, sep=";")
    
    # Preprocess the testing data
    test_df = preprocess_data(test_set)
    test_df = test_df.sample(frac=1, random_state=random_value).reset_index(drop=True)
    test_df.to_csv("test_data.csv", index=False, sep=";")
    
    
 
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()