import json
import jsonref
import math
import numpy as np
import os
import pandas as pd
import re
import sys
import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GroupShuffleSplit

import warnings
warnings.filterwarnings("ignore")

# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
JSON_FOLDER = "processed_jsons"
SCHEMA_FOLDER = "processed_schemas"

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
            # Load the schema from file
            schema = json.load(schema_file)
            return schema
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {schema_path}: {e}", flush=True)
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}", flush=True)
    
    return None


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
    """Check if there is an intersection between the schema and the document.

    Args:
        schema (dict): JSON Schema object.
        document (dict): JSON object.

    Returns:
        bool: True if there is a match, False otherwise.
    """

    # Check top-level properties
    schema_properties = schema.get("properties", {})
    if any(key in schema_properties for key in document):
        return True

    # Check patternProperties
    pattern_properties = schema.get("patternProperties", {})
    for pattern in pattern_properties:
        regex = re.compile(pattern)
        if any(regex.match(key) for key in document):
            return True
    
    # Check conditional subschemas
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword in schema:
            subschemas = schema[keyword]
            for subschema in subschemas:
                if match_properties(subschema, document):
                    return True
    
    return False


def get_json_format(value):
    """
    Determine the JSON format type based on the input value's type.

    Args:
        value: The value to check.

    Returns:
        str: The corresponding JSON format type.
    """
    # Mapping from Python types to JSON types
    python_to_json_type = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "NoneType": "null"
    }

    # Get the value's type name and map it to the JSON format type
    value_type = type(value).__name__
    return python_to_json_type.get(value_type, "unknown") 


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

            # Update frequency and type for the nested key
            path_types_dict[prefix][nested_key]["frequency"] += 1
            path_types_dict[prefix][nested_key]["type"].add(value_type)

            # Update parent frequency
            parent_frequency_dict[prefix] += 1


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

    # Iterate over the paths and their associated nested keys
    for path, nested_keys in path_types_dict.items():
        schema_info = {"properties": {}}
   
        values_types = set()
        frequencies = []

        # Iterate over subpaths to gather key details
        for nested_key, nested_key_info in nested_keys.items():
            frequency = nested_key_info["frequency"]
            value_type = nested_key_info["type"]
            values_types.add(json.dumps(list(value_type)))
            frequencies.append(frequency)

            # Calculate the relative frequency
            parent_frequency = parent_frequency_dict.get(path)
            relative_frequency = frequency / parent_frequency

            # Store the key details in the schema dictionary
            schema_info["properties"][nested_key] = {
                "relative_frequency": relative_frequency,
                "type": value_type
            }

        # Add the parent path length to the schema info
        schema_info["nesting_depth"] = len(path)
        # Calculate type homogeneity (True if all types are the same, False otherwise)
        schema_info["type_homogeneity"] = len(values_types) == 1
        schema_info["num_nested_keys"] = len(nested_keys)
        # Calculate datatype and key entropy
        datatype_entropy = 0 if len(values_types) == 1 else 1
        key_entropy = -sum((freq / num_docs) * math.log(freq / num_docs) for freq in frequencies)

        # Append a JSON object (dictionary) for this path
        data.append({
            "path": path,
            "schema": schema_info,
            "datatype_entropy": datatype_entropy,
            "key_entropy": key_entropy,
            "filename": dataset
        })

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df


def get_static_paths(schema, parent_path=("$",), is_data_level=False):
    """
    Recursively traverse a JSON schema and collect full paths of properties where
    'additionalProperties' is explicitly set to False.

    Args:
        schema (dict): The JSON schema.
        parent_path (tuple): The current path accumulated as a tuple (default is the root).
        is_data_level (bool, optional): Whether we are currently in the data level. Defaults to False.

    Returns:
        generator: A generator yielding full paths of object-type properties as tuples.
    """
    if not isinstance(schema, dict):
        print(f"Skipping invalid schema at path {parent_path}: Expected a dictionary, got {type(schema).__name__}")
        return

    # Check 'additionalProperties' at the current level
    if "additionalProperties" in schema:
        additional_properties = schema["additionalProperties"]
        if additional_properties is False:
            yield parent_path  # Disallowing extra properties at this level
        elif isinstance(additional_properties, dict):
            # Recursively traverse the schema under additionalProperties
            yield from get_static_paths(additional_properties, parent_path + ("additional_key",), True)

    # Handle 'properties'
    if "properties" in schema and isinstance(schema["properties"], dict):
        for prop, prop_schema in schema["properties"].items():
            full_path = parent_path + (prop,)
            yield from get_static_paths(prop_schema, full_path, True)

    # Handle 'patternProperties'
    if "patternProperties" in schema and isinstance(schema["patternProperties"], dict):
        for pattern, pattern_schema in schema["patternProperties"].items():
            full_path = parent_path + ("pattern_key", pattern, True)
            yield from get_static_paths(pattern_schema, full_path)

    # Handle 'items' for arrays
    if "items" in schema:
        items = schema["items"]
        if isinstance(items, dict):
            yield from get_static_paths(items, parent_path + ("*",))
        elif isinstance(items, list):
            for sub_schema in items:
                yield from get_static_paths(sub_schema, parent_path + ("*",), True)
    # Handle 'prefixItems' for arrays
    if "prefixItems" in schema:
        prefix_items = schema["prefixItems"]
        if isinstance(prefix_items, list):
            for sub_schema in prefix_items:
                yield from get_static_paths(sub_schema, parent_path + ("*",), True)
    
    # Handle schema combiners: allOf, anyOf, oneOf
    for combiner in ["allOf", "anyOf", "oneOf"]:
        if combiner in schema and isinstance(schema[combiner], list):
            for sub_schema in schema[combiner]:
                yield from get_static_paths(sub_schema, parent_path, is_data_level)

    # Handle 'if', 'then', and 'else'
    if "if" in schema and isinstance(schema["if"], dict):
        if "then" in schema and isinstance(schema["then"], dict):
            yield from get_static_paths(schema["then"], parent_path, is_data_level)
        if "else" in schema and isinstance(schema["else"], dict):
            yield from get_static_paths(schema["else"], parent_path, is_data_level)


def compare_paths(json_path, static_path):
    """
    Compare a JSON path with a static path. 'pattern_key' indicates the next element is a regex.

    Args:
        json_path (tuple): The JSON path to compare.
        static_path (tuple): The static path to compare against.

    Returns:
        bool: True if the paths match, False otherwise.
    """
    json_idx, static_idx = 0, 0
    json_len, static_len = len(json_path), len(static_path)

    while json_idx < json_len and static_idx < static_len:
        s_key = static_path[static_idx]

        # Handle 'additional_key' (wildcard at this position)
        if s_key == "additional_key":
            static_idx += 1  # Move to next static path key
            json_idx += 1  # Move to next JSON path key
            continue

        # Handle 'pattern_key' (regex check for the next element)
        elif s_key == "pattern_key":
            # Ensure the regex pattern exists in static_path
            if static_idx + 1 >= static_len or json_idx >= json_len:
                return False  # Invalid pattern usage

            pattern = static_path[static_idx + 1]  # Regex pattern
            if not re.fullmatch(pattern, str(json_path[json_idx])):
                return False  # Pattern did not match

            static_idx += 2  # Move past 'pattern_key' and regex pattern
            json_idx += 1  # Move past the matched JSON key
            continue

        # Handle normal key comparison
        elif json_path[json_idx] != s_key:
            return False

        # Move to the next key in both paths
        json_idx += 1
        static_idx += 1

    # Ensure both paths have been fully processed
    return json_idx == json_len and static_idx == static_len



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
    if len(static_paths) == 0:
        print(f"No static paths extracted from {dataset}.")
        return None

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
    Preprocess all dataset files in a folder, parallelizing the task across multiple CPUs.

    Args:
        schema_list (list): List of schema files to use for processing.

    Returns:
        pd.DataFrame: A merged DataFrame containing labeled data from all datasets.
    """
    datasets = os.listdir(JSON_FOLDER)

    # Filter datasets to only include files that match those in schema_list
    datasets = [dataset for dataset in datasets if dataset in schema_list]
  
    # Process each dataset in parallel
    with ProcessPoolExecutor() as executor:
        frames = list(tqdm.tqdm(executor.map(process_dataset, datasets), total=len(datasets)))

    # Filter out any datasets that returned None
    frames = [df for df in frames if df is not None]

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
    #train_df = train_df.sort_values(by=["filename", "path"])
    train_df.to_csv("train_data.csv", index=False)
    
    # Preprocess the testing data
    test_df = preprocess_data(test_set)
    #test_df = test_df.sample(frac=1, random_state=random_value).reset_index(drop=True)
    test_df.to_csv("test_data.csv", index=False)
    
    
 
if __name__ == "__main__":
    main()