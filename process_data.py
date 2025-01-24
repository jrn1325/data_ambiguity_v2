import json
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


def split_data(train_ratio=0.8, random_value=42):
    """
    Split the list of schemas into training and testing sets making sure that schemas from the same source are grouped together.

    Args:
        train_ratio (float, optional): The ratio of schemas to use for training. Defaults to 0.8.
        random_value (int, optional): The random seed value. Defaults to 42.

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
    Load the schema from the path.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict or None: The loaded schema, or None if loading fails.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
            return schema
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}")
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


def get_static_paths(schema, parent_path=("$",)):
    """
    Recursively traverse a JSON schema and collect full paths of properties where type is 'object',
    returning the paths as tuples and excluding paths for dynamic nested keys.

    Args:
    schema (dict): The JSON schema.
    parent_path (tuple): The current path accumulated as a tuple (default is the root).

    Returns:
        A generator yielding full paths of object-type properties as tuples
    """
    
    # Check if 'properties' exist in the schema
    if "properties" in schema:
        for prop, prop_schema in schema["properties"].items():
            full_path = parent_path + (prop,)

            if isinstance(prop_schema, dict):
                # Only yield the path if additionalProperties is explicitly set to False
                if prop_schema.get("additionalProperties") == False:
                    yield full_path  # Yield the path for well-defined properties only
                
                # Recursively check nested properties
                yield from get_static_paths(prop_schema, full_path)

    # Handle arrays containing objects
    if "items" in schema and isinstance(schema["items"], dict):
        if schema["items"].get("type") == "object":
            if schema["items"].get("additionalProperties") == False:
                yield from get_static_paths(schema["items"], parent_path + ("*",))

    # Handle additionalProperties if it's a schema (a dict)
    if "additionalProperties" in schema:
        if isinstance(schema["additionalProperties"], dict):
            # We only want to traverse if additionalProperties is a schema
            if schema["additionalProperties"].get("additionalProperties") == False:
                yield from get_static_paths(schema["additionalProperties"], parent_path + ("wildkey",))

        # If additionalProperties is explicitly False, yield the path
        elif schema["additionalProperties"] == False:
            yield parent_path

    # Handle allOf, anyOf, oneOf by merging schemas
    for combiner in ["allOf", "anyOf", "oneOf"]:
        if combiner in schema:
            for sub_schema in schema[combiner]:
                yield from get_static_paths(sub_schema, parent_path)

    # Handle patternProperties
    if "patternProperties" in schema:
        for pattern, pattern_schema in schema["patternProperties"].items():
            full_path = parent_path + (pattern,)

            if isinstance(pattern_schema, dict):
                 # If additionalProperties is explicitly False, yield the path
                if pattern_schema.get("additionalProperties") == False:
                    yield full_path 
                
                # Recursively check nested properties
                yield from get_static_paths(pattern_schema, full_path)
    
    # Handle if-then-else
    if "if" in schema and "then" in schema:
        yield from get_static_paths(schema["then"], parent_path)

    if "else" in schema:
        yield from get_static_paths(schema["else"], parent_path)


def is_path_match(row_path, static_path, wildkey):
    """
    Check if a given row path matches a static path, considering wildkey and regex patterns.

    Args:
        row_path (tuple): The path from the DataFrame row.
        static_path (tuple): The static path to match against.
        wildkey (str): The wildkey value used in static paths, which matches any segment.

    Returns:
        bool: True if the row_path matches the static_path, False otherwise.
    """
    if len(row_path) != len(static_path):
        return False

    # Check for exact match first
    if row_path == static_path:
        return True

    # Compare each element of the static path with the corresponding element in row path
    for sp, rp in zip(static_path, row_path):

        if sp == wildkey:  # Wildkey matches any segment
            continue

        # Check if sp (static path element) is a regex pattern
        if isinstance(sp, str):
            try:
                # Try regex match
                if re.fullmatch(sp, rp):
                    continue  # If regex matches, proceed to the next pair
                else:
                    return False  # If regex doesn't match, return False
            except re.error:
                # If sp is not a valid regex pattern, compare directly
                if sp != rp:
                    return False
        else:
            # Direct comparison for non-string elements (e.g., wildkey or other types)
            if sp != rp:
                return False
    return True


def label_paths(df, static_paths, wildkey="wildkey"):
    """
    Label rows in the DataFrame as:
    - 0 for static paths and their descendants, including regular expressions.
    - 1 for paths not matching static paths, treating 'wildkey' as a wildcard.

    Args:
        df (pd.DataFrame): The DataFrame containing the paths as tuples.
        static_paths (set): A set of static paths to compare against, potentially including regex patterns.
        wildkey (str): The wildcard value used in the static paths (default is 'wildkey').

    Returns:
        pd.DataFrame: The DataFrame with an additional 'label' column.
    """
    # Initialize the label column with default value of 1
    df["label"] = 1

    # Check each row's path and label it based on matching static paths
    for index, row in df.iterrows():
        for static_path in static_paths:
            if is_path_match(row["path"], static_path, wildkey):
                df.at[index, "label"] = 0  # Mark as matching static path
                break  # No need to check other static paths once a match is found

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

    dataset_path = os.path.join(JSON_FOLDER, dataset)

    with open(dataset_path, 'r') as file:
        for line in file:
            doc = json.loads(line)
            try: 
                if isinstance(doc, dict):
                    num_docs += 1
                    if match_properties(schema, doc):
                        matched_document_count += 1 
                        process_document(doc, path_types_dict, parent_frequency_dict)
                        num_docs += 1
                elif isinstance(doc, list):
                    for item in doc:
                        num_docs += 1
                        if isinstance(item, dict) and match_properties(schema, item):
                            matched_document_count += 1 
                            process_document(item, path_types_dict, parent_frequency_dict)
                            num_docs += 1
            except Exception as e:
                print(f"Error processing line in {dataset}: {e}")
                continue

    if len(path_types_dict) == 0:
        print(f"No paths extracted from {dataset}.")
        return None
    
    df = create_dataframe(path_types_dict, parent_frequency_dict, dataset, num_docs)
    static_paths = set(get_static_paths(schema))
    
    if len(static_paths) == 0:
        print(f"No static paths extracted from {dataset}.")
        return None
    
    print(f"Dataset: {dataset}, Total Paths: {len(df)}, Static Paths: {len(static_paths)}", flush=True)
    label_paths(df, static_paths)

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
    oversample = RandomOverSampler(sampling_strategy=0.5, random_state=42)

    # Apply the oversampler
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["label"])], axis=1)

    return df_resampled


def main():
    try:
        # Parse command-line arguments
        train_size, random_value = sys.argv[-2:]
        train_ratio = float(train_size)
        random_value = int(random_value)
        
        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)
        
        # Preprocess and oversample the training data
        train_df = preprocess_data(train_set)
        train_df = resample_data(train_df)
        train_df = train_df.sort_values(by=["filename", "path"])
        train_df.to_csv("train_data.csv", index=False)
        
        # Preprocess the testing data
        test_df = preprocess_data(test_set)
        test_df.to_csv("test_data.csv", index=False)
    
    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <train_size> <random_value>")
        sys.exit(1)
    
 
if __name__ == "__main__":
    main()