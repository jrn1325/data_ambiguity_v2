import json
import jsonref
import numpy as np
import os
import pandas as pd
import re
import sys
import tqdm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler  # For undersampling
from imblearn.combine import SMOTEENN, SMOTETomek  # For combining over- and under-sampling
from sklearn.model_selection import GroupShuffleSplit

import warnings
warnings.filterwarnings("ignore")

# Create constant variables
SCHEMA_KEYWORDS = ["definitions", "$defs", "properties", "additionalProperties", "patternProperties", "oneOf", "allOf", "anyOf", "items", "type", "not"]
DISTINCT_SUBKEYS_UPPER_BOUND = 1000

# Use os.path.expanduser to expand '~' to the full home directory path
SCHEMA_FOLDER = "processed_schemas"
JSON_FOLDER = "processed_jsons"


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


def process_document(doc, prefix_paths_dict, path_types_dict, parent_frequency_dict):
    """
    Extracts paths from the given JSON document and stores them in dictionaries,
    grouping paths that share the same prefix and capturing the frequency and data type of nested keys.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        prefix_paths_dict (dict): A dictionary where the keys are path prefixes 
                                  (tuples of path segments, excluding the last segment),
                                  and the values are sets of full paths that share the same prefix.
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
            # Add the path to the prefix_paths_dict
            prefix_paths_dict[prefix].add(path)

            # Initialize or update the frequency and type information
            if prefix not in path_types_dict:
                path_types_dict[prefix] = {}

            # Get the type of the current value
            value_type = get_json_format(value)

            # Update the frequency count for the nested key
            
            if nested_key in path_types_dict[prefix]:
                path_types_dict[prefix][nested_key]["frequency"] += 1
            else:
                path_types_dict[prefix][nested_key] = {"frequency": 1, "type": value_type}

            # Update parent frequency
            if prefix in parent_frequency_dict:
                parent_frequency_dict[prefix] += 1
            else:
                parent_frequency_dict[prefix] = 1


def create_dataframe(prefix_paths_dict, path_types_dict, parent_frequency_dict, dataset):
    """
    Create a DataFrame from dictionaries containing path types and path prefixes.

    Args:
        prefix_paths_dict (dict): A dictionary where keys are path prefixes and values are sets of paths.
        path_types_dict (dict): A dictionary containing frequency and type information for each nested key's value.
        parent_frequency_dict (dict): A dictionary with the frequency of parent keys.
        dataset (str): The name of the dataset.

    Returns:
        pd.DataFrame: A DataFrame with 'path', 'schema', and 'filename' columns.
    """
    data = []

    # Iterate over the paths and their associated subpaths
    for path, subpaths in prefix_paths_dict.items():
        schema_info = {}  # Collect schema information as a dictionary (JSON object)

        # Iterate over subpaths to gather key details
        for subpath in subpaths:
            nested_key = subpath[-1]  # The key at the current path level
            if nested_key in path_types_dict.get(path, {}):
                nested_key_info = path_types_dict[path][nested_key]
                frequency = nested_key_info["frequency"]
                value_type = nested_key_info["type"]

                # Calculate the relative frequency
                parent_frequency = parent_frequency_dict.get(path, 1)
                relative_frequency = frequency / parent_frequency

                # Store the key details in the schema dictionary
                schema_info[nested_key] = {
                    "relative_frequency": relative_frequency,
                    "type": value_type
                }

        # Add the parent path length to the schema info
        schema_info["nesting_depth"] = len(path)

        # Append a JSON object (dictionary) for this path
        data.append({
            "path": path,
            "schema": schema_info,  # This is a JSON object, not an array
            "filename": dataset
        })

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)

    return df


def get_object_paths(schema, parent_path=("$",)):
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
                yield from get_object_paths(prop_schema, full_path)

    # Handle arrays containing objects
    if "items" in schema and isinstance(schema["items"], dict):
        if schema["items"].get("type") == "object":
            if schema["items"].get("additionalProperties") == False:
                yield from get_object_paths(schema["items"], parent_path + ("*",))

    # Handle additionalProperties if it's a schema (a dict)
    if "additionalProperties" in schema:
        if isinstance(schema["additionalProperties"], dict):
            # We only want to traverse if additionalProperties is a schema
            if schema["additionalProperties"].get("additionalProperties") == False:
                yield from get_object_paths(schema["additionalProperties"], parent_path + ("wildpath",))

        # If additionalProperties is explicitly False, yield the path
        elif schema["additionalProperties"] == False:
            yield parent_path

    # Handle allOf, anyOf, oneOf by merging schemas
    for combiner in ["allOf", "anyOf", "oneOf"]:
        if combiner in schema:
            for sub_schema in schema[combiner]:
                yield from get_object_paths(sub_schema, parent_path)


def extract_static_schema_paths(schema, current_path=("$",)):
    """
    Extracts paths from a JSON schema where `additionalProperties` is set to False,
    ensuring all levels of the schema are traversed.
    
    Args:
        schema (dict): The JSON schema to extract paths from.
        current_path (tuple): The current path in the schema hierarchy, defaults to ("$",).
        
    Yields:
        tuple: Paths where `additionalProperties` is explicitly set to False.
    """
    stack = [(schema, current_path)]

    while stack:
        current_schema, current_path = stack.pop()

        # Ensure current_schema is a dictionary before processing
        if not isinstance(current_schema, dict):
            continue  # Skip non-dict values like True, False, etc.

        # Check if additionalProperties is explicitly False
        if current_schema.get("additionalProperties") is False:
            yield current_path

        # Process defined properties recursively
        if "properties" in current_schema:
            for key, value in current_schema["properties"].items():
                new_path = current_path + (key,)
                stack.append((value, new_path))

        # Handle arrays and check for nested objects inside arrays
        if current_schema.get("type") == "array":
            if "items" in current_schema:
                item_schema = current_schema["items"]
                stack.append((item_schema, current_path + ("*",)))

            for item in current_schema.get("prefixItems", []):
                stack.append((item, current_path + ("*",)))

        # Process composition keywords like anyOf, oneOf, allOf
        for applicator in ["anyOf", "oneOf", "allOf"]:
            for item in current_schema.get(applicator, []):
                stack.append((item, current_path))

        # Handle the 'then' keyword for conditional schemas
        if "then" in current_schema:
            stack.append((current_schema["then"], current_path))

        # Handle additionalProperties for nested schemas within additionalProperties
        if isinstance(current_schema.get("additionalProperties"), dict):
            stack.append((current_schema["additionalProperties"], current_path))


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
  

def is_descendant(path, prefix_path):
    """
    Check if a given path is a direct descendant of a prefix path.
    
    Args:
        path (tuple): The tuple-based path to check.
        prefix_path (tuple): A prefix of the path ending with 'wildpath' to compare against.
    
    Returns:
        bool: True if the path is a direct descendant of the prefix path, False otherwise.
    """
    # Check if the path starts with the prefix path and has exactly one additional element
    return path[:len(prefix_path)] == prefix_path and len(path) == len(prefix_path) + 1


def compare_tuples(tuple1, tuple2):
    """
    Compare two tuples element-wise, allowing for regular expression matching on elements in tuple2.
    If item2 is a valid regex pattern (string), it tries to match item1 with item2 as a regex.
    The 'wildpath' keyword in tuple2 matches any corresponding element in tuple1.
    
    Args:
        tuple1 (tuple): The first tuple, contains actual values.
        tuple2 (tuple): The second tuple, which may contain regex patterns, 'wildpath', or direct values.

    Returns:
        bool: True if tuples are considered equal, False otherwise.
    """
    if len(tuple1) != len(tuple2):
        return False

    for item1, item2 in zip(tuple1, tuple2):
        # Handle 'wildpath': It can match any value in tuple1
        if item2 == "wildpath":
            continue
        
        # Handle regex matching
        if isinstance(item2, str):
            try:
                # Try to compile and match item2 as a regex pattern
                pattern = re.compile(item2)
                if pattern.fullmatch(item1):
                    continue
            except re.error:
                pass  # If item2 is not a valid regex, treat it as a normal string
        
        # Direct comparison if not wildpath or regex match
        if item1 != item2:
            return False 

    return True 


def label_paths(df, static_paths):
    """
    Label rows in the DataFrame as:
    - 0 for static paths and their descendants
    - 1 for paths not matching static paths, treating 'wildpath' as a wildcard.

    Args:
        df (pd.DataFrame): The DataFrame containing the paths as tuples.
        static_paths (set): A set of static paths to compare against.
    """
    # Initialize the label column with default value of 1
    df["label"] = 1

    for static_path in static_paths:
        # Label the static path itself as 0
        df.loc[df["path"] == static_path, "label"] = 0

        # Handle paths containing 'wildpath'
        for index, row in df.iterrows():
            row_path = row["path"]
            
            # Check if the row path matches static path except for "wildpath"
            if len(row_path) == len(static_path):
                match = True
                for sp, rp in zip(static_path, row_path):
                    if sp != "wildpath" and sp != rp:
                        match = False
                        break
                if match:
                    df.at[index, "label"] = 0

    return df


def process_dataset(dataset, files_folder):
    """
    Process and extract data from the documents, and return a DataFrame.
    
    Args:
        dataset (str): The name of the dataset file.
        files_folder (str): The folder where the dataset files are stored.
    
    Returns:
        pd.DataFrame or None: A DataFrame for the dataset, or None if no data is processed.
    """
    prefix_paths_dict = defaultdict(set)
    path_types_dict = {}
    parent_frequency_dict = defaultdict(int)
    
    total_document_count = 0  
    matched_document_count = 0 
    static_schema_count = 0 

    # Load the schema
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    schema = load_schema(schema_path)

    dataset_path = os.path.join(files_folder, dataset)

    with open(dataset_path, 'r') as file:
        for line in file:
            doc = json.loads(line)
            try: 
                if isinstance(doc, dict):
                    total_document_count += 1
                    if match_properties(schema, doc):
                        matched_document_count += 1 
                        process_document(doc, prefix_paths_dict, path_types_dict, parent_frequency_dict)
                elif isinstance(doc, list):
                    for item in doc:
                        total_document_count += 1
                        if isinstance(item, dict) and match_properties(schema, item):
                            matched_document_count += 1 
                            process_document(item, prefix_paths_dict, path_types_dict, parent_frequency_dict)
            except Exception as e:
                print(f"Error processing line in {dataset}: {e}")
                continue

    if len(prefix_paths_dict) == 0:
        print(f"No paths extracted from {dataset}.")
        return None
    
    df = create_dataframe(prefix_paths_dict, path_types_dict, parent_frequency_dict, dataset)
    static_paths = set(get_object_paths(schema))
    
    if len(static_paths) == 0:
        print(f"No static paths extracted from {dataset}.")
        return None
    
    print(dataset)
    print(static_paths)
    print()
    label_paths(df, static_paths)

    static_schema_count += 1

    print(f"Total Documents in {dataset}: {total_document_count}")
    print(f"Matched Documents in {dataset}: {matched_document_count}")

    return df


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
        if dataset != "dotnetcli-host-json.json":#ecosystem-json.json":
            continue
        df = process_dataset(dataset, files_folder)
        sys.exit(0)
        if df is not None:
            if i == 0:
                frames = [df]
            else:
                frames.append(df)   
    '''
  
    # Process each dataset in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        frames = list(tqdm.tqdm(executor.map(process_dataset, datasets, [files_folder] * len(datasets)), total=len(datasets)))

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
    oversample = RandomOverSampler(sampling_strategy=0.5, random_state=101)

    # Apply the oversampler
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["label"])], axis=1)

    return df_resampled


def main():
    try:
        # Parse command-line arguments
        files_folder, train_size, random_value = sys.argv[-3:]
        train_ratio = float(train_size)
        random_value = int(random_value)
        
        # Split the data into training and testing sets
        train_set, test_set = split_data(train_ratio=train_ratio, random_value=random_value)

        # Preprocess and oversample the training data
        train_df = preprocess_data(train_set, files_folder)
        train_df = resample_data(train_df)
        train_df = train_df.sort_values(by=["filename", "path"])
        train_df.to_csv("train_data.csv", index=False)
    
        # Preprocess the testing data
        test_df = preprocess_data(test_set, files_folder)
        test_df.to_csv("test_data.csv", index=False)
    
    except (ValueError, IndexError) as e:
        print(f"Error: {e}\nUsage: script.py <files_folder> <train_size> <random_value>")
        sys.exit(1)
    
 
if __name__ == "__main__":
    main()
