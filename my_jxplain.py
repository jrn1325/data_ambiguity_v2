import json
import math
import numpy as np
import os
import pandas as pd
import re
import sys
import tqdm

from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sklearn.model_selection import train_test_split


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
        # Count the number of nested keys a path has
        #prefix_nested_keys_count_dict[path].append(nested_keys_count)
              

def create_dataframe(dataset_name, freq_paths_dict, prefix_paths_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict):
    """Create a dataframe

    Args:
        dataset_name (str): JSON dataset name
        freq_paths_dict (dict): dictionary of paths and their frequencies
        prefix_paths_dict (dict): dictionary of paths and their prefixes
        prefix_freqs_dict (dict): dictionary of prefixes and their frequencies
        prefix_nested_keys_freq_dict (dict): dictionary of prefixes and their nested keys

    Returns:
        2d list: dataframe
    """
    paths = []
   
    for(prefix, p_list) in prefix_paths_dict.items(): 
        prefix_freqs_dict[prefix] = [freq_paths_dict[p] for p in p_list]
    
    for (path, freq) in freq_paths_dict.items():
        paths.append(path)
    
        if len(path) > 1:
            # Get all the elements except for the last one
            prefix = path[:-1]
            prefix_nested_keys_freq_dict[prefix].append(freq)

    df = pd.DataFrame({"Path": paths})
    df["Filename"] = dataset_name
    
    return df


def populate_dataframe(df, prefix_paths_dict, prefix_freqs_dict, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs):
    """Create a dataframe of all the features

    Args:
        df (2d list): dataframe with JSON paths
        prefix_paths_dict (dict): dictionary of paths and their prefixes
        prefix_freqs_dict (dict): dictionary of paths and their frequencies
        prefix_types_dict (dict): dictionary of paths and their values's datatype
        prefix_nested_keys_freq_dict (dict): dictionary of paths and their nested keys frequencies
        num_docs (int): total number of documents in a dataset

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

    df["Datatype_entropy"] = np.nan
    df["Key_entropy"] = np.nan
        
    # Calculate the Jxplain entropy
    df = is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs)
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
    df["label"] = 0

    for key_path in dynamic_paths:
        key_path = ["$"] + key_path
        if "items" in key_path:
            continue
        #print(key_path)
        key_path = tuple([i for i in key_path if i not in keys_to_remove])
        #print(key_path)

        for index, row in df.iterrows():
            if compare_tuples(row["Path"], key_path) and row["Filename"] == dataset:
                df.at[index, "label"] = 1
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
        defaultdict(list)  # prefix_nested_keys_freq_dict
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

        with open(os.path.join(files_folder, dataset), 'r') as file:
            #lines = islice(file, 10)
            num_docs = 0
            for count, line in enumerate(file):

                json_doc = json.loads(line)
                process_document(json_doc, freq_paths_dict, prefix_paths_dict, prefix_types_dict)
                num_docs += 1

        #sys.stderr.write(f"Creating a dataframe for {dataset}\n")
        df = create_dataframe(dataset, freq_paths_dict, prefix_paths_dict, prefix_freqs_dict, prefix_nested_keys_freq_dict)
        df = populate_dataframe(df, prefix_paths_dict, prefix_freqs_dict, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs)
            
        #sys.stderr.write(f"Labeling data for {dataset}\n")
        df = label_paths(dataset, df, dynamic_paths, keys_to_remove)
        frames.append(df)

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index = True)
    
    return df


def is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs):
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



def split_data(df, test_size, random_value):
    # Split the data into training and testing sets based on filenames
    filenames = df["Filename"].tolist()
    train_filenames, test_filenames = train_test_split(filenames, test_size=test_size, random_state=random_value)
    train_df = df[df["Filename"].isin(train_filenames)]
    test_df = df[df["Filename"].isin(test_filenames)]
    return train_df, test_df



def filter_labels(true_labels, predicted_labels):
    # Get indices where true labels are 1
    positive_indices = [i for i, label in enumerate(true_labels) if label == 1]

    # Filter true labels for positive class
    true_labels_positive = [true_labels[i] for i in positive_indices]

    # Filter predicted labels for positive class
    predicted_labels_positive = [predicted_labels[i] for i in positive_indices]

    return true_labels_positive, predicted_labels_positive


def run_jxplain(test_df):
    # Perform Jxplain
    y_pred = ((test_df["Datatype_entropy"] == 0) & (test_df["Key_entropy"] > 1)).astype(int)
    y_test = test_df["label"]
    print(len(y_pred), len(y_test))

    # Calculate the precision, recall, f1-score, and support of dynamic keys
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
    print(f"precision: {precision}, recall: {recall}, F1 score: {f1_score}")

def main():
    dataset_folder, test_size = sys.argv[-2:]
    df = preprocess_data(dataset_folder)
    df = df.reset_index(drop = True)
    train_df, test_df = split_data(df, float(test_size), 101)
    run_jxplain(test_df)
    #print(df)
    #df.to_csv("data.csv")
    #df = pd.read_csv("data.csv")


 
if __name__ == '__main__':
    main()
