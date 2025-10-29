import argparse
import ast
import json
import hashlib
import os
import sys
import pandas as pd

from collections import defaultdict, Counter
from copy import deepcopy
from functools import reduce

JSON_FOLDER = "processed_jsons"

def parse_document(doc, path = ("$",), values = []):
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
            yield from parse_document(value, path + (key,), values)

def process_document(doc, paths_dict, path_freqs):
    """
    Extracts object-like paths from the given JSON document and stores them in dictionaries,
    grouping values by paths and tracking path frequency.

    Args:
        doc (dict): The JSON document from which paths are extracted.
        paths_dict (dict): Dictionary mapping each path to a list of values (as JSON strings).
        path_freqs (Counter): Dictionary tracking how often each path appears.
    """
    for path, value in parse_document(doc):
        path_freqs[path] += 1

        if path not in paths_dict:
            paths_dict[path] = []

        if isinstance(value, dict):
            value_str = json.dumps(value, sort_keys=True)
            paths_dict[path].append(value_str)

        elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
            sorted_list = sorted(
                [json.dumps(item, sort_keys=True) for item in value]
            )
            value_str = json.dumps(sorted_list, sort_keys=True)
            paths_dict[path].append(value_str)

def get_doc_hash(doc):
    """Return a hash of the canonical JSON form of the document."""
    try:
        canonical = json.dumps(doc, sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    except Exception:
        return None

def process_dataset(dataset):
    """
    Process and extract object-like paths from a JSON lines dataset, filtering documents
    by matching schema properties and ignoring duplicate documents.

    Args:
        dataset (str): The name of the dataset file.
    Returns:
        tuple: (paths_dict, num_docs)
    """
    paths_dict = defaultdict(list)
    path_freqs = Counter()
    num_docs = 0
    seen_hashes = set()

    dataset = dataset.replace("_results.csv", ".json")
    dataset_path = os.path.join(JSON_FOLDER, dataset)
    with open(dataset_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_number}] JSON decode error in {dataset}: {e}", flush=True)
                continue

            try:
                items = doc if isinstance(doc, list) else [doc]
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    doc_hash = get_doc_hash(item)
                    if doc_hash is None or doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)
                    process_document(item, paths_dict, path_freqs)
                    num_docs += 1

            except Exception as e:
                print(f"[Line {line_number}] Error processing document in {dataset}: {e}", flush=True)

    return paths_dict, num_docs, dataset

def unique_oneOf(schemas):
    """
    Ensures uniqueness in `oneOf` lists by removing duplicates.
    
    Args:
        schemas (list): List of JSON schemas to deduplicate.
    Returns:
        list: A list of unique schemas.
    """
    serialized = {json.dumps(s, sort_keys=True): s for s in schemas}
    return list(serialized.values())

def flatten_oneOf(schema):
    """
    Flattens nested `oneOf` constructs in a JSON schema.

    Args:
        schema (dict): A JSON schema.
    Returns:
        dict: A flattened JSON schema.
    """
    if "oneOf" in schema:
        result = []
        for s in schema["oneOf"]:
            if isinstance(s, dict) and "oneOf" in s:
                result.extend(flatten_oneOf(s)["oneOf"])
            else:
                result.append(s)
        return {"oneOf": unique_oneOf(result)}
    return schema

def merge_schemas(schema1, schema2):
    """
    Merges two JSON schemas into one.

    Args:
        schema1 (dict): First JSON schema.
        schema2 (dict): Second JSON schema.
    Returns:
        dict: Merged JSON schema.
    """
    if not isinstance(schema1, dict) or not isinstance(schema2, dict):
        raise TypeError("Both schema1 and schema2 must be dictionaries.")
    
    schema1 = flatten_oneOf(schema1)
    schema2 = flatten_oneOf(schema2)
    type1 = schema1.get("type")
    type2 = schema2.get("type")

    if type1 == type2:
        new_schema = deepcopy(schema1)

        if type1 == "object":
            new_schema["required"] = sorted(set(schema1.get("required", [])) | set(schema2.get("required", [])))
            for k, v in schema2.get("properties", {}).items():
                if k in new_schema["properties"]:
                    new_schema["properties"][k] = merge_schemas(new_schema["properties"][k], v)
                else:
                    new_schema["properties"][k] = v
            return new_schema

        elif type1 == "array" and "items" in schema1 and "items" in schema2:
            new_schema["items"] = merge_schemas(schema1["items"], schema2["items"])
            return new_schema

        return new_schema

    return flatten_oneOf({"oneOf": [schema1, schema2]})

def discover_schema(value):
    """
    Determine the structure (type) of the JSON key's value.

    Args:
        value: The value of the JSON key. It can be of any type.
    Returns:
        dict: An object representing the structure of the JSON key's value.
    """
    if value is None:
        return {"type": "null"}
    elif isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, list):
        item_schemas = [discover_schema(item) for item in value]
        if item_schemas:
            merged_items = reduce(merge_schemas, item_schemas)
        else:
            merged_items = {}
        return {"type": "array", "items": merged_items}
    elif isinstance(value, dict):
        schema = {"type": "object", "required": list(set(value.keys())), "properties": {}}
        for k, v in value.items():
            schema["properties"][k] = discover_schema(v)
        return schema
    else:
        raise TypeError(f"Unsupported value type: {type(value)}")

def discover_schema_from_values(values):
    """
    Infer a JSON schema from a list of JSON values.

    Args:
        values (list): List of JSON values.
    Returns:
        dict: Inferred JSON schema.
    """
    if not values:
        return {"type": "null"}
    schemas = [discover_schema(v) for v in values]
    return reduce(merge_schemas, schemas)

def normalize_dynamic_paths(dynamic_paths):
    """
    Normalize string to tuples like ('$', 'build', 'gpu').
    
    Args:
        dynamic_paths (list): List of dynamic paths as strings or tuples.
    Returns:
        list: Normalized and sorted list of dynamic paths as tuples.
    """
    normalized = []
    for p in dynamic_paths:
        if isinstance(p, tuple):
            normalized.append(p)
        elif isinstance(p, str):
            try:
                parsed = ast.literal_eval(p)
                if isinstance(parsed, tuple):
                    normalized.append(parsed)
                else:
                    raise ValueError(f"Invalid path format: {p}")
            except Exception:
                raise ValueError(f"Could not parse dynamic path: {p}")
        else:
            raise TypeError(f"Unsupported dynamic path type: {type(p)}")
    return sorted(normalized, key=len, reverse=True)

def transform_schema_with_dynamic_keys(schema, dynamic_paths):
    """
    Transform a Baazizi-style inferred schema by removing dynamic keys (bottom-up)
    and replacing them with 'additionalProperties' that contains a 'oneOf' of
    possible value structures.

    Args:
        schema (dict): Inferred schema (Baazizi-style).
        dynamic_paths (list): List of tuple-like paths (e.g. "('$', 'build', 'gpu')").

    Returns:
        dict: Transformed schema.
    """
    transformed = deepcopy(schema)

    # Normalize paths: parse string representations like "('$', 'build')" into tuples
    parsed_paths = normalize_dynamic_paths(dynamic_paths)

    for path in parsed_paths:
        keys = path[1:]  # skip '$'

        # Case 1: Root dynamic ('$') — replace all properties with oneOf
        if len(keys) == 0:
            if "properties" in transformed:
                candidates = list(transformed["properties"].values())
                transformed.pop("properties", None)
                transformed["additionalProperties"] = {"oneOf": candidates}
            continue

        # Case 2: Nested dynamic path
        parent = transformed
        for i, key in enumerate(keys):
            if "properties" in parent and key in parent["properties"]:
                # Reached the dynamic key
                if i == len(keys) - 1:
                    key_schema = parent["properties"][key]
                    del parent["properties"][key]

                    if "additionalProperties" not in parent:
                        parent["additionalProperties"] = {"oneOf": []}

                    ap = parent["additionalProperties"]
                    if "oneOf" not in ap:
                        ap["oneOf"] = []

                    # Avoid duplicates
                    serialized = json.dumps(key_schema, sort_keys=True)
                    if not any(json.dumps(s, sort_keys=True) == serialized for s in ap["oneOf"]):
                        ap["oneOf"].append(key_schema)

                else:
                    parent = parent["properties"][key]
            elif parent.get("type") == "array" and "items" in parent:
                parent = parent["items"]
            else:
                break

    return transformed

def get_schema_size(schema):
    """
    Get the size of a JSON schema in bytes without whitespace.

    Args:
        schema (dict): JSON schema.

    Returns:
        int: Size of the schema in bytes.
    """
    schema_str = json.dumps(schema, separators=(",", ":"))
    return len(schema_str.encode("utf-8")) 

def compare_schema_sizes(schema1, schema2):
    """
    Compare two JSON schemas for size (bytes).

    Args:
        schema1 (dict): First JSON schema.
        schema2 (dict): Second JSON schema.
    Returns:
        tuple: (size1, size2, size_difference)
    """
    size1 = get_schema_size(schema1)
    size2 = get_schema_size(schema2)
    return size1, size2, size1 - size2


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--input", type=str, help="Directory with input JSON files")
    #parser.add_argument("--output", type=str, help="Directory for ground truth CSVs")
    #args = parser.parse_args()

    #json_dir = args.input
    groundtruth_dir = "evaluation_results"
    inferred_schemas = "inferred_schemas"
    transformed_schemas = "transformed_schemas"

    # Create output directories if they don't exist
    os.makedirs(inferred_schemas, exist_ok=True)
    os.makedirs(transformed_schemas, exist_ok=True)

    for file in os.listdir(groundtruth_dir): 
        paths_dict, _, dataset = process_dataset(file)
        dataset = dataset.replace(".json", "_results.csv")
        dataset_path = os.path.join(groundtruth_dir, dataset)
        groundtruth = pd.read_csv(dataset_path, sep=";")
        file = file.replace("_results.csv", ".json")

        # Initialize schema from all paths
        all_json_values = []
        for values in paths_dict.values():
            all_json_values.extend([json.loads(v) for v in values])
        schema = discover_schema_from_values(all_json_values)
        print(f"Initial schema for {file} inferred.", flush=True)

        # Extract dynamic paths and sort bottom-up
        dynamic_paths = groundtruth[(groundtruth["label"] == 1) & (groundtruth["correct"] == True)]["path"].tolist()
        dynamic_paths = sorted(dynamic_paths, key=len, reverse=True)
        print(f"Dynamic paths for {file} extracted.", flush=True)

        # Transform schema
        transformed_schema = transform_schema_with_dynamic_keys(schema, dynamic_paths)
        print(f"Transformed schema for {file} created.", flush=True)

        # Compare schema sizes
        schema_size, transformed_size, size_diff = compare_schema_sizes(schema, transformed_schema)
        print(f"Dataset: {file}, Original Schema Size: {schema_size} bytes, Transformed Schema Size: {transformed_size} bytes, Size Difference: {size_diff} bytes", flush=True)
        
        # Store schemas
        inferred_schema_path = os.path.join(inferred_schemas, file)
        transformed_schema_path = os.path.join(transformed_schemas, file)

        with open(inferred_schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        with open(transformed_schema_path, "w") as f:
            json.dump(transformed_schema, f, indent=2)

        
if __name__ == "__main__":
    main()
