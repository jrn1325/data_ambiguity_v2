import argparse
import ast
import json
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from tqdm import tqdm

ARRAY_WILDCARD = "<ARRAY_ITEM>"

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

def normalize_additional_properties(parent):
    """Ensure parent['additionalProperties'] is a dict and return it."""
    
    ap = parent.get("additionalProperties")

    # Treat None, True, and False as empty dicts
    if ap in (None, True, False):
        ap = {}
        parent["additionalProperties"] = ap
        return ap

    # If already a dict, ensure it's stored and return it
    if isinstance(ap, dict):
        parent["additionalProperties"] = ap
        return ap

    # Unsupported type
    raise TypeError(f"Unexpected additionalProperties type: {type(ap)}")


def merge_dynamic_schema(ap, dynamic_schema):
    """
    Merge a dynamic schema into an existing additionalProperties.

    Args:
        ap (dict): existing additionalProperties schema.
        dynamic_schema (dict): dynamic schema to merge.
    """
    if not ap:
        ap.update(dynamic_schema)
    else:
        if "anyOf" not in ap:
            existing = deepcopy(ap)
            ap.clear()
            ap["anyOf"] = [existing]

        serialized = json.dumps(dynamic_schema, sort_keys=True)
        if not any(json.dumps(s, sort_keys=True) == serialized for s in ap["anyOf"]):
            ap["anyOf"].append(dynamic_schema)

        if len(ap["anyOf"]) == 1:
            only_schema = ap["anyOf"][0]
            ap.clear()
            ap.update(only_schema)

def process_path(schema, keys):
    """
    Traverse the schema following the provided path, removing the dynamic key.

    Args:
        schema (dict): JSON schema.
        keys (list): List of keys representing the path.    
    """
    if not keys:
        return

    key = keys[0]
    remaining = keys[1:]

    # Object case
    if schema.get("type") == "object" and "properties" in schema:
        props = schema["properties"]
        if key in props:
            if not remaining:
                # Dynamic key found
                dynamic_schema = props.pop(key)

                if "required" in schema and key in schema["required"]:
                    schema["required"].remove(key)
                    # Remove empty required lists (optional)
                    if len(schema["required"]) == 0:
                        del schema["required"]

                ap = normalize_additional_properties(schema)
                merge_dynamic_schema(ap, dynamic_schema)
            else:
                process_path(props[key], remaining)

    # Array case
    elif schema.get("type") == "array" and "items" in schema:
        if key == ARRAY_WILDCARD:
            # Wildcard: apply to the array's items
            process_path(schema["items"], remaining)
        else:
            # Rare case: explicit numeric index (if ever appears)
            process_path(schema["items"], keys)

    # Handle schema combinators (anyOf, oneOf, allOf)
    for combiner in ("anyOf", "oneOf", "allOf"):
        if combiner in schema:
            for sub_schema in schema[combiner]:
                process_path(sub_schema, keys)

def transform_schema_with_dynamic_keys(schema, dynamic_paths):
    """
    Transform a Baazizi-style inferred schema by removing dynamic keys recursively,
    merging their structures into `additionalProperties` while preserving constraints.
    Handles schemas that start with anyOf/oneOf/allOf.

    Args:
        schema (dict): JSON schema.
        dynamic_paths (list): List of dynamic paths as tuples.
    Returns:
        dict: Transformed JSON schema.
    """
    transformed = deepcopy(schema)
    parsed_paths = normalize_dynamic_paths(dynamic_paths)

    for path in parsed_paths:
        keys = path[1:]  # skip '$'
        process_path(transformed, keys)

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

def load_schema(path):
    """
    Load a JSON schema from a file.

    Args:
        path (str): File path to load the schema from.
    Returns:
        dict: JSON schema.
    """
    with open(path, "r") as f:
        schema = json.load(f)
    return schema

def save_schema(schema, path):
    """
    Save a JSON schema to a file.

    Args:
        schema (dict): JSON schema.
        path (str): File path to save the schema.
    """
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)

def process_single_dataset(dataset, inferred_schemas_dir, groundtruth_dir, mode):
    """Process datasets and save transformed schemas."""
    

    # Create output directory if it doesn't exist
    transformed_schemas_dir = f"{inferred_schemas_dir}_{mode}"
    os.makedirs(transformed_schemas_dir, exist_ok=True)

    # Skip if already processed
    transformed_schema_path = os.path.join(transformed_schemas_dir, dataset)
    if os.path.exists(transformed_schema_path):
        return f"Skipping {dataset} — already processed.", None

    try:
        # Load inferred schema
        inferred_schema_path = os.path.join(inferred_schemas_dir, dataset)
        inferred_schema = load_schema(inferred_schema_path)

        # Load groundtruth data
        groundtruth_path = os.path.join(groundtruth_dir + "_" + mode, dataset.replace(".json", "_results.csv"))
        groundtruth = pd.read_csv(groundtruth_path, sep=";")

        # Extract dynamic paths and sort bottom-up
        dynamic_paths = groundtruth[(groundtruth["pred"] == 1)]["path"].tolist()
        dynamic_paths = normalize_dynamic_paths(dynamic_paths)
        print(f"Processing {dataset}: Found {len(dynamic_paths)} dynamic paths.", flush=True)

        # Transform schema
        transformed_schema = transform_schema_with_dynamic_keys(inferred_schema, dynamic_paths)

        # Compare schema sizes
        schema_size, transformed_size, size_diff = compare_schema_sizes(inferred_schema, transformed_schema)
        pct_diff = size_diff / schema_size * 100 if schema_size else 0

        # Save schema
        save_schema(transformed_schema, transformed_schema_path)

        msg = (
            f"{dataset}: Original={schema_size} bytes, Transformed={transformed_size} bytes, "
            f"Diff={size_diff} bytes ({pct_diff:.2f}%)"
        )
        return msg, True

    except Exception as e:
        return f"Error processing {dataset}: {e}", False

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("inferred_schemas", type=str, help="Directory for inferred schemas")
    parser.add_argument("eval_input", type=str, help="Directory for ground truth CSVs")
    parser.add_argument("mode", type=str, help="Mode: adapter, full, jxplain")
    args = parser.parse_args()

    files = [f for f in os.listdir(args.inferred_schemas)]

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(
                process_single_dataset, f, args.inferred_schemas, args.eval_input, args.mode): f
            for f in files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Transforming schemas in {args.mode} mode"):
            msg, success = future.result()
            print(msg, flush=True)
            results.append((msg, success))

    end_time = time.time()
    total_success = sum(1 for _, s in results if s)
    total_errors = sum(1 for _, s in results if s is False)
    total_skipped = sum(1 for _, s in results if s is None)

    print("\n--- SUMMARY ---")
    print(f"Processed: {total_success}")
    print(f"Skipped:   {total_skipped}")
    print(f"Errors:    {total_errors}")
    print(f"Total time: {end_time - start_time:.2f} sec", flush=True)
        
if __name__ == "__main__":
    main()