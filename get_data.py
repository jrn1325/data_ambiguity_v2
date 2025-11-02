import json
import jsonref
import os
import shutil
import time
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Directories
SCHEMA_FOLDER = os.path.expanduser("~/Desktop/schemas")
JSON_FOLDER = os.path.expanduser("~/Desktop/jsons")
PROCESSED_SCHEMAS_FOLDER = "processed_schemas"
PROCESSED_JSONS_FOLDER = "processed_jsons"


def load_schema(schema_path):
    """
    Load a JSON schema.

    Args:
        schema_path (str): Path to the schema file.
    Returns:
        dict: Loaded schema or None if loading fails.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading schema {schema_path}: {e}", flush=True)
        return None

def save_schema(dereferenced_schema, dataset_name):
    """
    Save a schema
    Args:
        dereferenced_schema (dict): The dereferenced schema to save.
        dataset_name (str): The name of the dataset (used for filename).
    Returns:
        bool: True if saved successfully, False otherwise.
    """
    path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset_name)

    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(dereferenced_schema, indent=2))
        return True
    except Exception as e:
        print(f"Error saving schema {dataset_name}: {e}", flush=True)
        return False
    
def deref_to_dict(obj):
    """
    Recursively convert JsonRef objects to plain Python dicts/lists.

    Args:
        obj: Any Python object (dict, list, JsonRef, etc.)

    Returns:
        obj converted to plain dict/list if needed
    """
    if isinstance(obj, jsonref.JsonRef):
        return deref_to_dict(dict(obj))
    elif isinstance(obj, dict):
        return {k: deref_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deref_to_dict(i) for i in obj]
    else:
        return obj
    
def load_and_dereference_schema(schema_path):
    """
    Load and fully dereference a JSON schema.
    Args:
        schema_path (str): Path to the schema file.
    Returns:
        dict: Dereferenced schema or None if dereferencing fails.
    """
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        # Fully replace internal $refs
        resolved_schema = jsonref.JsonRef.replace_refs(schema)

        # Convert JsonRef objects to plain dict/list
        plain_schema = deref_to_dict(resolved_schema)
        if not isinstance(plain_schema, dict):
            print(f"Schema {schema_path} resolved to non-dict type: {type(plain_schema)}")
            return None

        return plain_schema

    except Exception as e:
        print(f"Error processing schema {schema_path}: {e}", flush=True)
        return None

def recreate_directory(directory_path):
    """Remove and recreate a directory."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

def process_documents(dataset_name, dataset_path):
    """
    Process JSON documents in a dataset and save them.
    
    Args:
        dataset_name (str): Name of the dataset file.
        dataset_path (str): Path to the dataset file.
    """
    output_path = os.path.join(PROCESSED_JSONS_FOLDER, dataset_name)
    try:
        with open(dataset_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    doc = json.loads(line)

                    # Only write if JSON is an object or array
                    if isinstance(doc, (dict, list)):
                        outfile.write(json.dumps(doc) + '\n')
                    else:
                        print(f"Skipping non-object/array JSON in {dataset_name}: {doc}", flush=True)

                except json.JSONDecodeError:
                    print(f"Invalid JSON line in {dataset_name}, skipping.", flush=True)
                    
    except Exception as e:
        print(f"Error processing documents for {dataset_name}: {e}", flush=True)


def process_single_dataset(dataset_name):
    """
    Process a single dataset and its schema.
    
    Args:
        dataset_name (str): Name of the dataset file.
    Returns:
        dict: Success flags for each processing step.
    """
    success_flags = {"exist": True, "empty": True, "loaded": True, "dereferenced": True}

    schema_path = os.path.join(SCHEMA_FOLDER, dataset_name)
    dataset_path = os.path.join(JSON_FOLDER, dataset_name)

    # Check dataset existence
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} does not exist. Skipping.", flush=True)
        success_flags["exist"] = False
        return success_flags

    if os.stat(dataset_path).st_size == 0:
        print(f"Dataset {dataset_name} is empty. Skipping.", flush=True)
        success_flags["empty"] = False
        return success_flags

    # Load schema
    schema = load_schema(schema_path)
    if schema is None:
        success_flags["loaded"] = False
        return success_flags

    # Dereference schema
    dereferenced_schema = load_and_dereference_schema(schema_path)
    if not isinstance(dereferenced_schema, dict):
        success_flags["dereferenced"] = False
        return success_flags

    # Save schema
    if not save_schema(dereferenced_schema, dataset_name):
        success_flags["dereferenced"] = False
        return success_flags

    # Process JSON dataset
    process_documents(dataset_name, dataset_path)

    return success_flags

def process_datasets(max_workers=8):
    """Process all datasets concurrently and track success metrics."""
    datasets = [ds for ds in os.listdir(SCHEMA_FOLDER) if ds.endswith(".json")]
    original_count = len(datasets)

    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    # Counters
    exist_count = empty_count = load_count = dereference_count = overall_success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset = {executor.submit(process_single_dataset, ds): ds for ds in datasets}

        for future in tqdm.tqdm(as_completed(future_to_dataset), total=original_count):
            dataset = future_to_dataset[future]
            try:
                flags = future.result()
                if flags["exist"]:
                    exist_count += 1
                    if flags["empty"]:
                        empty_count += 1
                        if flags["loaded"]:
                            load_count += 1
                            if flags["dereferenced"]:
                                dereference_count += 1
                                overall_success_count += 1
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}", flush=True)

    # Percentages
    exist_pct = exist_count / original_count * 100 if original_count else 0
    empty_pct = empty_count / exist_count * 100 if exist_count else 0
    load_pct = load_count / empty_count * 100 if empty_count else 0
    deref_pct = dereference_count / load_count * 100 if load_count else 0
    overall_pct = overall_success_count / original_count * 100 if original_count else 0

    print(f"Original datasets: {original_count}")
    print(f"Existing datasets: {exist_count} ({exist_pct:.1f}%)")
    print(f"Non-empty datasets: {empty_count} ({empty_pct:.1f}%)")
    print(f"Schemas loaded: {load_count} ({load_pct:.1f}%)")
    print(f"Schemas dereferenced: {dereference_count} ({deref_pct:.1f}%)")
    print(f"Overall successful datasets: {overall_success_count} ({overall_pct:.1f}%)")


def main():
    start_time = time.time()
    process_datasets()
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
