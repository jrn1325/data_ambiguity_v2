import json
import jsonref
import os
import shutil
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from jsonschema.validators import validator_for

# Use os.path.expanduser to expand '~' to the full home directory path
SCHEMA_FOLDER = os.path.expanduser("~/Desktop/schemas")
JSON_FOLDER = os.path.expanduser("~/Desktop/jsons")
PROCESSED_SCHEMAS_FOLDER = "processed_schemas"
PROCESSED_JSONS_FOLDER = "processed_jsons"


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
        print(f"Error loading schema {schema_path}: {e}", flush=True)
        return None


def load_and_dereference_schema(schema_path):
    """
    Load the JSON schema from the specified path and recursively resolve $refs within it.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict: The dereferenced JSON schema or None.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

            if "$ref" in schema:
                resolved_root = jsonref.JsonRef.replace_refs(schema)
                # Merge the resolved ref into the schema without discarding other properties
                partially_resolved_schema = {**resolved_root, **schema}
                # Remove $ref since it's now replaced with its resolution
                partially_resolved_schema.pop("$ref", None)
                resolved_schema = partially_resolved_schema
            else:
                resolved_schema = schema

            # Resolve any additional $refs that might exist after the first replacement
            resolved_schema = jsonref.JsonRef.replace_refs(resolved_schema)
            return resolved_schema

    except jsonref.JsonRefError as e:
        print(f"Error dereferencing schema {schema_path}: {e}", flush=True)
        return None
    except ValueError as e:
        print(f"Error parsing schema {schema_path}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"Unknown error dereferencing schema {schema_path}: {e}", flush=True)
        return None


def prevent_additional_properties(schema):
    """
    Recursively traverse the schema and set 'additionalProperties' to False
    if it is not explicitly declared, focusing on object-like structures.

    Args:
        schema (dict): The JSON schema to enforce the rule on.

    Returns:
        dict: The schema with 'additionalProperties' set to False where it's not declared.
    """
    if not isinstance(schema, dict):
        return schema

    # Treat the schema as an object if it has 'type: object', 'properties', or related keys
    if (
        "additionalProperties" not in schema and
        (schema.get("type") == "object" or "properties" in schema or "required" in schema)
    ):
        schema["additionalProperties"] = False

    # If additionalProperties is a schema, process it recursively
    elif isinstance(schema.get("additionalProperties"), dict):
        prevent_additional_properties(schema["additionalProperties"])

    # Recursively handle 'properties' for object-like schemas
    if "properties" in schema:
        for key, value in schema["properties"].items():
            if isinstance(value, dict):
                prevent_additional_properties(value)

    # Handle 'items' for array types
    if "items" in schema:
        if isinstance(schema["items"], dict):
            prevent_additional_properties(schema["items"])
        elif isinstance(schema["items"], list):
            for item in schema["items"]:
                if isinstance(item, dict):
                    prevent_additional_properties(item)

    # Handle 'patternProperties' for object-like schemas
    if "patternProperties" in schema:
        for pattern, pattern_schema in schema["patternProperties"].items():
            if isinstance(pattern_schema, dict):
                prevent_additional_properties(pattern_schema)

    # Handle complex schema keywords like 'allOf', 'anyOf', 'oneOf', 'if', 'then', 'else'
    for keyword in ["allOf", "anyOf", "oneOf", "if", "then", "else"]:
        if keyword in schema:
            if isinstance(schema[keyword], dict):
                prevent_additional_properties(schema[keyword])
            elif isinstance(schema[keyword], list):
                for subschema in schema[keyword]:
                    if isinstance(subschema, dict):
                        prevent_additional_properties(subschema)

    # Return the updated schema
    return schema


def validate_all_documents(dataset_path, modified_schema):
    """
    Validate all documents in the dataset against the modified schema.
    Return the list of all documents and the count of invalid documents.

    Args:
        dataset_path (str): The path of the dataset file.
        modified_schema (dict): The modified schema to validate against.

    Returns:
        tuple: (all_docs, invalid_docs_count) where:
               - all_docs (list): List of all documents (valid or invalid).
               - invalid_docs_count (int): Number of invalid documents.
    """
    all_docs = []
    invalid_docs_count = 0

    cls = validator_for(modified_schema)
    cls.check_schema(modified_schema)
    validator = cls(modified_schema)

    # Process each document in the dataset
    with open(dataset_path, 'r') as file:
        for line in file:
            try:
                doc = json.loads(line)                    
                all_docs.append(doc)

                # Validate the document against the modified schema
                errors = list(validator.iter_errors(doc))
                # Keep track of the number of invalid documents
                if errors:
                    invalid_docs_count += 1
            except Exception as e:
                print(f"Error validating document in {dataset_path}: {e}", flush=True)
                invalid_docs_count += 1
                continue
    return all_docs, invalid_docs_count


def recreate_directory(directory_path):
    # Remove the directory if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    
    # Create a fresh directory
    os.makedirs(directory_path)


def save_json_schema(content, dataset):
    """
    Save the given content (JSON) to the specified path.

    Args:
        content (dict): The content to save.
        dataset (str): The file path where the content will be stored.

    Returns:
        bool: True if the content was successfully saved, False otherwise.
    """
    path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset)

    try:
        json_content = json.dumps(content, indent=4)
    except TypeError as e:
        print(f"Error serializing JSON for {dataset}: {e}", flush=True)
        return False

    try:
        with open(path, 'w') as f:
            f.write(json_content)
        return True
    except Exception as e:
        print(f"Error saving schema for {dataset}: {e}", flush=True)
        return False


def save_json_documents(json_docs, dataset):
    """
    Save the given JSON documents to a file, with each document on a separate line (JSON Lines format).

    Args:
        json_docs (list): A list of JSON documents (dictionaries).
        dataset (str): The file path where the JSON documents will be stored.
    """
    path = os.path.join(PROCESSED_JSONS_FOLDER, dataset)
    try:
        with open(path, 'a') as f:
            for doc in json_docs:
                f.write(json.dumps(doc) + '\n')
    except Exception as e:
        print(f"Error saving documents to {path}: {e}", flush=True)


def process_single_dataset(dataset):
    """
    Process a single dataset.
    Also save the resulting schema and JSON files in processed folders.

    Args:
        dataset (str): The name of the dataset file.

    Returns:
        dict: A dictionary of flags tracking failures with keys:
            - dereferenced: 1 if the schema failed to dereference, else 0
            - modified: 1 if the schema failed to be modified, else 0
            - pattern_properties: 1 if the schema has patternProperties, else 0
            - empty: 1 if the dataset is empty, else 0
            - exist: 1 if the dataset was skipped due to not existing, else 0
            - loaded: 1 if the schema failed to load, else 0
            - validation: 1 if the schema failed to validate, else 0
    """
    # Initialize failure flags
    failure_flags = {
        "exist": 0,
        "empty": 0,
        "loaded": 0,
        "dereferenced": 0,
        "modified": 0,
        "validation": 0
    }
    
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    dataset_path = os.path.join(JSON_FOLDER, dataset)

    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset} does not exist in {JSON_FOLDER}. Skipping...", flush=True)
        failure_flags["exist"] = 1 
        return failure_flags
    
    # Check if the dataset is empty
    if os.stat(dataset_path).st_size == 0:
        print(f"Dataset {dataset} is empty. Skipping...", flush=True)
        failure_flags["empty"] = 1 
        return failure_flags

    # Load the schema
    schema = load_schema(schema_path)
    if schema is None:
        print(f"Failed to load schema for {dataset}.", flush=True)
        failure_flags["loaded"] = 1 
        return failure_flags
    
    # Load and dereference the schema
    dereferenced_schema = load_and_dereference_schema(schema_path)
    if not isinstance(dereferenced_schema, dict):
        print(f"Skipping {dataset} due to schema dereferencing failure.", flush=True)
        failure_flags["dereferenced"] = 1 
        return failure_flags
    
    # Try modifying the schema to prevent additional properties
    try:
        modified_schema = prevent_additional_properties(deepcopy(dereferenced_schema))
    except Exception as e:
        print(f"Error modifying schema for {dataset}: {e}. Reverting to dereferenced schema.", flush=True)
        failure_flags["modified"] = 1 
        modified_schema = dereferenced_schema

    # Validate documents
    all_docs, invalid_docs_count = validate_all_documents(dataset_path, modified_schema)
    if invalid_docs_count > 0:
        failure_flags["validation"] = 1
        
    print(f"{dataset}: {invalid_docs_count}/{len(all_docs)} documents failed validation.", flush=True)
    schema_to_save = modified_schema if invalid_docs_count == 0 else dereferenced_schema
    
    # Save schema and documents
    if save_json_schema(schema_to_save, dataset):
        save_json_documents(all_docs, dataset)
    
    return failure_flags
    

def process_datasets():
    """
    Process the datasets in parallel and save the resulting schemas and JSON files in processed folders.
    Keep track of the number of schemas successfully dereferenced, modified, those with pattern properties, 
    empty datasets, and skipped datasets. Print the remaining number of schemas after each criterion.
    """
    datasets = os.listdir(SCHEMA_FOLDER)
    #datasets = ["cspell.json"]
    original_count = len(datasets) 
    
    # Recreate the processed folders
    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    # Initialize counters
    exist_count = 0
    empty_count = 0
    load_count = 0 
    dereference_count = 0
    modify_count = 0
    validation_count = 0

    with ProcessPoolExecutor() as executor:
        future_to_dataset = {executor.submit(process_single_dataset, dataset): dataset for dataset in datasets}
        
        for future in tqdm.tqdm(as_completed(future_to_dataset), total=original_count):
            dataset = future_to_dataset[future]
            try:
                flags = future.result()

                # Track failures
                exist_count += flags["exist"] 
                empty_count += flags["empty"]
                load_count += flags["loaded"]
                validation_count += flags["validation"]

                # Only count dereferencing and modification if the schema had valid documents
                if flags["dereferenced"] and flags["empty"] == 0:
                    dereference_count += 1
                if flags["modified"] and flags["empty"] == 0:
                    modify_count += 1

            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    # Print the count after each criterion
    print(f"Original number of datasets: {original_count}", flush=True)
    print(f"Remaining after skipping non-existent datasets: {original_count - exist_count}", flush=True)
    print(f"Remaining after removing empty datasets: {original_count - exist_count - empty_count}", flush=True)
    print(f"Remaining after removing schemas failing to load: {original_count - exist_count - empty_count - load_count}", flush=True)
    print(f"Remaining after removing schemas failing to dereference: {original_count - exist_count - empty_count - load_count - dereference_count}", flush=True)
    print(f"Remaining after removing schemas failing to be valid: {original_count - exist_count - empty_count - load_count - dereference_count - validation_count}", flush=True)
    print(f"Schemas failing to be modified: {modify_count}", flush=True)


def main():
    process_datasets()

if __name__ == "__main__":
    main()