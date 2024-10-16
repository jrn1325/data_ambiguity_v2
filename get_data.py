import json
import jsonref
import os
import shutil
import sys
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        print(f"Error loading schema {schema_path}: {e}")
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
        print(f"Error dereferencing schema {schema_path}: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing schema {schema_path}: {e}")
        return None
    except Exception as e:
        print(f"Unknown error dereferencing schema {schema_path}: {e}")
        return None

        
def has_pattern_properties_string_search(schema):
    """
    Checks if 'patternProperties' exists in the schema by converting it to a string.

    Args:
        schema (dict): The JSON schema.

    Returns:
        bool: True if 'patternProperties' is found, False otherwise.
    """
    schema_str = json.dumps(schema)
    return "patternProperties" in schema_str


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

    # Treat the schema as an object if 'type' is 'object' or if 'properties' exist
    if (schema.get("type") == "object" or "properties" in schema) and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    elif isinstance(schema.get("additionalProperties"), dict):
        prevent_additional_properties(schema["additionalProperties"])

    # Recursively handle 'properties' for object-like schemas
    if "properties" in schema:
        for value in schema["properties"].values():
            if isinstance(value, dict):
                prevent_additional_properties(value)

    # Recursively handle 'items' for array types
    if "items" in schema:
        if isinstance(schema["items"], dict):
            prevent_additional_properties(schema["items"])
        elif isinstance(schema["items"], list):
            for item in schema["items"]:
                if isinstance(item, dict):
                    prevent_additional_properties(item)

    # Handle complex schema keywords
    for keyword in ["allOf", "anyOf", "oneOf", "not", "if", "then", "else"]:
        if keyword in schema:
            if isinstance(schema[keyword], dict):
                prevent_additional_properties(schema[keyword])
            elif isinstance(schema[keyword], list):
                for subschema in schema[keyword]:
                    if isinstance(subschema, dict):
                        prevent_additional_properties(subschema)

    return schema


def validate_all_documents(dataset_path, modified_schema):
    """
    Validate all documents in the dataset against the modified schema.
    Return the list of all documents and a flag indicating whether all documents are valid.

    Args:
        dataset_path (path): The path of the dataset file.
        modified_schema (dict): The modified schema to validate against.

    Returns:
        tuple: (all_docs, invalid_docs) where:
               - all_docs (list): List of all documents (valid or invalid).
               - invalid_docs (list): List of invalid documents.
    """
    all_docs = []
    invalid_docs = []

    try:
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
                    errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
                    # If there are validation errors, add to invalid_docs and set all_valid to False
                    if errors:
                        invalid_docs.append(doc)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON document in {dataset_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading dataset {dataset_path}: {e}")
        return [], []

    return all_docs, invalid_docs


def recreate_directory(directory_path):
    # Remove the directory if it exists
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    
    # Create a fresh directory
    os.makedirs(directory_path)


def save_to_file(content, path):
    """
    Save the given content (JSON) to the specified path.

    Args:
        content (dict): The content to save.
        path (str): The file path where the content will be stored.
    """
    with open(path, 'w') as f:
        json.dump(content, f, indent=4)


def save_json_lines(json_docs, path):
    """
    Save the given JSON documents to a file, with each document on a separate line (JSON Lines format).

    Args:
        json_docs (list): A list of JSON documents (dictionaries).
        path (str): The file path where the JSON documents will be stored.
    """
    with open(path, 'a') as f:
        for doc in json_docs:
            f.write(json.dumps(doc) + '\n')


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
    """
    # Initialize failure flags
    failure_flags = {
        "exist": 0,
        "empty": 0,
        "loaded": 0,
        "pattern_properties": 0,
        "dereferenced": 0,
        "modified": 0 
    }
    
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)
    dataset_path = os.path.join(JSON_FOLDER, dataset)

    # Check if the dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset} does not exist in {JSON_FOLDER}. Skipping...")
        failure_flags["exist"] = 1  # Mark failure if dataset does not exist
        return failure_flags
    
    # Check if the dataset is empty
    if os.stat(dataset_path).st_size == 0:
        print(f"Dataset {dataset} is empty. Skipping...")
        failure_flags["empty"] = 1  # Mark failure if dataset is empty
        return failure_flags

    # Load the schema
    schema = load_schema(schema_path)
    if schema is None:
        print(f"Failed to load schema for {dataset}.")
        failure_flags["loaded"] = 1  # Mark failure if schema fails to load
        return failure_flags
    
    # Check if the schema contains patternProperties
    if has_pattern_properties_string_search(schema):
        print(f"Skipping {dataset} due to patternProperties in the schema.")
        failure_flags["pattern_properties"] = 1  # Mark failure if schema has patternProperties
        return failure_flags
    
    # Load and dereference the schema
    dereferenced_schema = load_and_dereference_schema(schema_path)
    if dereferenced_schema is None:
        print(f"Skipping {dataset} due to schema dereferencing failure.")
        failure_flags["dereferenced"] = 1  # Mark failure if dereferencing fails
        return failure_flags

    # Try modifying the schema to prevent additional properties
    try:
        modified_schema = prevent_additional_properties(dereferenced_schema)
    except Exception as e:
        print(f"Error modifying schema for {dataset}: {e}")
        failure_flags["modified"] = 1  # Mark failure if modification fails
        modified_schema = dereferenced_schema  # Use dereferenced schema if modification fails

    # Validate all documents against the modified schema
    all_docs, invalid_docs = validate_all_documents(dataset_path, modified_schema)
    print(f"Total number of documents in {dataset} is {len(all_docs)}")
    print(f"Number of invalid documents in {dataset} is {len(invalid_docs)}")

    # Save the schema only if there are valid documents
    if len(all_docs) > 0:
        # Revert to dereferenced schema if validation fails
        if invalid_docs:
            print(f"Validation failed for at least one document in {dataset}, reverting to original schema.")
            schema_to_use = dereferenced_schema
        else:
            print(f"All documents in {dataset} passed validation with modified schema.")
            schema_to_use = modified_schema
        
        # Save the schema to the processed_schemas folder
        schema_save_path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset)
        save_to_file(schema_to_use, schema_save_path)

        # Save JSON lines file with the same name as the schema
        json_save_path = os.path.join(PROCESSED_JSONS_FOLDER, dataset)
        save_json_lines(all_docs, json_save_path)
    else:
        print(f"No valid documents found for {dataset}. Skipping schema save.")

    return failure_flags


def process_datasets():
    """
    Process the datasets in parallel and save the resulting schemas and JSON files in processed folders.
    Keep track of the number of schemas successfully dereferenced, modified, those with pattern properties, 
    empty datasets, and skipped datasets. Print the remaining number of schemas after each criterion.
    """
    datasets = os.listdir(SCHEMA_FOLDER)
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
    pattern_properties_count = 0

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
                pattern_properties_count += flags["pattern_properties"]

                # Only count dereferencing and modification if the schema had valid documents
                if flags["dereferenced"] and flags["empty"] == 0:
                    dereference_count += 1
                if flags["modified"] and flags["empty"] == 0:
                    modify_count += 1

            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    # Print the count after each criterion
    print(f"Original number of datasets: {original_count}")
    print(f"Remaining after skipping non-existent datasets: {original_count - exist_count}")
    print(f"Remaining after removing empty datasets: {original_count - exist_count - empty_count}")
    print(f"Remaining after removing schemas failing to load: {original_count - exist_count - empty_count - load_count}")
    print(f"Remaining after removing schemas with patternProperties: {original_count - exist_count - empty_count - load_count - pattern_properties_count}")
    print(f"Remaining after removing schemas failing to be dereferenced: {original_count - exist_count - empty_count - load_count - pattern_properties_count - dereference_count}")
    print(f"Schemas failing to be modified: {modify_count}")


def main():
    process_datasets()

if __name__ == "__main__":
    main()