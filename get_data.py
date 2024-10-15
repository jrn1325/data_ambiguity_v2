import json
import jsonref
import os
import shutil
import sys
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from jsonref import replace_refs
from jsonschema import RefResolver
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
        int: Status code indicating success (1) or failure (0).
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

            return resolved_schema, 1

    except jsonref.JsonRefError as e:
        print(f"Error dereferencing schema {schema_path}: {e}")
        return None, 0
    except ValueError as e:
        print(f"Error parsing schema {schema_path}: {e}")
        return None, 0
    except Exception as e:
        print(f"Unknown error dereferencing schema {schema_path}: {e}")
        return None, 0

        
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


def validate_all_documents(dataset, files_folder, modified_schema):
    """
    Validate all documents in the dataset against the modified schema.
    Return the list of all documents and a flag indicating whether all documents are valid.

    Args:
        dataset (str): The name of the dataset file.
        files_folder (str): The folder where the dataset files are stored.
        modified_schema (dict): The modified schema to validate against.

    Returns:
        tuple: (all_docs, invalid_docs) where:
               - all_docs (list): List of all documents (valid or invalid).
               - invalid_docs (list): List of invalid documents.
    """
    dataset_path = os.path.join(files_folder, dataset)
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
                    print(f"Error parsing JSON document in {dataset}: {e}")
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


def process_single_dataset(dataset, files_folder):
    """
    Process a single dataset to create and label a DataFrame.
    Also save the resulting schema and JSON files in processed folders.

    Args:
        dataset (str): The name of the dataset file.
        files_folder (str): The folder where the dataset files are stored.

    Returns:
        tuple: (1 if schema was dereferenced, 1 if schema was modified, 1 if schema has patternProperties) otherwise 0s.
    """
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)

    # Load the schema
    schema = load_schema(schema_path)
    if schema is None:
        return (0, 0, 0)
    
    # Check if the schema contains patternProperties
    if has_pattern_properties_string_search(schema):
        print(f"Skipping {dataset} due to patternProperties in the schema.")
        return (0, 0, 1)
    
    # Load and dereference the schema if needed
    dereferenced_schema, dereferenced_flag = load_and_dereference_schema(schema_path)
    
    # If dereferencing fails, return the appropriate flags
    if dereferenced_schema is None:
        print(f"Skipping {dataset} due to schema dereferencing failure.")
        return (dereferenced_flag, 0, 0)

    # Try modifying the schema to prevent additional properties
    try:
        modified_schema = prevent_additional_properties(dereferenced_schema)
        modified_flag = 1
    except Exception as e:
        print(f"Error modifying schema for {dataset}: {e}")
        modified_schema = deepcopy(dereferenced_schema)
        modified_flag = 0
    
    # Validate all documents against the modified schema
    all_docs, invalid_docs = validate_all_documents(dataset, files_folder, modified_schema)
    print(f"Total number of documents in {dataset} is {len(all_docs)}")
    print(f"Number of invalid documents in {dataset} is {len(invalid_docs)}")

    if invalid_docs:
        print(f"Validation failed for at least one document in {dataset}, reverting to original schema.")
        schema_to_use = deepcopy(modified_schema)
    else:
        print(f"All documents in {dataset} passed validation with modified schema.")
        schema_to_use = deepcopy(modified_schema)
        
    if not all_docs:
        print(f"No valid documents found in {dataset}. Skipping schema and JSON creation.")
        return (dereferenced_flag, modified_flag, 0)
    
    # Save the schema to the processed_schemas folder
    schema_save_path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset)
    save_to_file(schema_to_use, schema_save_path)

    # Save JSON Lines file with the same name as the schema
    json_save_path = os.path.join(PROCESSED_JSONS_FOLDER, dataset)
    save_json_lines(all_docs, json_save_path)

    return (dereferenced_flag, modified_flag, 0)  # Return success flags and patternProperties flag


def process_datasets():
    """
    Process the datasets in parallel and save the resulting schemas and JSON files in processed folders.
    Also keep track of the number of schemas successfully dereferenced, modified, and those with pattern properties.
    """
    datasets = os.listdir(SCHEMA_FOLDER)
    
    # Recreate the processed folders
    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    # Initialize counters for dereferenced, modified schemas, and pattern properties
    dereference_count = 0
    modify_count = 0
    pattern_properties_count = 0
    '''
    for dataset in datasets:
        if dataset != "sourcery.json":
            continue
        print(f"Processing dataset {dataset}...")
        process_single_dataset(dataset, JSON_FOLDER)
    sys.exit(0)
    '''
    with ProcessPoolExecutor() as executor:
        future_to_dataset = {executor.submit(process_single_dataset, dataset, JSON_FOLDER): dataset for dataset in datasets}
        
        for future in tqdm.tqdm(as_completed(future_to_dataset), total=len(datasets)):
            dataset = future_to_dataset[future]
            try:
                deref_flag, mod_flag, pattern_flag = future.result()
                dereference_count += deref_flag
                modify_count += mod_flag
                pattern_properties_count += pattern_flag
            except Exception as e:
                print(f"Error processing dataset {dataset}: {e}")

    print(f"Total successfully dereferenced schemas: {dereference_count}")
    print(f"Total successfully modified schemas: {modify_count}")
    print(f"Total schemas with patternProperties: {pattern_properties_count}")

def main():
    process_datasets()

if __name__ == "__main__":
    main()