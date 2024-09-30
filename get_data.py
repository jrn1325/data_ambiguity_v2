import json
import jsonref
import os
import shutil
import sys
import tqdm
from concurrent.futures import ProcessPoolExecutor
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
    Load the JSON schema from the specified path and dereference any $ref pointers.

    Args:
        schema_path (str): The path to the JSON schema file.

    Returns:
        dict: The dereferenced JSON schema or None.
    """
    try:
        with open(schema_path, 'r') as schema_file:
            schema = jsonref.load(schema_file)
            return schema
    except Exception as e:
        print(f"Error dereferencing schema {schema_path}: {e}")
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
    Iteratively enforce "additionalProperties": false in a JSON schema where it is not declared.
    
    Args:
        schema (dict): The JSON schema to enforce the rule on.
        
    Returns:
        dict: The modified JSON schema with "additionalProperties" set to false where not declared.
    """
    stack = [schema]

    while stack:
        current_schema = stack.pop()

        if not isinstance(current_schema, dict):
            continue

        # Apply additionalProperties: false if type is object and not declared
        if current_schema.get("type") == "object":
            if "additionalProperties" not in current_schema:
                current_schema["additionalProperties"] = False
            elif isinstance(current_schema["additionalProperties"], dict):
                # If additionalProperties is an object (schema), process it as well
                stack.append(current_schema["additionalProperties"])

        # Add nested properties to the stack
        if "properties" in current_schema:
            stack.extend(current_schema["properties"].values())

        # Handle array items
        if current_schema.get("type") == "array" and "items" in current_schema:
            stack.append(current_schema["items"])

        if "prefixItems" in current_schema:
            stack.extend(current_schema["prefixItems"])

        # Handle composition keywords like allOf, anyOf, oneOf, etc.
        for keyword in ["allOf", "anyOf", "oneOf", "not", "then", "else", "if"]:
            if keyword in current_schema:
                if isinstance(current_schema[keyword], list):
                    stack.extend(current_schema[keyword])
                elif isinstance(current_schema[keyword], dict):
                    stack.append(current_schema[keyword])

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
        tuple: (all_docs, all_valid) where:
               - all_docs (list): List of all documents (valid or invalid).
               - all_valid (bool): True if all documents are valid, False if any validation fails.
    """
    dataset_path = os.path.join(files_folder, dataset)
    all_docs = []
    all_valid = True

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

                    errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)

                    # If any document has errors, set all_valid to False
                    if errors:
                        all_valid = False

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON document in {dataset}: {e}")
                    all_valid = False
                    continue
    except Exception as e:
        print(f"Error reading dataset {dataset_path}: {e}")
        return [], False

    return all_docs, all_valid


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
    """
    schema_path = os.path.join(SCHEMA_FOLDER, dataset)

    # Load the schema
    schema = load_schema(schema_path)
    if schema is None:
        return None
    
    # Skip if schema has patternProperties
    if has_pattern_properties_string_search(schema):
        print(f"Skipping {dataset} due to patternProperties in the schema.")
        return None

    # Load and dereference the schema if needed
    dereferenced_schema = load_and_dereference_schema(schema_path)

    # Try modifying the schema to prevent additional properties
    try:
        modified_schema = prevent_additional_properties(dereferenced_schema)
    except Exception as e:
        print(f"Error modifying schema for {dataset}: {e}")
        modified_schema = dereferenced_schema
    
    # Validate all documents against the modified schema
    valid_docs, all_valid = validate_all_documents(dataset, files_folder, modified_schema)
    if all_valid:
        print(f"All documents in {dataset} passed validation with modified schema.")
        schema_to_use = modified_schema
    else:
        print(f"Validation failed for at least one document in {dataset}, reverting to original schema.")
        schema_to_use = dereferenced_schema

    if not valid_docs:
        print(f"No valid documents found in {dataset}. Skipping schema and JSON creation.")
        return None
    # Save the schema to the processed_schemas folder
    schema_save_path = os.path.join(PROCESSED_SCHEMAS_FOLDER, dataset)
    save_to_file(schema_to_use, schema_save_path)

    # Save JSON Lines file with the same name as the schema
    json_save_path = os.path.join(PROCESSED_JSONS_FOLDER, dataset)
    save_json_lines(valid_docs, json_save_path)


def process_datasets():
    """
    Process the datasets in parallel and save the resulting schemas and JSON files in processed folders.
    This function does not take any arguments and uses declared constants.
    """
    datasets = os.listdir(SCHEMA_FOLDER)
    
    # Recreate the processed folders
    recreate_directory(PROCESSED_SCHEMAS_FOLDER)
    recreate_directory(PROCESSED_JSONS_FOLDER)

    with ProcessPoolExecutor() as executor:
        tqdm.tqdm(executor.map(process_single_dataset, datasets, [JSON_FOLDER] * len(datasets)), total=len(datasets))


        
def main():
    process_datasets()

if __name__ == "__main__":
    main()
