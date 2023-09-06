import os
import sys
import json
import jsonref
import shutil
import requests
import yaml



# Define constants
SCHEMA_URL = "https://www.schemastore.org/api/json/catalog.json"
# Create a list to store the schema_names
schemas_written = []


def create_schemas_folder():
    # Create a folder to store schemas
    folder = "schemas/"
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    return folder


def get_schemas():
    '''
    Input: None
    Output: List of schema names
    Purpose: Get the list of schema names
    '''
    try:
        data = requests.get(SCHEMA_URL)
        data.raise_for_status()
        json_obj = data.json()
        return json_obj.get("schemas", [])
    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
        return []


def get_schema_and_name(schema):
    # Get the schema object and the schema name
    schema_url = schema["url"]
    json_schema = ""
    content = ""
    
    try:
        response = requests.get(schema_url)
        response.raise_for_status()
        content = response.content
        json_schema = json.loads(content)
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        try:
            json_schema = json.dumps(yaml.safe_load(content))
        except yaml.YAMLError as e:
            print(e)
    
    schema_name = schema_url.split('/')[-1]
    if schema_name.endswith(".yaml"):
        schema_name = schema_name[:-5] + '.json'
    return json_schema, schema_name


def write_schemas(schema, schema_name, schemas_folder):
    # Write the schema
    try:
        dereferenced_data = jsonref.JsonRef.replace_refs(schema)
        json_schema = json.dumps(dereferenced_data, indent=2)
        with open(schemas_folder + schema_name, 'w', encoding="utf-8") as outfile:
            outfile.write(json_schema)
        print(schema_name, "written.")
    except (jsonref.JsonRefError, ValueError, TypeError) as e:
        print(f"Error: {e}")


def main():
    schemas_folder = create_schemas_folder()
    schemas = get_schemas()
    for schema in schemas:
        json_schema, schema_name = get_schema_and_name(schema)
        write_schemas(json_schema, schema_name, schemas_folder)


if __name__ == '__main__':
    main()
