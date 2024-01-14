import os
import json
import glob
import jsonref
import jsonschema
import execjs
import shutil
import sys

def generate_data(schema, num_documents):
    js_function = """
    function generateJsonData(schema) {
        const migrate = require("json-schema-migrate");
        const jsf = require("@michaelmior/json-schema-faker");
        jsf.extend('faker', () => require('@faker-js/faker'));
        jsf.option('noAdditional', true);
        migrate.draft7(schema);
        return jsf.generate(schema);
    }
    """
    context = execjs.compile(js_function)

    data_list = []
    for _ in range(num_documents):
        try:
            data = context.call("generateJsonData", schema)
            if isinstance(data, dict) and len(data) > 0:
                data_list.append(data)
            else:
                continue
        except Exception as e:
            print(e)
            continue
        
    return data_list


def validate_schema(schema_path):
    try:
        with open(schema_path, 'r') as schema_file:
            #print(schema_path)
            json_schema = jsonref.load(schema_file)
            #jsonschema.Draft7Validator.check_schema(json_schema)
            return json_schema
    except (jsonschema.SchemaError, FileNotFoundError, json.decoder.JSONDecodeError, jsonref.JsonRefError, ValueError, AttributeError, RecursionError) as e:
        print(f"Schema validation failed for {schema_path}: {e}")
        return


def count_pattern_properties(schema):
    """
    Counts the occurrences of "patternProperties" key anywhere in the JSON schema.
    """
    count = 0

    if isinstance(schema, dict):
        if "patternProperties" in schema:
            count += 1
        elif "properties" not in schema and schema.get("additionalProperties", False) is not False:
            count += 1
        for key, value in schema.items():
            count += count_pattern_properties(value)

    elif isinstance(schema, list):
        for item in schema:
            count += count_pattern_properties(item)

    return count


def extract_pattern_properties_parents(schema, path=[]):
    if isinstance(schema, dict):
        if 'patternProperties' in schema:
            yield path
        elif 'properties' not in schema and schema.get('additionalProperties', False) is not False:
            yield path

        for key, value in schema.items():
            yield from extract_pattern_properties_parents(value, path + [key])

                
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            yield from extract_pattern_properties_parents(item, path)


def check_for_dynamic_paths(json_schema):
    return list(extract_pattern_properties_parents(json_schema))


def write_data_to_file(schema_path, doc_list):
    schema_name = os.path.splitext(os.path.basename(schema_path))[0]
    output_file = f'fake_files/{schema_name}.json'
    if os.path.isfile(output_file):
        pass
    else:
        with open(output_file, 'w') as json_file:
            for doc in doc_list:
                json.dump(doc, json_file, indent=None)
                json_file.write('\n')


def preprocess_data(dataset_folder, valid_schemas, valid_files):
    num_documents = 100
    #### Check the distribution of dynamic keys in schemas
    #### The percentage of the keys that are dynamic
    #### Histogram of F1-score by schema
    #### Maybe exlude schemas above a certain size?
    for dataset in os.listdir(dataset_folder):
        schema_path = os.path.join(dataset_folder, dataset)
        if schema_path == "/home/jrn1325/schemas/jenkins-x-pipelines.json" or schema_path == "/home/jrn1325/schemas/google-cloud-workflows.json":
            continue
        json_schema = validate_schema(schema_path)
        try:
            if json_schema:
                # Check if file exists and not empty
                output_file = f'/home/jrn1325/jsons/{dataset}'
                
            
                if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
                    dynamic_paths = check_for_dynamic_paths(json_schema)
                    num_dynamic_keys = count_pattern_properties(json_schema)
                    print("schema:", dataset, "# dynamic keys:", num_dynamic_keys)
                    if dynamic_paths and num_dynamic_keys >= 1:
                        valid_schema_path = os.path.join(valid_schemas, dataset)
                        with open(valid_schema_path, 'w') as schema_file:
                            json.dump(json_schema, schema_file)
                            
                        valid_file_path = os.path.join(valid_files, dataset)
                        with open(output_file, 'r') as source_file, open(valid_file_path, 'w') as json_file:
                            for line in source_file:
                                try:
                                    data = json.loads(line)
                                    json.dump(data, json_file)
                                    json_file.write('\n') 
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding JSON: {e}")        
        except (jsonschema.SchemaError, FileNotFoundError, json.decoder.JSONDecodeError, jsonref.JsonRefError, ValueError, AttributeError, RecursionError) as e:
            print(f"Schema processing failed for {schema_path}: {e}")
            continue
                        

        '''
        print(dataset)
        for path in dynamic_paths:
            print(path)
        print()
        '''
        #doc_list = generate_data(json_schema, num_documents)
        #write_data_to_file(schema_path, doc_list)
        #sys.stderr.write(f"Generated data for schema: {schema_path}\n")


def main():
    schema_folder = "/home/jrn1325/schemas"
    valid_schemas = "valid_schemas"
    valid_files = "real_files"
    
    if os.path.exists(valid_schemas):
        try:
            shutil.rmtree(valid_schemas)
            print(f"Directory '{valid_schemas}' deleted.")
            shutil.rmtree(valid_files)
            print(f"Directory '{valid_files}' deleted.")
        except OSError as e:
            print(f"Error deleting directory: {e}")

    os.makedirs(valid_schemas)
    os.makedirs(valid_files)
    #schema_files = glob.glob(os.path.join(schema_folder, '*.json'))
    preprocess_data(schema_folder, valid_schemas, valid_files)


if __name__ == "__main__":
    main()
