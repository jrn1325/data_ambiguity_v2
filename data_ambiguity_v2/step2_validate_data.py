import os
import json
import glob
import jsonschema
import execjs
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
            json_schema = json.load(schema_file)
            jsonschema.Draft7Validator.check_schema(json_schema)
            return json_schema
    except (jsonschema.SchemaError, FileNotFoundError) as e:
        #print(f"Schema validation failed for {schema_path}: {e}")
        return


def extract_pattern_properties_parents(schema, path=[]):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == 'patternProperties':
                yield path
            if isinstance(value, dict) or isinstance(value, list):
                yield from extract_pattern_properties_parents(value, path + [key])
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            #extract_pattern_properties_parents(item, path + [str(index)], paths)
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


def preprocess_data(dataset_folder):
    num_documents = 100

    for dataset in os.listdir(dataset_folder):
        schema_path = os.path.join("schemas", dataset)
        json_schema = validate_schema(schema_path)
        if json_schema:
            dynamic_paths = check_for_dynamic_paths(json_schema)
            if dynamic_paths:
                print(dataset)
                for path in dynamic_paths:
                    print(path)
                print()
                #doc_list = generate_data(json_schema, num_documents)
                #write_data_to_file(schema_path, doc_list)
                #sys.stderr.write(f"Generated data for schema: {schema_path}\n")


def main():
    schema_folder = 'schemas'
    #os.makedirs('fake_files', exist_ok=True)
    #schema_files = glob.glob(os.path.join(schema_folder, '*.json'))
    preprocess_data(schema_folder)


if __name__ == "__main__":
    main()
