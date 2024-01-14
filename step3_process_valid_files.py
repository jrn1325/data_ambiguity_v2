import os
import shutil
import sys
import json
from pyjson5 import decode, decode_io, encode, encode_io


def process_valid_files(files, processed_files): 
    #processed_files = "./processed_valid_files"
    # Create folder to store processed files
    if os.path.exists(processed_files):
        # Delete folder
        shutil.rmtree(processed_files)
    # Create folder to store processed files
    os.makedirs(processed_files)
    
    count = 0
    # Assign folder
    folder = "./valid_schemas"
    total_val = 0
    total = 0
    # Loop over the schemas
    for schema in os.listdir(folder):
        # Get a schema file
        s = os.path.join(folder, schema)
        # Load the schema file
        with open(s) as f:
            json_schema = decode_io(f)
        # Get the properties
        filename = schema
        if "properties" in json_schema:# and os.path.isfile(files + filename):
            count += 1
            schema_properties_keys = json_schema["properties"].keys()
            ''' for real files
            # Get filename 
            if ".schema" in s.split('/')[-1]:
                s = s.replace(".schema", "")
            filename = s.split('/')[-1]
            '''
            #filename = schema
            # Check if file exists
            #if os.path.isfile(files + filename):
            # Create a new folder if it doesn't exist to store the files founds
            new_file = processed_files + filename
            # Open file for writing
            outfile = open(new_file, 'a')
            num_docs = 0
            valid_docs = 0
            # Loop over each line of the file
            for line in open(files + filename, encoding = 'utf-8-sig'):
                num_docs += 1
                # Load line into a json document
                doc = decode(line)
                # Check if the doc is a dictionary
                if isinstance(doc, dict):
                    # Get the json keys
                    json_keys = doc.keys()
                    # Calculate the intersection between json file keys and schema properties
                    intersection = set(json_keys).intersection(set(schema_properties_keys))
                    # Check if the intersection list is not empty
                    if len(intersection) > 0:
                        # Write the json object to file
                        outfile.write(line)
                        valid_docs += 1
            # Close the file
            outfile.close()
            print(f"Schema: {filename}, Total number of documents: {num_docs}, Valid docs:", valid_docs)
            total_val += valid_docs
            total += num_docs
        else:
            print(s, "has no properties at the top level.")
    print(f"total: {total_val / total}")
    print(count)


def main():
    # Get arguments
    json_files, processed_files_folder = sys.argv[-2:]
 
    # Process valid files
    process_valid_files(json_files, processed_files_folder)

if __name__ == '__main__':
    main()

