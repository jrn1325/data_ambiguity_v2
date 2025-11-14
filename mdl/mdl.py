import os
import sys
import json
import argparse
import time

from copy import deepcopy
from tqdm.contrib.concurrent import process_map
from load_json import load_dataset, load_schema
#sys.path.insert(1, '/root/VLDB2024_ReCG/Experiment')


METACHAR_NUM = 12


def MDLExperiment(schema_path, instance_path):
    # algorithm, dataset, exp_num, train_perc

    # 1. Get schema and dataset
    print("[1. Loading Instances and Schemas...]")
    
    schema = load_schema(schema_path)
    original_schema = deepcopy(schema)
    instances = []
    load_dataset(instance_path, instances)
    
    print("[1. Loaded Instances, #: ", len(instances), "]\n")
    print("[1.1. Get number of distinct labels]")
    
    distinct_labels = set()
    metadata = {
        "max_obj_len": 0,
        "max_arr_len": 0
    }
    
    for instance in instances:
        getMetadataRecursive(instance, distinct_labels, metadata)
    distinct_labels_num = len(distinct_labels)
    print("[1.1. Getting number of distinct labels complete]")


    # 2. Calculate SRC
    print("[2. Calculating SRC...]")
    src = 0
    print("[2. Calculating SRC $ref]")
    
    for_ref = deepcopy(original_schema)
    if "definitions" in schema or " $defs" in schema:
        def_key = "definitions" if "definitions" in schema else " $defs"
        for ref_name in for_ref[def_key]:
            src += calculateSRCRecursive(for_ref[def_key][ref_name], distinct_labels_num)

    print("[2. Calculating SRC]")
    for_src = deepcopy(original_schema)
    src += calculateSRCRecursive(for_src, distinct_labels_num)
    print("[2. Calculating SRC Complete]")


    # 3. Calculate DRC
    print("[3. Calculating DRC...]\n")
    argments = [(instance, schema, original_schema, distinct_labels_num, metadata["max_obj_len"], metadata["max_arr_len"]) for instance in instances]
    total_drc = 0
    successes = 0
    
    outer = process_map(runner, argments)
    success_bits, drc_values = zip(*outer)    
    print("[3. DRC Calculation Complete]\n")

    # 4. For each schema node, calculate SRC and DRC
    print("[4. Aggregating Results...]")

    # 4.1. Calculate ordinary SRC, DRC
    for success_bit in success_bits:
        if success_bit:
            successes += 1
    for drc_value in drc_values:
        #drc_value /= len(instances)  # Justin added len(instances)
        total_drc += drc_value
    drc = total_drc
    print("[4. Aggregating Results Complete]")
    return src, drc, successes

def runner(inp):
    instance, schema, original_schema, distinct_labels_num, max_obj_len, max_array_len = inp
    success, drc = calculateDRCRecursive(instance, schema, original_schema, bitSize(distinct_labels_num), bitSize(max_obj_len), bitSize(max_array_len), distinct_labels_num)
    if success:
        return success, drc
    else:
        return success, calculateUnacceptedInstanceDRC(instance, distinct_labels_num)

def calculateSRCRecursive(schema, distinct_labels_num):
    bit_size = bitSize(distinct_labels_num + METACHAR_NUM)

    if isinstance(schema, bool): # Justin Added this
        return 0

    if "type" in schema and schema["type"] == "object":
        if isEmptyObjectSchema(schema):
            symbol_num = 3
            src = bit_size * symbol_num
            return src
        
        elif isHomObjectSchema(schema):
            labels_num = len(list(schema["properties"].keys()))
            symbol_num = 3 + 4 * labels_num
            src = bit_size * symbol_num
            
            for key in schema["properties"]:
                src += calculateSRCRecursive(schema["properties"][key], distinct_labels_num)
            return src

        elif isComObjectSchema(schema):
            sch_labels = set(schema["properties"].keys())
            symbol_num = 3 + 4 * len(sch_labels) + 3
            src = bit_size * symbol_num

            for key in schema["properties"]:
                src += calculateSRCRecursive(schema["properties"][key], distinct_labels_num)
            src += calculateSRCRecursive(schema["additionalProperties"], distinct_labels_num)
            return src

        elif isHetObjectSchema(schema):
            symbol_num = 3
            src = bit_size * symbol_num
            src += calculateSRCRecursive(schema["additionalProperties"], distinct_labels_num)
            return src
        
        elif isObjectSchemaWithNoKeys(schema):
            symbol_num = 3
            src = bit_size * symbol_num
            return src

        else:
            print(json.dumps(schema))
            print("Undefined type of schema 1")

    elif "type" in schema and schema["type"] == "array":
        if isEmptyArraySchema(schema):
            src = 3 * bit_size
            return src
        
        elif isHomArraySchema(schema):
            if "items" in schema:
                schema["prefixItems"] = schema["items"]
                del schema["items"]
                
            subschema_num = len(schema["prefixItems"])
            src = bit_size * (3 + 3 * subschema_num)

            for subschema in schema["prefixItems"]:
                src += calculateSRCRecursive(subschema, distinct_labels_num)
            return src

        elif isHetArraySchema(schema):
            src = 3 * bit_size
            src += calculateSRCRecursive(schema["items"], distinct_labels_num)
            return src
        
        elif isArraySchemaWithNoKeys(schema):
            src = 3 * bit_size
            return src

        else:
            print(json.dumps(schema))
            print("Undefined type of schema 2")

    elif "anyOf" in schema or "oneOf" in schema:
        if "oneOf" in schema:
            schema["anyOf"] = schema["oneOf"]
            del schema["oneOf"]
            
        subschema_len = len(schema["anyOf"])
        src = (subschema_len - 1) * bit_size    
        
        for subschema in schema["anyOf"]:
            src += calculateSRCRecursive(subschema, distinct_labels_num)
        return src
    
    elif "$ref" in schema:
        return 0
    elif "type" in schema and schema["type"] == "string":
        return 0
    elif "type" in schema and schema["type"] == "integer":
        return 0
    elif "type" in schema and schema["type"] == "number":
        return 0
    elif "type" in schema and schema["type"] == "boolean":
        return 0
    elif "type" in schema and schema["type"] == "null":
        return 0
    elif len(schema.keys()) == 0:
        return 0
    else:
        print(json.dumps(schema))
        raise Exception("Error 999: Unexpected schema in SRC")

def calculateDRCRecursive(instance, schema, original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num):
    # Case A. $ref -> unref
    if "$ref" in schema:
        schema, _ = getReferencingNode(original_schema, schema["$ref"])
        
    # Case B. AnyOf
    if "anyOf" in schema:
        drc = bitSize(len(schema["anyOf"]))
        smallest_drc = 100000000000000000
        success = False
        
        for subschema in schema["anyOf"]:
            subschema_success, subschema_drc = calculateDRCRecursive(instance, subschema, original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
            success |= subschema_success
            
            if subschema_success and subschema_drc < smallest_drc:
                smallest_drc = subschema_drc
        
        if success:
            return success, drc + smallest_drc
        else:
            return False, 0
    
    # Now look at instance
    if (type(instance) is int) or (type(instance) is float):
        if "type" in schema and (schema["type"] == "number" or schema["type"] == "integer"):
            return True, 0
        else:
            return False, 0
    
    elif type(instance) is str:
        if "type" in schema and schema["type"] == "string":
            return True, 0
        else:
            return False, 0
    
    elif type(instance) is bool:
        if "type" in schema and schema["type"] == "boolean":
            return True, 0
        else:
            return False, 0
        
    elif instance == None:
        if "type" in schema and schema["type"] == "null":
            return True, 0
        else:
            return False, 0
        
    elif type(instance) is dict:
        # Check type
        if "type" in schema and schema["type"] != "object":
            return False, 0

        if isEmptyObjectSchema(schema):
            if len(instance) == 0:
                return True, 0
            else:
                return False, 0


        if isHomObjectSchema(schema):
            instance_keys = set(instance.keys())
            schema_all_keys = set(schema["properties"].keys())
            if "required" in schema:
                schema_required_keys = set(schema["required"])
            else:
                schema_required_keys = set()
            schema_optional_keys = schema_all_keys - schema_required_keys

            # Check if keys are defined within `properties`
            if not (instance_keys <= schema_all_keys):
                return False, 0

            # Check if all required keys appeared
            if not (schema_required_keys <= instance_keys):
                return False, 0
            
            drc = len(schema_optional_keys)
            success = True
            for key in instance.keys():
                sub_success, sub_drc = calculateDRCRecursive(instance[key], schema["properties"][key], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                success &= sub_success
                drc += sub_drc
            return success, drc
            
        elif isComObjectSchema(schema):
            instance_keys = set(instance.keys())
            schema_all_keys = set(schema["properties"].keys())
            if "required" in schema:
                schema_required_keys = set(schema["required"])
            else:
                schema_required_keys = set()
            schema_optional_keys = schema_all_keys - schema_required_keys

            # Check if all required keys appeared
            if not (schema_required_keys <= instance_keys):
                return False, 0
            
            
            kleene_keys = instance_keys - schema_all_keys
            drc = len(schema_optional_keys) + obj_length_bits + len(kleene_keys) * kleene_bits
            success = True
            for key in instance.keys():
                if key in kleene_keys:
                    sub_success, sub_drc = calculateDRCRecursive(instance[key], schema["additionalProperties"], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                    success &= sub_success
                    drc += sub_drc
                else:
                    sub_success, sub_drc = calculateDRCRecursive(instance[key], schema["properties"][key], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                    success &= sub_success
                    drc += sub_drc
            return success, drc
        
        elif isHetObjectSchema(schema):
            success = True
            drc = obj_length_bits + len(instance) * kleene_bits
            
            for key in instance:
                sub_success, sub_drc = calculateDRCRecursive(instance[key], schema["additionalProperties"], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                success &= sub_success
                drc += sub_drc
            return success, drc
        
        elif isObjectSchemaWithNoKeys(schema):
            return True, calculateUnacceptedInstanceDRC(instance, distinct_labels_num)
            
        else:
            print(json.dumps(schema))
            print(json.dumps(instance))
            raise Exception("calculateDRCRecursive: Undefined object schema!")
    
    elif type(instance) is list:
        # Check type
        if "type" in schema and schema["type"] != "array":
            return False, 0
        
        if isHomArraySchema(schema):
            if "items" in schema:
                schema["prefixItems"] = schema["items"]
            if len(instance) != len(schema["prefixItems"]):
                return False, 0
            
            success = True
            drc = 0
            for i, _ in enumerate(instance):
                sub_success, sub_drc = calculateDRCRecursive(instance[i], schema["prefixItems"][i], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                success &= sub_success
                drc += sub_drc
            return success, drc
        
        elif isHetArraySchema(schema):
            success = True
            drc = arr_length_bits
            for subinstance in instance:
                sub_success, sub_drc = calculateDRCRecursive(subinstance, schema["items"], original_schema, kleene_bits, obj_length_bits, arr_length_bits, distinct_labels_num)
                success &= sub_success
                drc += sub_drc
            return success, drc
        
        elif isEmptyArraySchema(schema):
            if len(instance) == 0:
                return True, 0
            else:
                return False, 0
            
        elif isArraySchemaWithNoKeys(schema):
            return True, calculateUnacceptedInstanceDRC(instance, distinct_labels_num)
        
        if schema == {}:# Justin Added this
            # Accept all values when schema is empty
            if isinstance(instance, list):
                return True, bitSize(len(instance)) 
            else:
                return True, 0
        else:
            print(json.dumps(schema))
            print(json.dumps(instance))
            raise Exception("Error 4")
    
    
    else:
        print(json.dumps(schema))
        print(json.dumps(instance))
        raise Exception("Error 1")
        

#########################################################
# calculateUnacceptedInstanceDRC                        #
#                                                       #
# In: instance                                          #
# Out1: Number of bits needed to encode this instance   #
# Out1: Distinct symbols needed to encode this instance #
#                                                       #
#########################################################
def calculateUnacceptedInstanceDRC(instance, distinct_labels_num):
    bits_needed = InstanceDRCRecursive(instance)
    return bits_needed * bitSize(METACHAR_NUM + distinct_labels_num)

def InstanceDRCRecursive(instance):
    bits_needed = 0

    if type(instance) is dict:
        for key in instance:
            bits_needed += 3
            sub_bits_needed = InstanceDRCRecursive(instance[key])
            bits_needed += sub_bits_needed
        return bits_needed

    elif type(instance) is list:
        for subinstance in instance:
            bits_needed += 2
            sub_bits_needed = InstanceDRCRecursive(subinstance)
            bits_needed += sub_bits_needed
        return bits_needed

    else:
        return 0


# getDistinctLabelsRecursive
def getMetadataRecursive(instance, distinct_labels, metadata):
    if type(instance) is dict:
        keys = set(instance.keys())
        distinct_labels.update(keys)
        
        if len(keys) > metadata["max_obj_len"]:
            metadata["max_obj_len"] = len(keys)
        
        for key in keys:
            getMetadataRecursive(instance[key], distinct_labels, metadata)
        
    elif type(instance) is list:
        if len(instance) > metadata["max_arr_len"]:
            metadata["max_arr_len"] = len(instance)
        for subinstance in instance:
            getMetadataRecursive(subinstance, distinct_labels, metadata)
    else:
        return


# schemaClassification                              
def isHomObjectSchema(schema):
    if "type" in schema and schema["type"] == "object":
        if  ("properties" in schema and "additionalProperties" in schema and schema["additionalProperties"] == False) or \
            ("properties" in schema and "additionalProperties" not in schema):
                return True
    return False

def isComObjectSchema(schema):
    if "type" in schema and schema["type"] == "object":
        if "properties" in schema and "additionalProperties" in schema and schema["additionalProperties"] != False:
            return True
    return False

def isHetObjectSchema(schema):
    if "type" in schema and schema["type"] == "object":
        if "properties" not in schema and "additionalProperties" in schema and schema["additionalProperties"] != False:
            return True
    return False

def isEmptyObjectSchema(schema):
    if "type" in schema and schema["type"] == "object":
        if "maxProperties" in schema and schema["maxProperties"] == 0:
            return True
        if "properties" in schema and schema["properties"] == {} and ("additionalProperties" not in schema or ("additionalProperties" in schema and schema["additionalProperties"] == False)):
            return True
    return False

def isObjectSchemaWithNoKeys(schema):
    if "type" in schema and schema["type"] == "object":
        if len(schema.keys()) == 1 or (len(schema.keys()) == 2 and "description" in schema):
            return True
    elif len(schema.keys()) == 0:
        return True
        

def isHomArraySchema(schema):
    if "type" in schema and schema["type"] == "array":
        if "items" in schema and type(schema["items"]) is list:
            return True
        elif "prefixItems" in schema:
            return True
    return False

def isHetArraySchema(schema):
    if "type" in schema and schema["type"] == "array":
        if "items" in schema and type(schema["items"]) is dict:
            return True
    return False

def isEmptyArraySchema(schema):
    if "maxItems" in schema and schema["maxItems"] == 0:
        return True
    return False

def isArraySchemaWithNoKeys(schema):
    if "type" in schema and schema["type"] == "array":
        if len(schema.keys()) == 1:
            return True


# Schema UTILS
def getReferencingNode(original_schema, ref_path):
    # Currently only treats references starting with "#"!
    # Others have to be implemented additionaly if needed
    # As I've seen to this date, haven't seen any uses different from "#..."

    # if ref_path == '#':
    #     exc = RecursionError("Ref to self")
    #     logging.exception(exc)
    #     raise exc

    split_res = ref_path.split('#')
    split_res.append('')
    root_, key_ = split_res[0], split_res[1]
    res = resolveKey(key_, original_schema)
    return res, tuple(key_.split('/')[1:])

def resolveKey(key_, tree_):
    while key_.startswith('/'):
        key_ = key_[1:]
    ref_key = key_.split('/')
    for step in ref_key:
        if not step:
            continue
        try:
            tree_ = tree_[step]
        except KeyError:
            print("WRONG")
    return tree_

def getToPath(schema, path):
    for step in path:
        schema = schema[step]
    return schema


# UTILS
def clearDict(dict_):
    for key in dict_:
        dict_[key].clear()

def extendDictionaries(to_dict, from_dict):
    for key in to_dict:
        to_dict[key].extend(from_dict[key])

def getSchemaWithPath(schema, path):
    if len(path) == 0:
        return schema
    else:
        return getSchemaWithPath(schema[path[0]], path[1:])

def encodeLength(length):
    return 2 * bitSize(length) + 1

def bitSize(length):
    return (length - 1).bit_length()



def main(argv):
    original_schemas = "converted_processed_schemas/"
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Using MDL to evaluate JSON Schema quality on datasets")
    parser.add_argument("dataset_dir", help="Directory containing datasets corresponding to the schemas")
    parser.add_argument("schema_dir", help="Directory containing JSON Schema files")
    parser.add_argument("output_dir", help="Directory to save the output results")
    parser.add_argument("mode", help="Mode of operation", choices=["adapter", "full", "jxplain", "infer", "gt"])
    args = parser.parse_args(argv)
    print(f"[INFO] Running with args: {args}")

    dataset_dir = args.dataset_dir
    mode = args.mode
    if mode != "infer":
        schema_dir = f"{args.schema_dir}_{args.mode}"
    else:
        schema_dir = args.schema_dir

    output_dir = args.output_dir

    out_path = os.path.join(output_dir, f"{schema_dir}_mdl_scores.json")
    result_list = []

    for schema in os.listdir(schema_dir):
        if mode == "gt":
            schema_path = os.path.join(original_schemas, schema)
        else:   
            schema_path = os.path.join(schema_dir, schema)
            
        dataset_path = os.path.join(dataset_dir, schema)

        # Skip non-JSON files if needed
        if not os.path.isfile(schema_path):
            continue

        # Run MDL Experiment
        src, drc, accepted_num = MDLExperiment(schema_path, dataset_path)
        mdl = src + drc

        result = {
            "filename": schema,
            "SRC": src,
            "DRC": drc,
            "MDL": mdl,
            "Accepted": accepted_num
        }

        result_list.append(result)
        print(f"[RESULT] Dataset: {schema}\tSRC: {src}\tDRC: {drc}\tMDL: {mdl}\tAccepted: {accepted_num}")

    # Write all results to output file
    with open(out_path, "w") as file:
        for r in result_list:
            json.dump(r, file)
            file.write("\n")

    print(f"[INFO] Results saved to {out_path}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Total elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main((sys.argv)[1:])