import argparse
import json
import os
import time
from collections import defaultdict, Counter
from copy import deepcopy
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

ARRAY_WILDCARD = "<ARRAY_ITEM>"

# -------------------------------
# Document Parsing
# -------------------------------
def parse_document(doc, path=("$",)):
    """
    Parse a JSON document.
    
    Args:
        doc: JSON document
        path: tuple representing the location in the JSON structure
    Yields:
        (path, value) pairs..
    """
    yield path, doc

    if isinstance(doc, dict):
        for k, v in doc.items():
            current_path = path + (k,)
            yield from parse_document(v, current_path)

    elif isinstance(doc, list):
        for item in doc:
            current_path = path + (ARRAY_WILDCARD,)
            yield from parse_document(item, current_path)

def process_document(doc, paths_dict, path_freqs):
    """
    Process a single JSON document.

    Args:
        doc: JSON document
        paths_dict: dict mapping path tuples to list of JSON values
        path_freqs: Counter for path frequencies
    """ 
    for path, value in parse_document(doc):
        path_freqs[path] += 1
        try:
            serialized = json.dumps(value, sort_keys=True)
        except TypeError:
            serialized = str(value)
        paths_dict[path].append(serialized)

def process_dataset(test_json_dir, dataset):
    paths_dict = defaultdict(list)
    path_freqs = Counter()
    num_docs = 0

    dataset_path = os.path.join(test_json_dir, dataset)
    with open(dataset_path, "r") as f:
        for line in f:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            items = doc if isinstance(doc, list) else [doc]
            for item in items:
                if not isinstance(item, dict):
                    continue
                process_document(item, paths_dict, path_freqs)
                num_docs += 1

    return paths_dict, path_freqs, num_docs


# -------------------------------
# Discover schema
# -------------------------------
def discover_schema(value):
    """
    Discover JSON schema for a given JSON value.

    Args:
        value: JSON value
    Returns:
        dict: JSON schema
    """
    if value is None:
        return {"type": "null"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, (int, float)):
        return {"type": "number"}
    elif isinstance(value, str):
        return {"type": "string"}
    elif isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        item_schemas = [discover_schema(item) for item in value]
        items_schema = reduce(merge_schemas, item_schemas)
        return {"type": "array", "items": items_schema}
    elif isinstance(value, dict):
        props = {k: discover_schema(v) for k, v in value.items()}
        return {"type": "object", "properties": props}
    else:
        raise TypeError(f"Unsupported type: {type(value)}")

def remove_duplicate_schemas(schemas):
    """
    Remove duplicate schemas from a list.

    Args:
        schemas: list of JSON schemas
    Returns:
        list of unique JSON schemas
    """
    unique = []
    for schema in schemas:
        if schema not in unique:
            unique.append(schema)
    return unique

def merge_schemas(schema1, schema2):
    """
    Merge two JSON schemas into a unified schema.

    Args:
        schema1: first JSON schema
        schema2: second JSON schema
    Returns:
        dict: merged JSON schema
    """
    # Avoid modifying the original input
    schema1 = deepcopy(schema1)
    schema2 = deepcopy(schema2)

    # If schemas are identical, no need to merge
    if schema1 == schema2:
        return schema1

    # If either is an anyOf, expand it before merging
    if "anyOf" in schema1 or "anyOf" in schema2:
        schemas = []
        if "anyOf" in schema1:
            schemas.extend(schema1["anyOf"])
        else:
            schemas.append(schema1)
        if "anyOf" in schema2:
            schemas.extend(schema2["anyOf"])
        else:
            schemas.append(schema2)
        return {"anyOf": remove_duplicate_schemas(schemas)}

    type1 = schema1.get("type")
    type2 = schema2.get("type")

    # If types differ, wrap both under anyOf
    if type1 != type2:
        return {"anyOf": remove_duplicate_schemas([schema1, schema2])}

    # Merge by type
    if type1 == "object":
        props1 = schema1.get("properties", {})
        props2 = schema2.get("properties", {})
        merged_props = {}

        all_keys = set(props1.keys()) | set(props2.keys())
        for key in all_keys:
            if key in props1 and key in props2:
                merged_props[key] = merge_schemas(props1[key], props2[key])
            elif key in props1:
                merged_props[key] = props1[key]
            else:
                merged_props[key] = props2[key]

        schema1["properties"] = merged_props
        return schema1

    if type1 == "array":
        items1 = schema1.get("items", {})
        items2 = schema2.get("items", {})
        schema1["items"] = merge_schemas(items1, items2)
        return schema1

    return schema1

def discover_schema_from_paths(paths_dict):
    """
    Build a JSON schema from paths.

    Args:
        paths_dict: dict mapping path tuples to list of JSON values
    Returns:
        dict: unified JSON schema
    """
    root_schema = None

    for path, values in paths_dict.items():
        if not values:
            continue

        # Build schema for all values at this path
        value_schemas = [discover_schema(json.loads(v)) for v in values]
        value_schema = reduce(merge_schemas, value_schemas)
        schema = deepcopy(value_schema)

        # Nest schema according to path (skip root '$')
        for key in reversed(path[1:]):
            if key == ARRAY_WILDCARD:
                schema = {"type": "array", "items": schema}
            else:
                schema = {"type": "object", "properties": {key: schema}}

        # Initialize or merge into root_schema
        if root_schema is None:
            root_schema = schema
        else:
            root_schema = merge_schemas(root_schema, schema)

    return root_schema

def compact_anyof(schema):
    """
    Compact 'anyOf' lists by merging compatible array or object schemas
    and removing redundant or single-entry anyOfs.
    """
    if not isinstance(schema, dict):
        return schema

    if "anyOf" in schema:
        merged_object = None
        merged_array = None
        others = []

        for subschema in schema["anyOf"]:
            subschema = compact_anyof(subschema)

            if isinstance(subschema, dict) and subschema.get("type") == "object":
                if merged_object is None:
                    merged_object = subschema
                else:
                    merged_object = merge_schemas(merged_object, subschema)

            elif isinstance(subschema, dict) and subschema.get("type") == "array":
                if merged_array is None:
                    merged_array = subschema
                else:
                    merged_array = merge_schemas(merged_array, subschema)

            else:
                others.append(subschema)

        compacted = []
        if merged_object:
            compacted.append(merged_object)
        if merged_array:
            compacted.append(merged_array)
        compacted.extend(others)

        # Remove duplicates
        unique = []
        for s in compacted:
            if s not in unique:
                unique.append(s)

        # Collapse trivial anyOfs
        if len(unique) == 1:
            return unique[0]

        return {"anyOf": unique}

    # Recursively compact nested schemas
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        for k, v in list(props.items()):
            props[k] = compact_anyof(v)
        schema["properties"] = props

    elif schema.get("type") == "array":
        schema["items"] = compact_anyof(schema.get("items", {}))

    return schema


# -------------------------------
# Add required and additionalProperties
# -------------------------------
def infer_type(values):
    """
    Infer a type from a list of JSON values.
    
    Args:
        values: list of JSON values
    Returns:
        set of inferred types
    """
    types = set()
    for v in values:
        if v is None:
            types.add("null")
        elif isinstance(v, bool):
            types.add("boolean")
        elif isinstance(v, (int, float)):
            types.add("number")
        elif isinstance(v, str):
            types.add("string")
        elif isinstance(v, list):
            types.add("array")
        elif isinstance(v, dict):
            types.add("object")
        else:
            types.add("string")

    return types

def add_required_and_additional(schema, paths_dict, path_freqs, num_docs, path=("$",)):
    """
    Enhance schema with:
    - 'required' fields (appear in all instances of parent)
    - 'additionalProperties' for optional or unknown fields
    
    Rules:
    - A key is 'required' if its path frequency == parent path frequency
      (or == num_docs at root level)
    - Non-required or unknown keys are merged into additionalProperties
    - If all have same type, just use {"type": t}, else use {"anyOf": ...}
    - If no optional/unknown keys exist, set additionalProperties=False
    - If object always empty, omit both 'properties' and 'additionalProperties'
    """

    schema_type = schema.get("type")

    if schema_type == "object":
        props = schema.get("properties", {})
        required = []

        # Determine frequency for this object
        parent_freq = path_freqs.get(path, num_docs if path == ("$",) else 0)

        # Determine required fields
        for key in list(props.keys()):
            child_path = path + (key,)
            if path_freqs.get(child_path, 0) == parent_freq:
                required.append(key)

        if required:
            schema["required"] = sorted(required)

        # Gather possible extra key values
        extra_values = []
        empty_object_seen = False

        for serialized_obj in paths_dict.get(path, []):
            try:
                obj = json.loads(serialized_obj)
            except Exception:
                continue

            if isinstance(obj, dict):
                if not obj:
                    empty_object_seen = True
                else:
                    for key, value in obj.items():
                        # key not required -> belongs in additionalProperties
                        if key not in required:
                            extra_values.append(value)

        # Infer type(s) for additional properties
        if extra_values:
            extra_types = infer_type(extra_values)

            if len(extra_types) == 1:
                schema["additionalProperties"] = {"type": next(iter(extra_types))}
            else:
                schema["additionalProperties"] = {
                    "anyOf": [{"type": t} for t in sorted(extra_types)]
                }

            # Remove non-required keys from properties to avoid redundancy
            for key in list(props.keys()):
                if key not in required:
                    props.pop(key, None)

        elif not empty_object_seen:
            # No extra keys, and not always empty object
            schema["additionalProperties"] = False

        # Remove empty properties dict for always-empty objects
        if not props and not extra_values:
            schema.pop("properties", None)
            schema.pop("additionalProperties", None)

        # Recurse into properties
        for key, subschema in props.items():
            add_required_and_additional(subschema, paths_dict, path_freqs, num_docs, path + (key,))

    elif schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            add_required_and_additional(items, paths_dict, path_freqs, num_docs, path + (ARRAY_WILDCARD,))

    elif "anyOf" in schema:
        anyof_value = schema["anyOf"]
        if isinstance(anyof_value, list):
            for subschema in anyof_value:
                add_required_and_additional(subschema, paths_dict, path_freqs, num_docs, path)

    return schema


# -------------------------------
# Add $defs for repeated schemas
# -------------------------------
def serialize_schema(schema):
    """
    Convert schema dict to a canonical string for hashing.

    Args:
        schema: JSON schema
    Returns:
        str: serialized schema
    """
    return json.dumps(schema, sort_keys=True)

def replace_with_ref(schema, definitions, seen, min_size=2):
    """
    Replace repeated object/array schemas with $ref to definitions.
    Only replaces if a schema structure repeats at least `min_size` times.

    Args:
        schema: JSON schema
        definitions: dict to store definitions
        seen: dict counting seen schema structures
        min_size: minimum occurrences to create a definition
    Returns:
        dict: updated JSON schema with $ref replacements
    """
    if not isinstance(schema, dict):
        return schema

    schema_type = schema.get("type")

    # Recurse into children first
    if schema_type == "object":
        props = schema.get("properties", {})
        new_props = {}
        for k, v in props.items():
            new_props[k] = replace_with_ref(v, definitions, seen, min_size)
        schema["properties"] = new_props

    elif schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            schema["items"] = replace_with_ref(items, definitions, seen, min_size)

    elif "anyOf" in schema:
        schema["anyOf"] = [replace_with_ref(sub, definitions, seen, min_size) for sub in schema["anyOf"]]

    # Only consider non-trivial objects/arrays for definitions
    if schema_type in ("object", "array") or "anyOf" in schema:
        key = serialize_schema(schema)
        seen[key] += 1

        # If we've seen it enough times, assign a $ref
        if seen[key] == min_size:
            def_name = f"{len(definitions) + 1}"
            definitions[def_name] = deepcopy(schema)
            return {"$ref": f"#/$defs/{def_name}"}
        elif seen[key] > min_size:
            # Already defined
            for name, s in definitions.items():
                if serialize_schema(s) == key:
                    return {"$ref": f"#/$defs/{name}"}

    return schema  

def generate_definitions(schema, min_size=2):
    """
    Generate $defs for repeated schemas.
    Only creates a definition if it appears at least `min_size` times.

    Args:
        schema: JSON schema
        min_size: minimum occurrences to create a definition
    Returns:
        dict: updated JSON schema with $defs
    """
    definitions = {}
    seen = defaultdict(int)
    new_schema = replace_with_ref(deepcopy(schema), definitions, seen, min_size=min_size)

    if definitions:
        new_schema["$defs"] = definitions
    return new_schema


# -------------------------------
# Processing and Saving Schemas
# -------------------------------
def save_schema(schema, path):
    """
    Save JSON schema to file.

    Args:
        schema: JSON schema
        path: file path to save the schema
    """
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)

def process_single_dataset(test_json_dir, file, inferred_schemas):
    """
    Process a single dataset to infer its JSON schema.
    
    Args:
        test_json_dir: directory containing test JSON files
        file: filename of the test JSON
        inferred_schemas: directory to save inferred schemas
    Returns:
        tuple: (message, success flag)
    """

    inferred_schema_path = os.path.join(inferred_schemas, file)
    if os.path.exists(inferred_schema_path):
        return f"Skipping {file} — already processed.", None

    try:
        print(f"Processing dataset: {file}", flush=True)
        paths_dict, path_freqs, num_docs = process_dataset(test_json_dir, file)
        inferred_schema = discover_schema_from_paths(paths_dict)
        compacted_schema = compact_anyof(inferred_schema)
        #schema = add_required_and_additional(compacted_schema, paths_dict=paths_dict, path_freqs=path_freqs, num_docs=num_docs, path=("$",))
        #schema = generate_definitions(schema)
        save_schema(compacted_schema, inferred_schema_path)
        return f"Processed {file}", True
    except Exception as e:
        return f"Error processing {file}: {e}", False

# -------------------------------
# Main
# -------------------------------
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("test_jsons", type=str, help="Directory of test JSONs")
    parser.add_argument("inferred_schemas", type=str, help="Directory of inferred schemas")
    args = parser.parse_args()

    os.makedirs(args.inferred_schemas, exist_ok=True)
    files = [f for f in os.listdir(args.test_jsons)]

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_dataset, args.test_jsons, f, args.inferred_schemas): f for f in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring schemas"):
            try:
                msg, success = future.result()
            except Exception as e:
                msg, success = f"Worker exception: {e}", False
            print(msg, flush=True)
            results.append((msg, success))

    end_time = time.time()
    total_success = sum(1 for _, s in results if s)
    total_errors = sum(1 for _, s in results if s is False)
    total_skipped = sum(1 for _, s in results if s is None)

    print("\n--- SUMMARY ---")
    print(f"Processed: {total_success}")
    print(f"Skipped:   {total_skipped}")
    print(f"Errors:    {total_errors}")
    print(f"Total time: {end_time - start_time:.2f} sec", flush=True)


if __name__ == "__main__":
    main()