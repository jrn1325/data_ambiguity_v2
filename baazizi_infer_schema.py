import argparse
import json
import os
import hashlib
import time
from collections import defaultdict, Counter
from copy import deepcopy
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

JSON_FOLDER = "processed_jsons"
ARRAY_WILDCARD = "<ARRAY_ITEM>"

# -------------------------------
# Document Parsing
# -------------------------------
def parse_document(doc, path=("$",)):
    """Yield (path, value) for all nodes in the JSON document."""
    yield path, doc  # Always emit the current node

    if isinstance(doc, dict):
        for k, v in doc.items():
            current_path = path + (k,)
            yield from parse_document(v, current_path)

    elif isinstance(doc, list):
        for item in doc:
            current_path = path + (ARRAY_WILDCARD,)
            yield from parse_document(item, current_path)


def process_document(doc, paths_dict, path_freqs):
    for path, value in parse_document(doc):
        path_freqs[path] += 1
        try:
            serialized = json.dumps(value, sort_keys=True)
        except TypeError:
            serialized = str(value)
        paths_dict[path].append(serialized)

def process_dataset(dataset):
    paths_dict = defaultdict(list)
    path_freqs = Counter()
    num_docs = 0

    dataset_file = os.path.join(JSON_FOLDER, dataset.replace("_results.csv", ".json"))
    if not os.path.exists(dataset_file):
        print(f"Dataset not found: {dataset_file}")
        return paths_dict, path_freqs, 0, dataset

    with open(dataset_file, "r") as f:
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

    return paths_dict, path_freqs, num_docs, dataset


# -------------------------------
# Discover schema
# -------------------------------
def discover_schema(value):
    """Infer JSON schema for any JSON value."""
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
    """Remove duplicate schemas from a list, preserving order."""
    unique = []
    for schema in schemas:
        if schema not in unique:
            unique.append(schema)
    return unique

def merge_schemas(schema1, schema2):
    """Merge two JSON schemas into one safely without nested helper functions."""
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

    # Primitives of same type are compatible
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
    """Infer a generalized type from a list of JSON values."""
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

def add_required_and_additional(schema, paths_dict, path_freqs, path=("$",)):
    """
    Recursively add 'required' and infer 'additionalProperties' based on observed data.

    - A field is 'required' if it appears in all objects at its parent path.
    - additionalProperties is False if no extra keys exist, else a generalized schema
      covering the types of extra keys observed.
    - Empty objects are allowed implicitly (no additionalProperties added).
    """
    schema_type = schema.get("type")

    if schema_type == "object":
        props = schema.get("properties", {})
        required = []

        # Determine required fields
        parent_freq = path_freqs.get(path, 0)
        for key in props.keys():
            child_path = path + (key,)
            if path_freqs.get(child_path, 0) == parent_freq:
                required.append(key)
        if required:
            schema["required"] = sorted(required)

        # Determine additionalProperties
        extra_values = []
        empty_object_seen = False
        for serialized_obj in paths_dict.get(path, []):
            obj = json.loads(serialized_obj)
            if isinstance(obj, dict):
                if not obj:
                    empty_object_seen = True
                else:
                    for key, value in obj.items():
                        if key not in props:
                            extra_values.append(value)

        if extra_values:
            extra_types = infer_type(extra_values)

            if len(extra_types) == 1:
                schema["additionalProperties"] = {"type": extra_types.pop()}
            else:
                schema["additionalProperties"] = {"anyOf": [{"type": t} for t in sorted(extra_types)]}

        elif not empty_object_seen:
            schema["additionalProperties"] = False

        # Recurse into properties
        for key, subschema in props.items():
            add_required_and_additional(subschema, paths_dict, path_freqs, path + (key,))

    elif schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            add_required_and_additional(items, paths_dict, path_freqs, path + (ARRAY_WILDCARD,))

    elif "anyOf" in schema:
        anyof_value = schema["anyOf"]
        if isinstance(anyof_value, list):
            for subschema in anyof_value:
                add_required_and_additional(subschema, paths_dict, path_freqs, path)

    return schema




# -------------------------------
# Add $defs for repeated schemas
# -------------------------------
def serialize_schema(schema):
    """Convert schema dict to a canonical string for hashing."""
    return json.dumps(schema, sort_keys=True)

def replace_with_ref(schema, definitions, seen):
    """
    Replace repeated object/array schemas with $ref to definitions.
    
    Returns the transformed schema.
    """
    schema_type = schema.get("type")

    # Only consider objects, arrays, or anyOf for definitions
    if schema_type in ("object", "array") or "anyOf" in schema:
        key = serialize_schema(schema)
        if key in seen:
            return {"$ref": f"#/$defs/{seen[key]}"}
        else:
            def_name = f"{len(seen) + 1}"
            seen[key] = def_name

            schema_copy = deepcopy(schema)

            # Recurse into children explicitly
            if schema_type == "object":
                props = schema_copy.get("properties", {})
                new_props = {}
                for k, v in props.items():
                    new_props[k] = replace_with_ref(v, definitions, seen)
                schema_copy["properties"] = new_props
            elif schema_type == "array":
                items = schema_copy.get("items")
                if isinstance(items, dict):
                    schema_copy["items"] = replace_with_ref(items, definitions, seen)
            elif "anyOf" in schema_copy:
                schema_copy["anyOf"] = [replace_with_ref(sub, definitions, seen) for sub in schema_copy["anyOf"]]

            definitions[def_name] = schema_copy
            return {"$ref": f"#/$defs/{def_name}"}
    return schema  # primitives remain unchanged

def generate_definitions(schema):
    """
    Generate $defs for repeated schemas.
    
    Returns a new schema with a top-level $defs section.
    """
    definitions = {}
    seen = {}
    new_schema = replace_with_ref(deepcopy(schema), definitions, seen)
    return {"$defs": definitions, **new_schema}




def save_schema(schema, path):
    """Save JSON schema to file."""
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)


def process_single_dataset(file, inferred_schemas):
    """Process a single dataset end-to-end."""
    if not file.endswith("_results.csv"):
        return f"Skipping {file} (not a results CSV)", None

    dataset = file.replace("_results.csv", ".json")
    inferred_schema_path = os.path.join(inferred_schemas, dataset)
    if os.path.exists(inferred_schema_path):
        return f"Skipping {dataset} — already processed.", None

    try:
        print(f"Processing dataset: {dataset}", flush=True)
        paths_dict, path_freqs, _, _ = process_dataset(dataset)
        inferred_schema = discover_schema_from_paths(paths_dict)
        compacted_schema = compact_anyof(inferred_schema)
        schema = add_required_and_additional(compacted_schema, paths_dict=paths_dict, path_freqs=path_freqs, path=("$",))
        #schema = generate_definitions(schema)
        save_schema(schema, inferred_schema_path)
        return f"Processed {dataset}", True
    except Exception as e:
        return f"Error processing {file}: {e}", False

# -------------------------------
# Main
# -------------------------------
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_input", type=str, help="Directory for ground truth CSVs")
    parser.add_argument("inferred_schemas", type=str, help="Directory for inferred schemas")
    args = parser.parse_args()

    os.makedirs(args.inferred_schemas, exist_ok=True)
    files = [f for f in os.listdir(args.eval_input) if f.endswith("_results.csv")]

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_dataset, f, args.inferred_schemas): f for f in files}
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