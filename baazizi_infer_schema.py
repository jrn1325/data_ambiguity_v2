import argparse
import json
import hashlib
import os
import time

from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from functools import reduce
from tqdm import tqdm

JSON_FOLDER = "processed_jsons"


def parse_document(doc, path=("$",)):
    """Recursively yield (path, value) pairs for all keys and their values."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            current_path = path + (key,)
            yield current_path, value
            if isinstance(value, (dict, list)):
                yield from parse_document(value, current_path)
    elif isinstance(doc, list):
        for item in doc:
            current_path = path + ("*",)
            yield current_path, item
            if isinstance(item, (dict, list)):
                yield from parse_document(item, current_path)

def process_document(doc, paths_dict, path_freqs, paths_seen=None):
    """
    Extract all paths and their values from a JSON document, tracking frequency.

    Args:
        doc (dict | list): JSON document.
        paths_dict (defaultdict(list)): Maps each path to list of observed values.
        path_freqs (Counter): Counts occurrences of each path across documents.
        paths_seen (defaultdict(set), optional): Tracks seen serialized values for efficiency.
    """
    if paths_seen is None:
        paths_seen = defaultdict(set)

    for path, value in parse_document(doc):
        path_freqs[path] += 1

        # Serialize the value
        if isinstance(value, (dict, list)):
            try:
                serialized = json.dumps(value, sort_keys=True)
            except TypeError:
                serialized = str(value)
        else:
            serialized = json.dumps(value)

        # Only store new values
        if serialized not in paths_seen[path]:
            paths_dict[path].append(serialized)
            paths_seen[path].add(serialized)

def get_doc_hash(doc):
    """Return a hash of the canonical JSON form of the document."""
    try:
        canonical = json.dumps(doc, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return None

def process_dataset(dataset):
    """Process a JSON dataset, deduplicate docs, and extract path info."""
    paths_dict = defaultdict(list)
    path_freqs = Counter()
    paths_seen = defaultdict(set)
    num_docs = 0
    seen_hashes = set()

    dataset = dataset.replace("_results.csv", ".json")
    dataset_path = os.path.join(JSON_FOLDER, dataset)

    try:
        with open(dataset_path, "r") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[Line {line_number}] JSON decode error in {dataset}: {e}", flush=True)
                    continue

                items = doc if isinstance(doc, list) else [doc]
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    doc_hash = get_doc_hash(item)
                    if doc_hash is None or doc_hash in seen_hashes:
                        continue

                    seen_hashes.add(doc_hash)
                    process_document(item, paths_dict, path_freqs, paths_seen)
                    num_docs += 1

    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}", flush=True)

    return paths_dict, path_freqs, num_docs, dataset




    """Heuristic to allow/disallow additionalProperties."""
    if total_keys <= 0:
        return True
    return not (len(keys) / total_keys >= threshold)


def unique_anyOf(schemas):
    """Deduplicate schemas in an anyOf list."""
    seen, unique = set(), []
    for s in schemas:
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def flatten_anyOf(schema):
    """Flatten nested anyOf constructs."""
    if not isinstance(schema, dict) or "anyOf" not in schema:
        return schema

    result = []
    for s in schema["anyOf"]:
        if isinstance(s, dict) and "anyOf" in s:
            result.extend(flatten_anyOf(s)["anyOf"])
        else:
            result.append(s)
    rest = {k: v for k, v in schema.items() if k != "anyOf"}
    return {**rest, "anyOf": unique_anyOf(result)}


def merge_schemas(schema1, schema2):
    """Merge two JSON schemas."""
    if not isinstance(schema1, dict) or not isinstance(schema2, dict):
        return flatten_anyOf({"anyOf": [schema1, schema2]})

    schema1, schema2 = flatten_anyOf(schema1), flatten_anyOf(schema2)
    type1, type2 = schema1.get("type"), schema2.get("type")
    if type1 != type2:
        return flatten_anyOf({"anyOf": [schema1, schema2]})

    new_schema = deepcopy(schema1)

    if type1 == "object":
        props1, props2 = schema1.get("properties", {}), schema2.get("properties", {})
        if "*" in props1 and "*" in props2:
            return {"type": "array", "items": merge_schemas(props1["*"], props2["*"])}
        elif "*" in props1 or "*" in props2:
            return flatten_anyOf({"anyOf": [schema1, schema2]})

        merged_props = deepcopy(props1)
        for k, v in props2.items():
            merged_props[k] = merge_schemas(merged_props[k], v) if k in merged_props else v
        new_schema["properties"] = merged_props

        counts1, counts2 = schema1.get("field_counts", {}), schema2.get("field_counts", {})
        if counts1 or counts2:
            merged_counts = deepcopy(counts1)
            for k, v in counts2.items():
                merged_counts[k] = merged_counts.get(k, 0) + v
            new_schema["field_counts"] = merged_counts
            new_schema["total_count"] = schema1.get("total_count", 0) + schema2.get("total_count", 0)

        return new_schema

    if type1 == "array":
        items1, items2 = schema1.get("items", {}), schema2.get("items", {})
        if all(isinstance(i, dict) and i.get("type") == "object" for i in [items1, items2]):
            return {"type": "array", "items": merge_schemas(items1, items2)}
        items1_list = items1.get("anyOf", [items1]) if isinstance(items1, dict) else [items1]
        items2_list = items2.get("anyOf", [items2]) if isinstance(items2, dict) else [items2]
        return {"type": "array", "items": {"anyOf": unique_anyOf(items1_list + items2_list)}}

    return new_schema

def discover_schema(value):
    """Infer schema for a JSON value."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        item_schemas = [discover_schema(item) for item in value]
        return {"type": "array", "items": {"anyOf": unique_anyOf(item_schemas)}}
    if isinstance(value, dict):
        properties = {k: discover_schema(v) for k, v in value.items()}
        return {"type": "object", "properties": properties, "field_counts": {k: 1 for k in properties}, "total_count": 1}
    raise TypeError(f"Unsupported type: {type(value)}")

def set_additional_properties(keys, total_keys, threshold=0.9):

    """Heuristic to allow/disallow additionalProperties."""
    if total_keys <= 0:
        return True
    return not (len(keys) / total_keys >= threshold)

def add_required_and_additional_properties(schema, path_freqs, prefix=("$",)):
    """
    Recursively adds `required` and `additionalProperties` to object schemas
    using path frequency counts.

    Args:
        schema (dict): Current JSON schema to update.
        path_freqs (Counter): Maps full paths to counts across all objects.
        prefix (tuple): Current path prefix.
    Returns:
        dict: Updated schema.
    """
    if not isinstance(schema, dict):
        return schema

    if schema.get("type") == "object" and "properties" in schema:
        # Count how many objects at this path contain each key
        key_counts = {
            key: path_freqs.get(prefix + (key,), 0)
            for key in schema["properties"]
        }

        # Total number of documents (objects) at this path
        total_objects_here = max(key_counts.values(), default=0)

        # Determine required keys (appear in all objects at this path)
        required_keys = [k for k, count in key_counts.items() if count == total_objects_here]
        if required_keys:
            schema["required"] = required_keys

        # Determine additionalProperties: False if no extra keys exist
        observed_keys = {p[-1] for p in path_freqs if p[:-1] == prefix}
        schema["additionalProperties"] = not all(k in schema["properties"] for k in observed_keys)

        # Recurse into properties
        for key, val in schema["properties"].items():
            schema["properties"][key] = add_required_and_additional_properties(val, path_freqs, prefix + (key,))

    # Arrays: recurse into items using '*'
    elif schema.get("type") == "array" and "items" in schema:
        schema["items"] = add_required_and_additional_properties(schema["items"], path_freqs, prefix + ("*",))

    # anyOf / oneOf
    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            schema[key] = [add_required_and_additional_properties(s, path_freqs, prefix) for s in schema[key]]

    return schema



def canonicalize_schema(schema):
    """Canonicalize schema for deduplication."""
    if isinstance(schema, dict):
        return {k: canonicalize_schema(schema[k]) for k in sorted(schema.keys())}
    if isinstance(schema, list):
        try:
            return sorted((canonicalize_schema(x) for x in schema), key=lambda x: json.dumps(x, sort_keys=True))
        except Exception:
            return [canonicalize_schema(x) for x in schema]
    return schema

def dedupe_schema(schema, seen=None):
    """Deduplicate repeated sub-schemas."""
    if seen is None:
        seen = {}
    if isinstance(schema, dict):
        key = json.dumps(canonicalize_schema(schema), sort_keys=True)
        if key in seen:
            return seen[key]
        copy_schema = {}
        seen[key] = copy_schema
        for k, v in schema.items():
            copy_schema[k] = dedupe_schema(v, seen)
        return copy_schema
    if isinstance(schema, list):
        return [dedupe_schema(item, seen) for item in schema]
    return schema

def normalize_star(schema):
    """Convert properties with '*' into arrays recursively."""
    if not isinstance(schema, dict):
        return schema
    if schema.get("type") == "object" and "properties" in schema and "*" in schema["properties"]:
        item_schema = normalize_star(schema["properties"]["*"])
        return {"type": "array", "items": item_schema}
    for k, v in list(schema.items()):
        if isinstance(v, dict):
            schema[k] = normalize_star(v)
        elif isinstance(v, list):
            schema[k] = [normalize_star(x) for x in v]
    return schema

def remove_temporary_fields(schema):
    if isinstance(schema, dict):
        schema.pop("field_counts", None)
        schema.pop("total_count", None)
        for v in schema.values():
            remove_temporary_fields(v)
    elif isinstance(schema, list):
        for item in schema:
            remove_temporary_fields(item)
    return schema

def finalize_schema(schema):
    """Finalize schema: normalize '*', deduplicate anyOf/oneOf, remove temp fields."""
    schema = normalize_star(schema)
    schema = remove_temporary_fields(schema)

    for key in ("anyOf", "oneOf"):
        if isinstance(schema, dict) and key in schema:
            merged = []
            for s in schema[key]:
                s = finalize_schema(s)
                if s not in merged:
                    merged.append(s)
            schema[key] = merged[0] if len(merged) == 1 else merged

    return dedupe_schema(schema)

def build_schema_for_path(path, values):
    """
    Build a schema for a given path and list of JSON string values.
    Avoids artificial object wrapping that creates unnecessary anyOfs.
    """
    # Infer schema for all values at this path
    value_schemas = [discover_schema(json.loads(v)) for v in values]
    merged_schema = reduce(merge_schemas, value_schemas)

    # Only wrap intermediate keys in objects if path length > 1
    for key in path[1:-1]:
        merged_schema = {
            "type": "object",
            "properties": {key: merged_schema},
            "required": [key],
            "additionalProperties": False
        }

    return merged_schema

def insert_schema_at_path(root, path, value_schema):
    """Insert a value schema into the root schema following the given path safely."""
    current = root
    for key in path[1:]:  # skip "$"
        if key == "*":
            # Current must become an array
            if current.get("type") != "array":
                current.clear()
                current.update({"type": "array", "items": {}})
            current = current["items"]
        else:
            if current.get("type") != "object":
                current.clear()
                current.update({"type": "object", "properties": {}})
            props = current.setdefault("properties", {})
            if key not in props:
                props[key] = {}
            current = props[key]

    # Merge the leaf schema safely
    if current:
        # Only merge if current is not empty
        merged = merge_schemas(current, value_schema)
        current.clear()
        current.update(merged)
    else:
        # If empty, just set it
        current.update(value_schema)

def collect_subobjects(schema, path=()):
    """
    Traverse schema and collect object sub-schemas, handling arrays ('*').
    Returns:
        hash_to_paths: {hash: list of paths where this schema appears}
        hash_to_schema: {hash: schema object}
    """
    hash_to_paths = defaultdict(list)
    hash_to_schema = {}

    if not isinstance(schema, dict):
        return hash_to_paths, hash_to_schema

    schema_type = schema.get("type")
    if schema_type == "object" and "properties" in schema:
        canonical = json.dumps(canonicalize_schema(schema), sort_keys=True)
        h = hashlib.md5(canonical.encode()).hexdigest()
        hash_to_paths[h].append(path)
        hash_to_schema[h] = schema

        for key, val in schema["properties"].items():
            child_paths, child_schemas = collect_subobjects(val, path + (key,))
            for k, v in child_paths.items():
                hash_to_paths[k].extend(v)
            hash_to_schema.update(child_schemas)

    elif schema_type == "array" and "items" in schema:
        child_paths, child_schemas = collect_subobjects(schema["items"], path + ("*",))
        for k, v in child_paths.items():
            hash_to_paths[k].extend(v)
        hash_to_schema.update(child_schemas)

    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            for idx, s in enumerate(schema[key]):
                child_paths, child_schemas = collect_subobjects(s, path + (f"{key}[{idx}]",))
                for k, v in child_paths.items():
                    hash_to_paths[k].extend(v)
                hash_to_schema.update(child_schemas)

    return hash_to_paths, hash_to_schema

def replace_with_refs(schema, definitions, hash_to_defname):
    """
    Recursively replace sub-schemas with $ref to definitions.
    Handles arrays with heterogeneous items safely.
    """
    if isinstance(schema, list):
        return [replace_with_refs(s, definitions, hash_to_defname) for s in schema]

    if not isinstance(schema, dict):
        return schema

    # Already a $ref? return as-is
    if "$ref" in schema and isinstance(schema["$ref"], str):
        return schema

    # Check if this schema matches any definition
    canonical = json.dumps(canonicalize_schema(schema), sort_keys=True)
    for h, def_name in hash_to_defname.items():
        def_canonical = json.dumps(canonicalize_schema(definitions[def_name]), sort_keys=True)
        if canonical == def_canonical:
            return {"$ref": f"#/$defs/{def_name}"}

    # Recurse into object properties
    if schema.get("type") == "object" and "properties" in schema:
        schema["properties"] = {k: replace_with_refs(v, definitions, hash_to_defname)
                                for k, v in schema["properties"].items()}

    # Recurse into arrays
    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = replace_with_refs(schema["items"], definitions, hash_to_defname)

    # Recurse into anyOf / oneOf
    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            schema[key] = [replace_with_refs(s, definitions, hash_to_defname) for s in schema[key]]

    return schema

def collect_used_refs(schema, used_defs):
    """Traverse schema to collect all $ref names used safely."""
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            used_defs.add(ref.split("/")[-1])
        for v in schema.values():
            collect_used_refs(v, used_defs)
    elif isinstance(schema, list):
        for item in schema:
            collect_used_refs(item, used_defs)

def discover_schema_from_paths(paths_dict, path_freqs, num_docs):
    """Infer full schema from paths/values using a single root object."""
    root_schema = {"type": "object", "properties": {}}

    for path, values in paths_dict.items():
        if not values:
            continue
        # Discover schema for all observed values at this path
        value_schemas = [discover_schema(json.loads(v)) for v in values]
        value_schema = reduce(merge_schemas, value_schemas)
        insert_schema_at_path(root_schema, path, value_schema)

    # Add required/additionalProperties based on path frequencies
    root_schema = add_required_and_additional_properties(root_schema, path_freqs)
    # Finalize schema: normalize '*', dedupe anyOf/oneOf, remove temp fields
    full_schema = finalize_schema(root_schema)
    hash_to_paths, hash_to_schema = collect_subobjects(full_schema)
    
    definitions = {}
    hash_to_defname = {}
    for h, paths in hash_to_paths.items():
        if len(paths) >= 2:
            def_name = f"Def{len(definitions)+1}"
            definitions[def_name] = deepcopy(hash_to_schema[h])
            hash_to_defname[h] = def_name

    if definitions:
        full_schema = replace_with_refs(full_schema, definitions, hash_to_defname)

        # Remove unused definitions
        used_defs = set()
        collect_used_refs(full_schema, used_defs)
        definitions = {k: v for k, v in definitions.items() if k in used_defs}

        if definitions:
            full_schema["$defs"] = definitions
    
    return full_schema

def schema_to_key(schema):
    """Return a canonical string for a schema for deduplication."""
    return json.dumps(schema, sort_keys=True)

def process_schema_node(schema, defs, def_counter):
    """
    Recursively process schema node: replace repeated object/array schemas with $ref
    and collect them into $defs (JSON Schema 2020-12).
    """
    if isinstance(schema, dict):
        node_type = schema.get("type")

        if node_type in ("object", "array"):
            key = schema_to_key(schema)
            if key in defs:
                # Already defined: replace with $ref
                return {"$ref": f"#/$defs/{defs[key]['name']}"}, defs, def_counter

            # New definition
            def_name = f"Def{def_counter[0]}"
            def_counter[0] += 1
            defs[key] = {"name": def_name, "schema": deepcopy(schema)}
            return {"$ref": f"#/$defs/{def_name}"}, defs, def_counter

        # Recurse into properties/items/anyOf/oneOf
        new_schema = {}
        for k, v in schema.items():
            if k in ("properties", "items", "anyOf", "oneOf"):
                new_v, defs, def_counter = process_schema_node(v, defs, def_counter)
                new_schema[k] = new_v
            else:
                new_schema[k] = v
        return new_schema, defs, def_counter

    elif isinstance(schema, list):
        new_list = []
        for item in schema:
            new_item, defs, def_counter = process_schema_node(item, defs, def_counter)
            new_list.append(new_item)
        return new_list, defs, def_counter

    else:
        return schema, defs, def_counter

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
        paths_dict, path_freqs, num_docs, _ = process_dataset(dataset)
        inferred_schema = discover_schema_from_paths(paths_dict, path_freqs, num_docs)
        save_schema(inferred_schema, inferred_schema_path)
        return f"Processed {dataset}", True
    except Exception as e:
        return f"Error processing {file}: {e}", False


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