import csv
import fasttext
import itertools
import json
import math
import numpy as np
import os
import pandas as pd
import re
import statistics
#mport statsmodels.api as sm
import sys
import time

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler
from itertools import chain, combinations, islice
from scipy.spatial.distance import cosine
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


# Create constant variables
DISTINCT_SUBKEYS_UPPER_BOUND = 1000
# Load the fasttext model
ft_model = fasttext.load_model("./fastText/cc.en.300.bin")


def extract_paths(doc, paths = [], types = [], num_nested_keys = []):
    '''
    Input: json document, list of keys full path, list of keys' datatypes, number of paths nested keys
    Output: path of each key and subkey
    Purpose: Get the path of each key and its value from the json documents
    '''
    if isinstance(doc, dict):
        iterator = doc.items()
    elif isinstance(doc, list):
        if len(doc) > 0:
            #iterator = [('*', doc[0])]
            iterator = []
        else:
            iterator = []
    else:
        raise ValueError('Invalid type')
  
    for (key, value) in iterator:
        yield paths + [key], value, len(value) if isinstance(value, (dict, list))  else 0
        if isinstance(value, (dict, list)):
            yield from extract_paths(value, paths + [key], types, num_nested_keys)
    
    
def extract_key_vectors(df, f, key, nested_keys, ft_model):
    '''
    Input: dataframe, file the key belongs to, a key or cluster of keys and its unique nested keys 
    Output: dataframe
    Purpose: Calculate a new feature called mean vectors and add it to the dataframe
    '''

    key_vectors = {k: ft_model.get_word_vector(k) for k in nested_keys}
    
    if len(nested_keys) > 1:
        pairs = list(itertools.combinations(nested_keys, 2))
        sims = [cosine(key_vectors[key1], key_vectors[key2]) for (key1, key2) in pairs]
        mean_sim = np.mean(sims)
    else:
        mean_sim = np.mean(list(key_vectors.values())[0])
    
    condition = (df.Path.isin(key)) if isinstance(key, list) else (df.Path == key)
    df.loc[condition & (df.Filename == f), "Mean_vectors"] = mean_sim
    
    return df


def is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs):
    # Calculate datatype entropy
    for (key, value) in prefix_types_dict.items():
        # Check if values of the nested keys of a set have the same datatype
        #result = all(isinstance(nested_key, type(value[0])) for nested_key in value[1:])
        df.loc[df.Path == key, "Datatype_entropy"] = 0 if all(value) else 1
            
    # Calculate key entropy and add as a new column
    for (key, value) in prefix_nested_keys_freq_dict.items():
        key_entropy = 0
        for freq in value:
            key_entropy += (freq/num_docs) * math.log(freq/num_docs)
        df.loc[df.Path == key, "Key_entropy"] = -key_entropy
    return df


def update_moments(prefix_nested_keys_stats_dict, path, nested_keys_count):
    '''
    Input: number of nested keys
    Purpose: calculate statistics efficiently to reduce memory usage
    '''
    n = prefix_nested_keys_stats_dict[path]["n"]

    # Get the moments
    M1 = prefix_nested_keys_stats_dict[path]["M1"]
    M2 = prefix_nested_keys_stats_dict[path]["M2"]
    M3 = prefix_nested_keys_stats_dict[path]["M3"]
    M4 = prefix_nested_keys_stats_dict[path]["M4"]
    min = prefix_nested_keys_stats_dict[path]["min"]
    max = prefix_nested_keys_stats_dict[path]["max"]

    n1 = n
    n += 1
    delta = nested_keys_count - M1
    delta_n = delta / n
    delta_n2 = delta_n * delta_n
    term1 = delta * delta_n * n1

    # Update the moments
    M1 += delta_n
    M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
    M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
    M2 += term1
    
    # Calculate min and max 
    if nested_keys_count < min:
        min = nested_keys_count
    if nested_keys_count > max:
        max = nested_keys_count

    # Update dictionary
    prefix_nested_keys_stats_dict[path]["n"] = n
    prefix_nested_keys_stats_dict[path]["M1"] = M1
    prefix_nested_keys_stats_dict[path]["M2"] = M2
    prefix_nested_keys_stats_dict[path]["M3"] = M3
    prefix_nested_keys_stats_dict[path]["M4"] = M4
    prefix_nested_keys_stats_dict[path]["min"] = min
    prefix_nested_keys_stats_dict[path]["max"] = max


def process_document(doc, freq_paths_dict, prefix_paths_dict, prefix_types_dict, prefix_nested_keys_stats_dict):
    new_obj = {"$": doc}
    
    for (path, value, nested_keys_count) in extract_paths(new_obj):
        path = tuple(path)
        freq_paths_dict[path] += 1
        
        if len(path) > 1:
            prefix = path[:-1]
            prefix_paths_dict[prefix].add(path)
            
        prefix_types_dict[path].add(type(value).__name__)
        update_moments(prefix_nested_keys_stats_dict, path, nested_keys_count)


def calculate_relative_frequencies(freq_paths_dict, num_docs, percentages):
    for (path, freq) in freq_paths_dict.items():
        if len(path) > 1:
            prefix = path[:-1]
            #prefix_nested_keys_freq_dict[prefix].append(freq)
            
            relative_freq = freq / freq_paths_dict[prefix]
            assert 0 <= relative_freq <= 1
            percentages.append(relative_freq)
        elif len(path) == 1:
            relative_freq = freq / num_docs
            assert 0 <= relative_freq <= 1
            percentages.append(relative_freq)

    
def parse_dataset(doc_list, dataset_name):
    num_docs = len(doc_list)
    
    freq_paths_dict = defaultdict(lambda: 0)
    prefix_paths_dict = defaultdict(set)
    prefix_types_dict = defaultdict(set)
    prefix_nested_keys_freq_dict = defaultdict(list)
    prefix_nested_keys_stats_dict = defaultdict(lambda: {"n": 0, "M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0, "min": 0.0, "max": 0.0})
    prefix_freqs_dict = defaultdict(lambda: 0)
    
    for doc in doc_list:
        process_document(doc, freq_paths_dict, prefix_paths_dict, prefix_types_dict, prefix_nested_keys_stats_dict)
    
    for(prefix, p_list) in prefix_paths_dict.items(): 
        prefix_freqs_dict[prefix] = [freq_paths_dict[p] for p in p_list]

    percentages = []
    calculate_relative_frequencies(freq_paths_dict, num_docs, percentages)
    
    paths = list(freq_paths_dict.keys())
    nesting_levels = [len(path) for path in paths]
    
    merged_list = list(zip(paths, percentages, nesting_levels))
    df = pd.DataFrame(merged_list, columns=["Path", "Percentage", "Nesting_level"])
    
    populate_dataframe(df, prefix_paths_dict, prefix_types_dict, prefix_nested_keys_stats_dict, prefix_nested_keys_freq_dict, prefix_freqs_dict)
    df["Filename"] = dataset_name
    
    return df

"""
def parse_dataset(doc_list, dataset_name):
    '''
    Input: doc_list, dataset name
    Output: dataframe
    Purpose: Create a dataframe of intrinsic features for each dataset
    '''
    num_docs = 0
    freq_paths_dict = defaultdict(lambda: 0)
    prefix_paths_dict = defaultdict(set)
    prefix_types_dict = defaultdict(set)
    prefix_nested_keys_freq_dict = defaultdict(list)
    prefix_nested_keys_stats_dict = defaultdict(lambda: {"n": 0, "M1": 0.0, "M2": 0.0, "M3": 0.0, "M4": 0.0, "min": 0.0, "max": 0.0})
    
    for doc in doc_list:
        num_docs += 1
        new_obj = {"$": doc}
        # Get all the keys (cast to tuple), values, and types
        for (path, value, nested_keys_count) in extract_paths(new_obj):
            path = tuple(path)
            # Count the occurrence of each path and its subpaths datatype and store results in dictionary
            freq_paths_dict[path] += 1
            # Check if the path has nested keys
            if len(path) > 1:
                prefix = path[:-1]
                prefix_paths_dict[prefix].add(path)
            prefix_types_dict[path].add(type(value).__name__)
            # Count the number of nested keys a path has
            update_moments(prefix_nested_keys_stats_dict, path, nested_keys_count)
    prefix_freqs_dict = {}
    # Count the total number of keys with a given prefix
    for(prefix, p_list) in prefix_paths_dict.items(): 
        prefix_freqs_dict[prefix] = [freq_paths_dict[p] for p in p_list]

    paths, percentages, nesting_levels = [],[],[]

    # Loop through the frequency path dictionary and record the frequency of each 
    for (path, freq) in freq_paths_dict.items():
        paths.append(path)
        nesting_levels.append(len(path))
        # Check if the delimiter is in a path
        if len(path) > 1:
            # Get all the elements except for the last one
            prefix = path[:-1]
            prefix_nested_keys_freq_dict[prefix].append(freq)
    
            # Get the frequency of that path and divide it by its immediate parent's frequency as a percent
            relative_freq =  freq / freq_paths_dict[prefix]
            assert relative_freq >= 0 and relative_freq <= 1
            percentages.append(relative_freq)

        elif len(path) == 1:
            relative_freq = freq / num_docs
            assert relative_freq >= 0 and relative_freq <= 1
            percentages.append(relative_freq)
        
    # Zip the keys, percentages, and nesting levels lists and cast to list
    merged_list = list(zip(paths, percentages, nesting_levels))

    # Convert list to a dataframe
    df = pd.DataFrame(merged_list, columns = ["Path", "Percentage", "Nesting_level"])
    #df.to_csv(dataset_name + ".csv")
    #df["Datatype_entropy"] = np.nan
    #df["Key_entropy"] = np.nan
    # Calculate the Jxplain entropy
    #df = is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs)
    # Populate dataframe
    df = populate_dataframe(df, prefix_paths_dict, prefix_types_dict, prefix_nested_keys_stats_dict, prefix_freqs_dict)
    # Add a column for the filename
    df["Filename"] = dataset_name
    # Return dataframe
    return df
"""

def populate_dataframe(df, prefix_paths_dict, prefix_types_dict, prefix_nested_keys_stats_dict, prefix_nested_keys_freq_dict, prefix_freqs_dict):
    '''
    Input: paths = json filename, list of paths, frequency of paths, num_docs = the number of documents in a file
    Output: dataframe of descritive statistics
    Purpose: Create a dataframe for each file that contains descriptive statistics about the json keys in that file
    '''
    # Create new columns for simple descriptive stats
    df["Mean"], df["Range"], df["Standard_deviation"], df["Kurtosis"], df["Skewness"] = None, None, None, None, None
    # Loop over the dictionary
    for path in df.Path:
        # Get the location of the key within the dataframe
        key_loc = df.loc[df.Path == path].index[0]
        
        # Get moments to calculate simple descriptive stats
        n = prefix_nested_keys_stats_dict[path]["n"]
        M1 = prefix_nested_keys_stats_dict[path]["M1"]
        M2 = prefix_nested_keys_stats_dict[path]["M2"]
        M3 = prefix_nested_keys_stats_dict[path]["M3"]
        M4 = prefix_nested_keys_stats_dict[path]["M4"]
        min = prefix_nested_keys_stats_dict[path]["min"]
        max = prefix_nested_keys_stats_dict[path]["max"]

        # Calculate the mean, variance, standard deviation, skewness and kurtosis
        mean, variance, standard_deviation, skewness, kurtosis = M1, 0, 0, 0, 0

        try:
            variance = M2 / (n - 1.0)
            standard_deviation = math.sqrt(variance)
            skewness = math.sqrt(n) * M3 / math.pow(M2, 1.5)
            kurtosis = n * M4 / (M2 * M2) - 3.0
        except ZeroDivisionError as zde:
            skewness = 0.0
            kurtosis = 0.0

        # Add list of values for simple descriptive statistics
        df.at[key_loc, "Mean"] = mean
        df.at[key_loc, "Range"] = max - min
        df.at[key_loc, "Standard_deviation"] = standard_deviation
        df.at[key_loc, "Kurtosis"] = kurtosis
        df.at[key_loc, "Skewness"] = skewness
    
    # Create new columns for complex descriptive stats
    df["Mean_vectors"], df["Distinct_subkeys"], df["Distinct_subkeys_datatypes"] = None, None, None

    # Calculate complex descriptive statistics
    for path in prefix_freqs_dict.keys():
        # Get the location of the key within the dataframe
        key_loc = df.loc[df.Path == path].index[0]

        # Get the number of unique subkeys under a parent key
        distinct_subkeys = set()
        for k in prefix_paths_dict[path]:
            distinct_subkeys.add(k[-1])
            if len(distinct_subkeys) > DISTINCT_SUBKEYS_UPPER_BOUND:
                break
            
        df.at[key_loc, "Mean_vectors"] = distinct_subkeys
        df.at[key_loc, "Distinct_subkeys"] = list(distinct_subkeys)
        df.at[key_loc, "Distinct_subkeys_datatypes"] = list(set(prefix_types_dict[path]))

    #df = df[~df['Distinct_subkeys'].apply(check_for_asterisks)]
    # Calculate the Jxplain entropy
    #df = is_tuple_or_collection(df, prefix_types_dict, prefix_nested_keys_freq_dict, num_docs)
    # Remove rows of keys that have no nested keys
    df = df[df.Distinct_subkeys.notnull()]
    df.dropna(inplace=True)
    return df


def calculate_nested_keys_stats(df, all_cluster_keys):
    '''
    Input: dataframe, clusters
    Output: dataframe
    Purpose: Calculate statistics without grouping
    '''
    for k, f in zip(df.Path, df.Filename):
        if (k, f) in all_cluster_keys:
            continue
        nested_keys = df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys"].values[0]
        if nested_keys is None:
            continue
        extract_key_vectors(df, f, k, nested_keys, ft_model)
        df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys"] = len(nested_keys)
        df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys_datatypes"] = len(df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys_datatypes"].values[0])
    return df


def group_keys(df):
    '''
    Input: dataframe, paths frequencies dict, and count of nested keys dict
    Output: Updated dataframe
    Purpose: Calculate and replace single statistics with group values
    '''
    groups_freq = defaultdict(list)
    df["Cluster"] = 99999
    all_cluster_keys = set()
    
    # Open file in read mode
    with open('clusters.json') as file:
        for line in file:
            obj = json.loads(line)
            f = obj["filename"]
            cluster = obj["properties"]
            # Create lists for the average of all the values belonging to each feature
            cluster_frequency = []
            cluster_nesting_level = []
            cluster_mean = []
            cluster_range = []
            cluster_standard_deviation = []
            cluster_skewness = []
            cluster_kurtosis = []
            #cluster_mean_vectors = []
            cluster_distinct_subkeys = []
            cluster_distinct_subkeys_datatypes = []

            # Convert List of Lists to Tuple of Tuples
            #res = tuple(tuple(sub) for sub in cluster)
            path = cluster[0]
            # Update descriptive statistics
            if f in list(df.Filename) and path in list(df.Path):
                for key_path in cluster:
                    cluster_frequency.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Percentage"].values[0])
                    cluster_nesting_level.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Nesting_level"].values[0])
                    cluster_mean.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Mean"].values[0])
                    cluster_range.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Range"].values[0])
                    cluster_standard_deviation.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Standard_deviation"].values[0])
                    cluster_skewness.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Skewness"].values[0])
                    cluster_kurtosis.append(df.loc[(df.Path == key_path) & (df.Filename == f), "Kurtosis"].values[0])
                    cluster_distinct_subkeys.extend(df.loc[(df.Path == key_path) & (df.Filename == f), "Distinct_subkeys"].values[0])
                    cluster_distinct_subkeys_datatypes.extend(df.loc[(df.Path == key_path) & (df.Filename == f), "Distinct_subkeys_datatypes"].values[0])

                    # Add the key path to the list
                    all_cluster_keys.add((key_path, f))

                # Remove duplicates
                cluster_distinct_subkeys = list(set(cluster_distinct_subkeys))
                cluster_distinct_subkeys_datatypes = list(set(cluster_distinct_subkeys_datatypes))
                      
                # Calculate the averages for the above lists and store them as values of a dictionary and group as keys
                groups_freq[(f, cluster)] = [cluster_frequency, cluster_nesting_level, cluster_mean, cluster_range, cluster_standard_deviation, cluster_skewness, cluster_kurtosis, cluster_distinct_subkeys, cluster_distinct_subkeys_datatypes]
            else:
                continue
    if len(groups_freq) == 0:
        return df, all_cluster_keys
    # Incorporate grouped keys statistics
    for cluster_number, ((f, keys), value) in enumerate(groups_freq.items()):
        for k in keys:
            df.loc[(df.Path == k) & (df.Filename == f), "Percentage"] = statistics.mean(value[0])
            df.loc[(df.Path == k) & (df.Filename == f), "Nesting_level"] = statistics.mean(value[1])
            df.loc[(df.Path == k) & (df.Filename == f), "Mean"] = statistics.mean(value[2])
            df.loc[(df.Path == k) & (df.Filename == f), "Range"] = max(value[3]) - min(value[3])
            df.loc[(df.Path == k) & (df.Filename == f), "Standard_deviation"] = statistics.mean(value[4])
            df.loc[(df.Path == k) & (df.Filename == f), "Skewness"] = statistics.mean(value[5])
            df.loc[(df.Path == k) & (df.Filename == f), "Kurtosis"] = statistics.mean(value[6])
            df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys"] = len(value[7])
            df.loc[(df.Path == k) & (df.Filename == f), "Distinct_subkeys_datatypes"] = len(value[8])
            df.loc[(df.Path == k) & (df.Filename == f), "Cluster"] = cluster_number
        
        # Calculate the word embedding of the nested keys under keys within a cluster
        df = extract_key_vectors(df, f, list(keys), value[7])
    return df, all_cluster_keys


def find_best_features_combination(columns):
    '''
    Purpose: Come up with all the possible combinations that can be made with all the predictors (features)
    '''
    features = list(columns)
    return chain.from_iterable(combinations(features, i) for i in range(6, len(features) + 1)) 


def print_misclassified_keys(df, test_idx, y_pred, y_test, afile):
    '''
    Input: dataframe, index of the test split, predictions, test set, file
    Purpose: Write to a file keys that are mis-classified
    '''

    for (index, (prediction, label)) in enumerate(zip(y_pred, y_test)):
        if prediction != label:
            df_index = test_idx[index]
            afile.write(str(df.iloc[df_index]["Filename"]) + " " + str( df.iloc[df_index]["Path"]) + " is classified as " + str(prediction) + " but should be " + str(label) + "\n")


def extract_pattern_properties_parents(schema, path=[]):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "patternProperties":
                if 'type' in schema:
                    yield path + [key]
            if isinstance(value, (dict, list)):
                yield from extract_pattern_properties_parents(value, path + [key])
    elif isinstance(schema, list):
        for index, item in enumerate(schema):
            #extract_pattern_properties_parents(item, path + [str(index)])
            yield from extract_pattern_properties_parents(item, path)


def find_dynamic_paths(obj, path=""):
    if isinstance(obj, dict):
        if 'properties' in obj:
            for (k, v) in obj['properties'].items():
                yield from find_dynamic_paths(v, path + "." + k)
        elif 'patternProperties' in obj:
            if len(obj['patternProperties']) == 1:
                yield path
                yield from find_dynamic_paths(next(iter(obj['patternProperties'].values())), path + ".*")
    elif isinstance(obj, list):
        for (i, v) in enumerate(obj):
            yield from find_dynamic_paths(v, path + "[" + str(i) + "]")


def compare_tuples(tuple1, tuple2):
    if len(tuple1) != len(tuple2):
        return False

    for item1, item2 in zip(tuple1, tuple2):
        if isinstance(item2, str) and re.match(item2, item1):
            continue
        elif item1 != item2:
            return False
    return True


def label_paths(dataset, df, dynamic_paths, keys_to_remove):
    # Label static and dynamic paths
    df["Category"] = 0

    for key_path in dynamic_paths:
        key_path = ["$"] + key_path
        if "items" in key_path:
            continue
        #print(key_path)
        key_path = tuple([i for i in key_path if i not in keys_to_remove])
        #print(key_path)

        for index, row in df.iterrows():
            if compare_tuples(row["Path"], key_path) and row["Filename"] == dataset:
                df.at[index, "Category"] = 1
    return df


def preprocess_data(files_folder):
    keys_to_remove = ["definitions", "$defs", "properties", "patternProperties", "oneOf", "allOf", "anyOf", "items"]
    frames = []

    for dataset in os.listdir(files_folder):
        schema_path = os.path.join("schemas", dataset)
        with open(schema_path, 'r') as schema_file:
            json_schema = json.load(schema_file)

        doc_list = []
        file_path = os.path.join("fake_files", dataset)
        with open(file_path, 'r') as file:
            lines = islice(file, 50)
            for line in lines:
                json_doc = json.loads(line)
                doc_list.append(json_doc)

        dynamic_paths = list(extract_pattern_properties_parents(json_schema))
        if not dynamic_paths:
            continue
        
        sys.stderr.write(f"Creating a dataframe for {dataset}\n")
        df = parse_dataset(doc_list, dataset)
        sys.stderr.write(f"Calculating features for {dataset}\n")
        df = calculate_nested_keys_stats(df, set())
        sys.stderr.write(f"Labeling data for {dataset}\n")
        df = label_paths(dataset, df, dynamic_paths, keys_to_remove)
        frames.append(df)

    sys.stderr.write('Merging dataframes...\n')
    df = pd.concat(frames, ignore_index = True)
    return df


def choose_classifier(X, y, df, features, writer):
    '''
    Input: X = Array of features values, y = dependent values, columns = headers of dataframe, df = merged dataframe
    Output: classification results
    Purpose: categorize keys with nested values as dynamic or static
    '''
    lr_precisions = []
    lr_recalls = []
    lr_f1_scores = []
    rf_precisions = []
    rf_recalls = []
    rf_f1_scores = []
    svm_precisions = []
    svm_recalls = []
    svm_f1_scores = []
    mlp_precisions = []
    mlp_recalls = []
    mlp_f1_scores = []
    #jx_precisions = [], jx_recalls = [], jx_f1_scores = []

    logo = LeaveOneGroupOut()
    groups, _ = pd.factorize(df['Filename'])
    scaler = MinMaxScaler()

    lr = LogisticRegression(random_state = 42)
    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    svm = SVC() 
    mlp = MLPClassifier(hidden_layer_sizes = (100, 50), max_iter = 1000, random_state = 42) 
 

    # Split the data into groups for testing and training
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocess data
        ros = RandomOverSampler(sampling_strategy = 0.7)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test) 
        
        # Perform logistic regression
        result = lr.fit(X_train_scaled, y_train_resampled)
        y_pred = (result.predict(X_test_scaled) >= 0.5).astype(int)
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
        lr_precisions.append(precision)
        lr_recalls.append(recall)
        lr_f1_scores.append(f1_score)
        
        # Perform random forest
        result = rf.fit(X_train_scaled, y_train_resampled)
        y_pred = (result.predict(X_test_scaled) >= 0.5).astype(int)
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
        rf_precisions.append(precision)
        rf_recalls.append(recall)
        rf_f1_scores.append(f1_score)

        # Perform support vector machines                           
        result = svm.fit(X_train_scaled, y_train_resampled)
        y_pred = (result.predict(X_test_scaled) >= 0.5).astype(int)     
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
        svm_precisions.append(precision)
        svm_recalls.append(recall)
        svm_f1_scores.append(f1_score)

        # Perform MLP                        
        result = mlp.fit(X_train_scaled, y_train_resampled)
        y_pred = (result.predict(X_test_scaled) >= 0.5).astype(int)     
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average = None, labels = [1])
        mlp_precisions.append(precision)
        mlp_recalls.append(recall)
        mlp_f1_scores.append(f1_score)
            
    sys.stderr.write(f" Writing to file results with the features: {features} \n")
    writer.writerow(["LR", features, np.mean(lr_precisions), np.mean(lr_recalls), np.mean(lr_f1_scores), np.std(lr_f1_scores)])
    writer.writerow(["RF", features, np.mean(rf_precisions), np.mean(rf_recalls), np.mean(rf_f1_scores), np.std(rf_f1_scores)])
    writer.writerow(["SVM", features, np.mean(svm_precisions), np.mean(svm_recalls), np.mean(svm_f1_scores), np.std(svm_f1_scores)])    
    writer.writerow(["MLP", features, np.mean(mlp_precisions), np.mean(mlp_recalls), np.mean(mlp_f1_scores), np.std(mlp_f1_scores)])   
    

def get_features(df, writer):
    """
    Input: Merged dataframe, writer object
    Output: Classifier results and summary
    Purpose: Perform ML classifier model to identify dynamic keys
    """

    df = df.reset_index(drop = True)
    df[["Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]] = df[["Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]].astype(float)
    training_features = ["Percentage", "Nesting_level", "Mean", "Range", "Standard_deviation", "Skewness", "Kurtosis", "Mean_vectors", "Distinct_subkeys", "Distinct_subkeys_datatypes"]
    #training_features = ["Datatype_entropy", "Key_entropy"]
    X = df.loc[:, training_features].values
    y = df.Category

    # Perform classification using the best features/predictors
    #choose_classifier(X, y, df, training_features, writer)

    
    columns = df.loc[:, training_features].columns
    ## Loop over the powerset to get all combinations of features
    for i, subset in enumerate(find_best_features_combination(columns)):
        if len(subset) > 0:
            X = df.loc[:, list(subset)].values
            sys.stderr.write(f"{str(i)} - Classifying using the following columns: {subset}\n")
            choose_classifier(X, y, df, list(subset), writer)



def main():
    dataset_folder, dynamic_keys_folder = sys.argv[-2:]

    pd.set_option('display.max_rows', None)
    df = preprocess_data(dataset_folder)
    df.dropna(inplace=True)
    df.to_csv("df.csv")
    #print(df)
    
    sys.stderr.write("Classifying data...\n")
    f = open("pattern_no_grouping.csv", "a")
    writer = csv.writer(f)
    writer.writerow(["Model", "Features", "Precision", "Recall", "f1_scores", "Standard_deviation"])
    get_features(df, writer)
    
 
if __name__ == '__main__':
    main()
