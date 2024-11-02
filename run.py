import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from skmultilearn.adapt import MLkNN
from skmultilearn.dataset import load_dataset
import heapq
from sklearn.utils import resample



# Define random seeds for reproducibility
random_seeds = [42, 24, 56, 78, 91, 35, 17, 83, 65, 12]
num_runs = len(random_seeds)  # Number of runs

def evaluate_multilabel(y_test, y_pred):
    # Calculate Hamming Loss
    hamming = hamming_loss(y_test, y_pred)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate F1 Scores (micro and macro)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate AUC-ROC (macro)
    auc_roc = roc_auc_score(y_test, y_pred, average='macro', multi_class="ovr")
    
    # Calculate AUC-PR (macro)
    auc_pr = average_precision_score(y_test, y_pred, average='macro')
    
    # Calculate F-measure (F1 Score per label)
    precision, recall, f1_per_label, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    fmeasure = f1_per_label.mean()  # Taking the mean as an aggregate F-measure score
    
    # Return results as a dictionary
    results = {
        'Hamming Loss': hamming,
        'Accuracy': accuracy,
        'F1 Micro': f1_micro,
        'F1 Macro': f1_macro,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'F-measure': fmeasure
    }
    
    return results

def mmo(X, y):
    selected_samples = []
    N, num_features = X.shape
    _, M = y.shape

    current_label_counts = np.sum(y, axis=0)
    T = np.max(current_label_counts)
    samples_needed_per_label = T - current_label_counts

    # List-based approach for resampled data
    X_resampled = []
    y_resampled = []

    # Priority queue (max-heap) to store the best candidates
    candidate_heap = []

    sample_costs = np.zeros(N, dtype=int)
    
    # Initialize the heap with the initial scores
    for i in range(N):
        sample_labels = y[i]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        if contribution > 0:
            score = contribution / (sample_costs[i] + 1)
            # Push negative score to simulate a max-heap
            heapq.heappush(candidate_heap, (-score, i))

    while np.any(current_label_counts < T):
        if not candidate_heap:
            break

        # Select the best candidate
        best_score, best_idx = heapq.heappop(candidate_heap)
        best_score = -best_score  # Revert score back to positive

        # Add the selected sample to the resampled dataset
        X_resampled.append(X[best_idx])
        y_resampled.append(y[best_idx])

        # Update label counts and samples needed per label
        current_label_counts += y[best_idx]
        samples_needed_per_label = np.maximum(0, T - current_label_counts)
        
        # Track sample costs to avoid redundant resampling
        sample_costs[best_idx] += 1
        selected_samples.append(best_idx)

        # Update the contribution and score for the selected sample
        sample_labels = y[best_idx]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        if contribution > 0:
            score = contribution / (sample_costs[best_idx] + 1)
            # Push updated score and index back to the heap
            heapq.heappush(candidate_heap, (-score, best_idx))

    
    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)

    X_resampled = np.concatenate((X, X_resampled), axis=0)
    y_resampled = np.concatenate((y, y_resampled), axis=0)

    return X_resampled, y_resampled

def random_oversample(X, Y, random_state=None):
    """
    Perform optimized random oversampling on a multilabel dataset.
    
    Parameters:
    X (pd.DataFrame or np.ndarray): Feature matrix
    Y (pd.DataFrame or np.ndarray): Multilabel binary matrix
    random_state (int, optional): Random seed for reproducibility

    Returns:
    X_resampled (np.ndarray): Resampled feature matrix
    Y_resampled (np.ndarray): Resampled multilabel matrix
    """
    # Ensure consistent DataFrame usage for easier operations
    X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
    Y_df = pd.DataFrame(Y) if isinstance(Y, np.ndarray) else Y

    # Combine features and label combinations for grouping
    data = X_df.copy()
    data['label_combination'] = list(map(tuple, Y_df.values))

    # Get counts for each label combination and the maximum count
    label_counts = data['label_combination'].value_counts()
    max_count = label_counts.max()

    # Prepare a list to collect oversampled subsets
    resampled_data = []

    # Iterate over each unique label combination and resample if necessary
    for label_combo, count in label_counts.items():
        subset = data[data['label_combination'] == label_combo]
        # Perform oversampling only if the count is below the maximum
        if count < max_count:
            oversampled_subset = resample(
                subset,
                replace=True,
                n_samples=max_count,
                random_state=random_state
            )
            resampled_data.append(oversampled_subset)
        else:
            resampled_data.append(subset)

    # Concatenate all resampled data
    resampled_data = pd.concat(resampled_data, ignore_index=True)

    # Separate back into features and labels without extra conversions
    X_resampled = resampled_data.drop(columns=['label_combination']).to_numpy()
    Y_resampled = np.array([list(label) for label in resampled_data['label_combination']])

    return X_resampled, Y_resampled

def no_oversample(X, y):
    return X, y


# Load datasets
datasets = [
    'yeast', 'scene', 'emotions', 'birds', 'cal500',
    'enron', 'mediamill', 'medical'
]

data_dict = {}

for dataset in datasets:
    try:
        # Load training data
        X_train, y_train, _, _ = load_dataset(dataset, 'train')
        # Load test data
        X_test, y_test, _, _ = load_dataset(dataset, 'test')
        
        # Store in dictionary
        data_dict[dataset] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        print(f"Successfully loaded {dataset} dataset.")
    except Exception as e:
        print(f"Error loading {dataset} dataset: {e}")

# Define oversampling methods
oversampling_methods = {
    'mmo': mmo,
    'random': random_oversample,
    'none': no_oversample
}

def process_dataset_with_seed(dataset_name, data, oversampling_methods, seed):
    # Define the file path for this dataset's results
    dataset_csv_path = f"datasets/{dataset_name}_results_seed_{seed}.csv"

    results = []
    X_train = np.asarray(data['X_train'].todense())
    y_train = np.asarray(data['y_train'].todense())
    X_test = np.asarray(data['X_test'].todense())
    y_test = np.asarray(data['y_test'].todense())

    # Set random seed
    np.random.seed(seed)

    # Normalize features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for oversample_name, oversample_func in oversampling_methods.items():
        print(f"Applying {oversample_name} oversampling for {dataset_name} with seed {seed}.")
        X_resampled, y_resampled = oversample_func(X_train, y_train)

        clf_name = "RandomForest"
        classifier = LabelPowerset(classifier=RandomForestClassifier(n_estimators=100, random_state=seed))
        classifier.fit(X_resampled, y_resampled)
        y_pred = classifier.predict(X_test).toarray()

        metrics = evaluate_multilabel(y_test, y_pred)

        # Store results
        row = {
            'Dataset': dataset_name,
            'Oversampling': oversample_name,
            'Classifier': clf_name,
            'Seed': seed
        }
        row.update(metrics)
        results.append(row)

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(dataset_csv_path, index=False)
    print(f"Results for {dataset_name} with seed {seed} saved to {dataset_csv_path}.")
    
    return dataset_csv_path  # Return the path of the generated CSV

def run_experiment_parallel(data_dict, oversampling_methods, random_seeds):
    dataset_csv_paths = []

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submitting each dataset experiment with each seed as a separate job
        futures = [
            executor.submit(process_dataset_with_seed, dataset_name, data, oversampling_methods, seed)
            for dataset_name, data in data_dict.items()
            for seed in random_seeds
        ]

        # Collecting results as they complete
        for future in as_completed(futures):
            dataset_csv_path = future.result()
            dataset_csv_paths.append(dataset_csv_path)
            print(f"Completed processing for {dataset_csv_path}.")

    # Consolidate all individual dataset results into a single DataFrame
    combined_results = pd.concat([pd.read_csv(path) for path in dataset_csv_paths])
    
    # Save consolidated results to a single CSV file
    combined_results.to_csv('datasets/consolidated_results.csv', index=False)
    print("\nConsolidated results saved to 'consolidated_results.csv'.")

    return combined_results

# Run parallel experiments with multiple seeds
results_df = run_experiment_parallel(data_dict, oversampling_methods, random_seeds)