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

    # Current label counts
    current_label_counts = np.sum(y, axis=0)

    # Target label counts
    T = np.max(current_label_counts)
    samples_needed_per_label = T - current_label_counts

    # Initialize resampled datasets
    X_resampled = X.copy()
    y_resampled = y.copy()

    # Initialize sample costs
    sample_costs = np.zeros(N, dtype=int)

    # Keep track of how many times each sample has been selected
    sample_selection_counts = np.zeros(N, dtype=int)

    # While there are labels that need more samples
    while np.any(samples_needed_per_label > 0):
        # List to hold candidate samples and their scores
        candidates = []

        for i in range(N):
            sample_labels = y[i]
            # Potential label counts if we add this sample
            potential_counts = current_label_counts + sample_labels
            # Check if adding this sample exceeds any target counts
            if np.any(potential_counts > T):
                continue
            # Calculate the contribution
            contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
            if contribution > 0:
                # Calculate the score
                score = (sample_costs[i] + 1) / contribution
                candidates.append((score, i))

        if not candidates:
            break

        # Select the sample with the lowest score
        candidates.sort()
        best_score, best_idx = candidates[0]

        # Add the sample
        X_resampled = np.vstack((X_resampled, X[best_idx:best_idx+1]))
        y_resampled = np.vstack((y_resampled, y[best_idx:best_idx+1]))

        # Update counts
        current_label_counts += y[best_idx]
        samples_needed_per_label = T - current_label_counts

        # Update sample cost and selection count
        sample_selection_counts[best_idx] += 1
        sample_costs[best_idx] = sample_selection_counts[best_idx]

        selected_samples.append(best_idx)

    return X_resampled, y_resampled

# Random oversampling function
def random_oversample(X, y):
    N, num_features = X.shape
    _, M = y.shape

    # Current label counts
    current_label_counts = np.sum(y, axis=0)

    # Target label counts
    T = np.max(current_label_counts)
    samples_needed_per_label = T - current_label_counts

    # Indices of samples for each label
    label_indices = [np.where(y[:, i] == 1)[0] for i in range(M)]

    X_resampled = X.copy()
    y_resampled = y.copy()

    for i in range(M):
        indices = label_indices[i]
        samples_needed = samples_needed_per_label[i]
        if samples_needed > 0 and len(indices) > 0:
            # Randomly sample with replacement
            sampled_indices = np.random.choice(indices, size=samples_needed, replace=True)
            X_resampled = np.vstack((X_resampled, X[sampled_indices]))
            y_resampled = np.vstack((y_resampled, y[sampled_indices]))

    # Verify label counts after oversampling
    current_label_counts = np.sum(y_resampled, axis=0)

    return X_resampled, y_resampled

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

# Define classifiers
classifiers = {
    # Random Forest Classifiers
    'BinaryRelevance-RF': BinaryRelevance(classifier=RandomForestClassifier(n_estimators=100, random_state=42)),
    'ClassifierChain-RF': ClassifierChain(classifier=RandomForestClassifier(n_estimators=100, random_state=42)),
    'LabelPowerset-RF': LabelPowerset(classifier=RandomForestClassifier(n_estimators=100, random_state=42)),
    
    # k-Nearest Neighbors Classifiers
    'BinaryRelevance-kNN': BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=5)),
    'ClassifierChain-kNN': ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=5)),
    'LabelPowerset-kNN': LabelPowerset(classifier=KNeighborsClassifier(n_neighbors=5)),
    
    # Naive Bayes Classifiers (MultinomialNB)
    'BinaryRelevance-NB': BinaryRelevance(classifier=MultinomialNB()),
    'ClassifierChain-NB': ClassifierChain(classifier=MultinomialNB()),
    'LabelPowerset-NB': LabelPowerset(classifier=MultinomialNB()),
    
    # Support Vector Machine Classifiers (using linear kernel)
    'BinaryRelevance-SVM': BinaryRelevance(classifier=SVC(kernel='linear', probability=True, random_state=42)),
    'ClassifierChain-SVM': ClassifierChain(classifier=SVC(kernel='linear', probability=True, random_state=42)),
    'LabelPowerset-SVM': LabelPowerset(classifier=SVC(kernel='linear', probability=True, random_state=42)),
}

def process_dataset_with_seed(dataset_name, data, oversampling_methods, classifiers, seed):
    # Define the file path for this dataset's results
    dataset_csv_path = f"{dataset_name}_results_seed_{seed}.csv"

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

        for clf_name, clf in classifiers.items():
            print(f"Training classifier: {clf_name} on {dataset_name} with {oversample_name} oversampling and seed {seed}.")
            classifier = clf
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

def run_experiment_parallel(data_dict, oversampling_methods, classifiers, random_seeds):
    dataset_csv_paths = []

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submitting each dataset experiment with each seed as a separate job
        futures = [
            executor.submit(process_dataset_with_seed, dataset_name, data, oversampling_methods, classifiers, seed)
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
    combined_results.to_csv('consolidated_oversampling_experiment_results.csv', index=False)
    print("\nConsolidated results saved to 'consolidated_oversampling_experiment_results.csv'.")

    return combined_results

# Run parallel experiments with multiple seeds
results_df = run_experiment_parallel(data_dict, oversampling_methods, classifiers, random_seeds)