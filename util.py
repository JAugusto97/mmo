# util.py

import numpy as np
import pandas as pd
import heapq
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support
)
from sklearn.neighbors import NearestNeighbors
import time

def evaluate_multilabel(y_test, y_pred):
    hamming = hamming_loss(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred, average='macro', multi_class="ovr")
    auc_pr = average_precision_score(y_test, y_pred, average='macro')
    precision, recall, f1_per_label, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    fmeasure = f1_per_label.mean()
    
    return {
        'Hamming Loss': hamming,
        'Accuracy': accuracy,
        'F1 Micro': f1_micro,
        'F1 Macro': f1_macro,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'F-measure': fmeasure
    }

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

def ml_smote(X, y, k_neighbors=5, n_samples=100):
    """
    Apply multilabel SMOTE to generate synthetic samples for a multilabel dataset.

    Parameters:
    X : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The multilabel binary target matrix of shape (n_samples, n_classes).
    k_neighbors : int, optional
        Number of nearest neighbors to consider for synthetic sample generation.
    n_samples : int, optional
        Number of synthetic samples to generate.

    Returns:
    X_resampled : np.ndarray
        The resampled feature matrix including synthetic samples.
    y_resampled : np.ndarray
        The resampled target matrix including synthetic samples.
    """
    X_resampled, y_resampled = list(X), list(y)
    
    for label_idx in range(y.shape[1]):
        # Identify samples for the current label
        minority_indices = np.where(y[:, label_idx] == 1)[0]
        
        if len(minority_indices) < 2:
            # Skip if there's too few minority samples to apply SMOTE
            continue
        
        # Select only minority samples for this label
        X_minority = X[minority_indices]
        
        # Fit the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn_model.fit(X_minority)
        neighbors = nn_model.kneighbors(X_minority, return_distance=False)[:, 1:]
        
        for _ in range(n_samples // y.shape[1]):
            idx = np.random.choice(len(X_minority))
            neighbor_idx = np.random.choice(neighbors[idx])
            
            # Generate a synthetic sample
            lam = np.random.rand()
            X_synthetic = X_minority[idx] + lam * (X_minority[neighbor_idx] - X_minority[idx])
            
            # Add synthetic sample to the dataset
            X_resampled.append(X_synthetic)
            
            # Create the synthetic label with the current label active
            y_synthetic = np.zeros(y.shape[1])
            y_synthetic[label_idx] = 1
            y_resampled.append(y_synthetic)
    
    return np.array(X_resampled), np.array(y_resampled)