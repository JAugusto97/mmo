# util.py

import numpy as np
import pandas as pd
import heapq
from sklearn.utils import resample
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import NearestNeighbors


def label_density(y):
    """
    Calculate the Label Density of a multilabel dataset.

    Parameters:
    y (numpy.ndarray): Binary matrix of shape (n_samples, n_labels).

    Returns:
    float: Label Density.
    """
    n_samples, n_labels = y.shape
    label_cardinality = np.sum(y) / n_samples
    label_density = label_cardinality / n_labels
    return label_density


def mean_imbalance_ratio(y):
    """
    Calculate the Mean Imbalance Ratio (MeanIR) of a multilabel dataset.

    Parameters:
    y (numpy.ndarray): Binary matrix of shape (n_samples, n_labels).

    Returns:
    float: Mean Imbalance Ratio.
    """
    n_samples, n_labels = y.shape
    label_frequencies = np.sum(y, axis=0)
    max_frequency = np.max(label_frequencies)
    imbalance_ratios = max_frequency / (label_frequencies + 1e-6)
    mean_ir = np.mean(imbalance_ratios)
    return mean_ir


def evaluate_multilabel(y_test, y_pred):
    hamming = hamming_loss(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = 0.0

    precision, recall, f1_per_label, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    fmeasure = f1_per_label.mean()

    return {
        "Hamming Loss": hamming,
        "Accuracy": accuracy,
        "F1 Micro": f1_micro,
        "F1 Macro": f1_macro,
        "AUC-ROC": roc_auc,
        "F-measure": fmeasure,
    }


def mmo(X, y, **kwargs):
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


def random_oversample(X, Y, random_state=None, **kwargs):
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
    data["label_combination"] = list(map(tuple, Y_df.values))

    # Get counts for each label combination and the maximum count
    label_counts = data["label_combination"].value_counts()
    max_count = label_counts.max()

    # Prepare a list to collect oversampled subsets
    resampled_data = []

    # Iterate over each unique label combination and resample if necessary
    for label_combo, count in label_counts.items():
        subset = data[data["label_combination"] == label_combo]
        # Perform oversampling only if the count is below the maximum
        if count < max_count:
            oversampled_subset = resample(
                subset, replace=True, n_samples=max_count, random_state=random_state
            )
            resampled_data.append(oversampled_subset)
        else:
            resampled_data.append(subset)

    # Concatenate all resampled data
    resampled_data = pd.concat(resampled_data, ignore_index=True)

    # Separate back into features and labels without extra conversions
    X_resampled = resampled_data.drop(columns=["label_combination"]).to_numpy()
    Y_resampled = np.array(
        [list(label) for label in resampled_data["label_combination"]]
    )

    return X_resampled, Y_resampled


def no_oversample(X, y, **kwargs):
    return X, y


def ml_smote(X, y, k=3, n_samples=100, **kwargs):
    """
    Apply multilabel SMOTE to generate synthetic samples for a multilabel dataset.

    Parameters:
    X : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The multilabel binary target matrix of shape (n_samples, n_classes).
    k : int, optional
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

        if len(minority_indices) < k + 1:
            # Skip if there's too few minority samples to apply SMOTE
            continue

        # Select only minority samples for this label
        X_minority = X[minority_indices]

        # Fit the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_model.fit(X_minority)
        neighbors = nn_model.kneighbors(X_minority, return_distance=False)[:, 1:]

        for _ in range(n_samples // y.shape[1]):
            idx = np.random.choice(len(X_minority))
            neighbor_idx = np.random.choice(neighbors[idx])

            # Generate a synthetic sample
            lam = np.random.rand()
            X_synthetic = X_minority[idx] + lam * (
                X_minority[neighbor_idx] - X_minority[idx]
            )

            # Add synthetic sample to the dataset
            X_resampled.append(X_synthetic)

            # Create the synthetic label with the current label active
            y_synthetic = np.zeros(y.shape[1])
            y_synthetic[label_idx] = 1
            y_resampled.append(y_synthetic)

    return np.array(X_resampled), np.array(y_resampled)


def mle_nn(X, y, k=3, threshold=0.5, **kwargs):
    """
    Apply MLeNN (Multi-Label Edited Nearest Neighbor) algorithm to filter noisy instances.

    Parameters:
    X : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The multilabel binary target matrix of shape (n_samples, n_classes).
    k_neighbors : int, optional
        Number of nearest neighbors to consider for label consistency.
    threshold : float, optional
        Proportion of neighbors that must share the label for an instance to be considered consistent.

    Returns:
    X_filtered : np.ndarray
        The feature matrix after filtering.
    y_filtered : np.ndarray
        The target matrix after filtering.
    """
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_model.fit(X)

    # Find neighbors for each instance (excluding itself)
    neighbors = nn_model.kneighbors(X, return_distance=False)[:, 1:]

    # List to hold indices of instances to keep
    keep_indices = []

    for i, neighbor_indices in enumerate(neighbors):
        # Get the labels of the neighbors
        neighbor_labels = y[neighbor_indices]

        # Calculate label consistency for each label
        label_agreement = np.mean(neighbor_labels == y[i], axis=0)

        # Check if the instance meets the consistency threshold for each label
        consistent_labels = label_agreement >= threshold

        # Keep the instance if it's consistent for the majority of its labels
        if np.mean(consistent_labels) > threshold:
            keep_indices.append(i)

    # Filter the dataset to keep only the consistent instances
    X_filtered = X[keep_indices]
    y_filtered = y[keep_indices]

    return X_filtered, y_filtered


def ml_ros(X, y, random_state, target_proportion=1.0, **kwargs):
    """
    Apply Multi-Label Random Over-Sampling (ML-ROS) to balance a multilabel dataset.

    Parameters:
    X : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The multilabel binary target matrix of shape (n_samples, n_classes).
    target_proportion : float, optional
        The target frequency proportion of each label relative to the majority label.

    Returns:
    X_resampled : np.ndarray
        The resampled feature matrix after applying ML-ROS.
    y_resampled : np.ndarray
        The resampled target matrix after applying ML-ROS.
    """
    # Calculate label frequencies
    label_counts = np.sum(y, axis=0)
    max_count = np.max(label_counts)
    target_counts = (target_proportion * max_count).round().astype(int) * np.ones_like(
        label_counts
    )

    # Initialize resampled data
    X_resampled, y_resampled = list(X), list(y)

    for label_idx in range(y.shape[1]):
        if label_counts[label_idx] < target_counts[label_idx]:
            # Find instances with this label
            minority_indices = np.where(y[:, label_idx] == 1)[0]
            # Determine the number of samples needed
            num_samples_needed = target_counts[label_idx] - label_counts[label_idx]

            # Only proceed if there are samples needed and available indices to sample from
            if num_samples_needed > 0 and len(minority_indices) > 0:
                # Randomly sample with replacement
                samples_to_add = resample(
                    minority_indices,
                    n_samples=num_samples_needed,
                    replace=True,
                    random_state=random_state,
                )
                # Add resampled instances to the data
                for idx in samples_to_add:
                    X_resampled.append(X[idx])
                    y_resampled.append(y[idx])

    return np.array(X_resampled), np.array(y_resampled)


def mmo_smote(X, y, k=3, **kwargs):
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

    # Set up nearest neighbor model for SMOTE-style interpolation
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(
        X
    )  # k+1 to include the sample itself

    while np.any(current_label_counts < T):
        if not candidate_heap:
            break

        # Select the best candidate
        best_score, best_idx = heapq.heappop(candidate_heap)
        best_score = -best_score  # Revert score back to positive

        # Find the k nearest neighbors for the selected sample (excluding itself)
        _, indices = nn.kneighbors(X[best_idx].reshape(1, -1))
        neighbor_indices = indices[0][1:]  # Exclude the sample itself

        # Generate a synthetic sample with a random neighbor
        neighbor_idx = np.random.choice(neighbor_indices)
        lam = np.random.rand()
        X_synthetic = X[best_idx] + lam * (X[neighbor_idx] - X[best_idx])

        X_resampled.append(X_synthetic)
        y_resampled.append(y[best_idx])  # Label remains the same as the original sample

        # Update label counts and samples needed per label
        current_label_counts += y[best_idx]
        samples_needed_per_label = np.maximum(0, T - current_label_counts)

        # Track sample costs to avoid redundant resampling
        sample_costs[best_idx] += 1
        selected_samples.append(best_idx)

        # Update the contribution and score for the selected sample if still needed
        sample_labels = y[best_idx]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        if contribution > 0:
            score = contribution / (sample_costs[best_idx] + 1)
            # Push updated score and index back to the heap
            heapq.heappush(candidate_heap, (-score, best_idx))

    # Convert resampled lists to arrays
    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)

    # Concatenate the original dataset with the synthetic samples
    X_resampled = np.concatenate((X, X_resampled), axis=0)
    y_resampled = np.concatenate((y, y_resampled), axis=0)

    return X_resampled, y_resampled


def mmo_mle_nn(X, y, k=3, consistency_threshold=0.5, **kwargs):
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

    # Calculate nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Helper function to calculate label consistency
    def label_consistency(sample_idx):
        neighbor_labels = y[indices[sample_idx]]
        consistency = (neighbor_labels * y[sample_idx]).sum(axis=1) / (
            y[sample_idx].sum() + 1
        )
        return (consistency > consistency_threshold).mean()

    # Initialize the heap with the initial scores and MLeNN criteria
    for i in range(N):
        sample_labels = y[i]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        consistency = label_consistency(i)

        # Only consider samples with sufficient label consistency
        if contribution > 0 and consistency >= consistency_threshold:
            score = (contribution / (sample_costs[i] + 1)) * consistency
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
        consistency = label_consistency(best_idx)

        if contribution > 0 and consistency >= consistency_threshold:
            score = (contribution / (sample_costs[best_idx] + 1)) * consistency
            # Push updated score and index back to the heap
            heapq.heappush(candidate_heap, (-score, best_idx))

    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)

    X_resampled = np.concatenate((X, X_resampled), axis=0)
    y_resampled = np.concatenate((y, y_resampled), axis=0)

    return X_resampled, y_resampled
