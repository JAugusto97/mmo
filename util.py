# util.py

import numpy as np
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
from collections import Counter


class FeatureSelectorByFrequency:
    """
    A feature selector class that selects the top N% of features based on
    the frequency of non-zero values in a dataset.
    """

    def __init__(self, percentage):
        """
        Initializes the selector with the percentage of top features to retain.

        Parameters:
        - percentage: The percentage of top features to retain (e.g., 0.1 for 10%).
        """
        self.percentage = percentage
        self.selected_features_indices = None

    def fit(self, X):
        """
        Fits the selector to the dataset by determining the top features.

        Parameters:
        - X: A NumPy array where rows are samples/documents and columns are features/terms.
        """
        feature_nonzero_counts = np.count_nonzero(X, axis=0)

        num_features_to_select = int(np.ceil(self.percentage * X.shape[1]))

        self.selected_features_indices = np.argsort(feature_nonzero_counts)[
            -num_features_to_select:
        ]
        self.selected_features_indices = np.sort(self.selected_features_indices)

    def transform(self, X):
        """
        Transforms the dataset by selecting the features identified during fitting.

        Parameters:
        - X: A NumPy array with the same number of features as the dataset used in `fit`.

        Returns:
        - X_reduced: The dataset containing only the selected features.
        """
        if self.selected_features_indices is None:
            raise ValueError("The selector has not been fitted yet. Call fit() first.")

        X_reduced = X[:, self.selected_features_indices]
        return X_reduced

    def fit_transform(self, X):
        """
        Fits the selector and transforms the dataset in a single step.

        Parameters:
        - X: A NumPy array where rows are samples/documents and columns are features/terms.

        Returns:
        - X_reduced: The dataset containing only the selected features.
        """
        self.fit(X)
        return self.transform(X)


def normalized_shannon_entropy(indices):
    counts = Counter(indices)

    total = len(indices)
    probabilities = np.array(list(counts.values())) / total

    entropy = -np.sum(probabilities * np.log2(probabilities))

    max_entropy = np.log2(total) if total > 1 else 1

    normalized_entropy = np.abs(entropy / max_entropy)
    return float(normalized_entropy)


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

    roc_aucs = []
    for i in range(y_test.shape[1]):
        try:
            roc_auc = roc_auc_score(y_test[:, i], y_pred[:, i])
            roc_aucs.append(roc_auc)
        except ValueError:
            roc_aucs.append(0.0)

    roc_auc = np.mean(roc_aucs)

    precision, recall, f1_per_label, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )

    return {
        "Hamming Loss": hamming,
        "Accuracy": accuracy,
        "AUC-ROC": roc_auc,
        "F1 Micro": f1_micro,
        "F1 Macro": f1_macro,
    }


def mmo(X, y, target_proportion=1, **kwargs):
    selected_samples = []
    N, num_features = X.shape
    _, M = y.shape

    current_label_counts = np.sum(y, axis=0)
    T = np.max(current_label_counts) * int(target_proportion)
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

    return X_resampled, y_resampled, selected_samples


def no_oversample(X, y, **kwargs):
    return X, y, []


def ml_smote(X, y, target_proportion=1, k=5, **kwargs):
    """
    Apply multilabel SMOTE to generate synthetic samples for a multilabel dataset to achieve balanced classes.

    Parameters:
    X : np.ndarray
        The input feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        The multilabel binary target matrix of shape (n_samples, n_classes).
    k : int, optional
        Number of nearest neighbors to consider for synthetic sample generation.

    Returns:
    X_resampled : np.ndarray
        The resampled feature matrix including synthetic samples.
    y_resampled : np.ndarray
        The resampled target matrix including synthetic samples.
    """
    selected_samples = []
    X_resampled, y_resampled = list(X), list(y)

    # Determine the number of samples needed for each label to achieve balance
    label_counts = np.sum(y, axis=0)
    max_count = np.max(label_counts) * int(target_proportion)
    samples_needed_per_label = max_count - label_counts

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

        # Generate enough synthetic samples to balance this label
        for _ in range(samples_needed_per_label[label_idx]):
            idx = np.random.choice(len(X_minority))
            neighbor_idx = np.random.choice(neighbors[idx])

            # Generate a synthetic sample
            lam = np.random.rand()
            X_synthetic = X_minority[idx] + lam * (
                X_minority[neighbor_idx] - X_minority[idx]
            )

            # Add synthetic sample to the dataset
            X_resampled.append(X_synthetic)
            y_resampled.append(y[minority_indices[idx]])

            selected_samples.append(minority_indices[idx])

    return np.array(X_resampled), np.array(y_resampled), selected_samples


def ml_ros(X, y, random_state=None, target_proportion=1, **kwargs):
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
    selected_samples = []
    # Calculate label frequencies
    label_counts = np.sum(y, axis=0)
    max_count = np.max(label_counts) * int(target_proportion)
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

                    selected_samples.append(idx)

    return np.array(X_resampled), np.array(y_resampled), selected_samples


def mmo_smote(X, y, k=5, **kwargs):
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

    return X_resampled, y_resampled, selected_samples


def mmo_mle_nn(X, y, k=5, consistency_threshold=0.3, **kwargs):
    selected_samples = []
    N, num_features = X.shape
    _, M = y.shape

    current_label_counts = np.sum(y, axis=0)
    T = np.max(current_label_counts)
    samples_needed_per_label = T - current_label_counts

    X_resampled = []
    y_resampled = []

    candidate_heap = []

    sample_costs = np.zeros(N, dtype=int)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    def label_consistency(sample_idx):
        neighbor_labels = y[indices[sample_idx]]
        consistency = (neighbor_labels * y[sample_idx]).sum(axis=1) / (
            y[sample_idx].sum() + 1
        )
        return (consistency > consistency_threshold).mean()

    for i in range(N):
        sample_labels = y[i]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        consistency = label_consistency(i)

        if contribution > 0 and consistency >= consistency_threshold:
            score = (contribution / (sample_costs[i] + 1)) * consistency
            heapq.heappush(candidate_heap, (-score, i))

    while np.any(current_label_counts < T):
        if not candidate_heap:
            break

        best_score, best_idx = heapq.heappop(candidate_heap)
        best_score = -best_score

        X_resampled.append(X[best_idx])
        y_resampled.append(y[best_idx])

        current_label_counts += y[best_idx]
        samples_needed_per_label = np.maximum(0, T - current_label_counts)

        sample_costs[best_idx] += 1
        selected_samples.append(best_idx)

        sample_labels = y[best_idx]
        contribution = np.minimum(samples_needed_per_label, sample_labels).sum()
        consistency = label_consistency(best_idx)

        if contribution > 0 and consistency >= consistency_threshold:
            score = (contribution / (sample_costs[best_idx] + 1)) * consistency
            heapq.heappush(candidate_heap, (-score, best_idx))

    X_resampled = np.array(X_resampled)
    y_resampled = np.array(y_resampled)

    X_resampled = np.concatenate((X, X_resampled), axis=0)
    y_resampled = np.concatenate((y, y_resampled), axis=0)

    return X_resampled, y_resampled, selected_samples
