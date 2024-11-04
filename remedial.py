import numpy as np


class REMEDIAL:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def scumble_score(self, labels):
        """Calculate the concurrence score for a set of labels."""
        minority_labels = [
            label
            for label, freq in self.label_frequencies.items()
            if freq < self.minority_threshold
        ]
        majority_labels = [
            label
            for label, freq in self.label_frequencies.items()
            if freq >= self.minority_threshold
        ]

        minority_count = sum([1 for label in labels if label in minority_labels])
        majority_count = sum([1 for label in labels if label in majority_labels])

        return (minority_count * majority_count) / (len(labels) ** 2)

    def fit_transform(self, X, y):
        """Apply the REMEDIAL algorithm to the dataset X, y."""
        selected_samples = []
        self.label_frequencies = self._calculate_label_frequencies(y)
        self.minority_threshold = np.percentile(
            list(self.label_frequencies.values()), 25
        )

        self.minority_labels = [
            label
            for label, freq in self.label_frequencies.items()
            if freq < self.minority_threshold
        ]
        self.majority_labels = [
            label
            for label, freq in self.label_frequencies.items()
            if freq >= self.minority_threshold
        ]

        new_instances = []
        num_labels = y.shape[1]

        for i, labels in enumerate(y):
            scumble = self.scumble_score(labels)
            if scumble >= self.threshold:
                modified_instances = self._modify_instance(X[i], labels)
                for instance, modified_labels in modified_instances:
                    padded_labels = np.zeros(num_labels)
                    padded_labels[: len(modified_labels)] = modified_labels
                    new_instances.append((instance, padded_labels))
                    selected_samples.append(i)

        X_augmented = np.vstack([X] + [inst[0] for inst in new_instances])
        y_augmented = np.vstack([y] + [inst[1] for inst in new_instances])

        return X_augmented, y_augmented, selected_samples

    def _modify_instance(self, instance, labels):
        """Generate modified instances by decoupling labels."""
        modified_instances = []
        minority_labels = [label for label in labels if label in self.minority_labels]
        majority_labels = [label for label in labels if label in self.majority_labels]

        if minority_labels:
            modified_instances.append((instance, minority_labels))
        if majority_labels:
            modified_instances.append((instance, majority_labels))

        return modified_instances

    def _calculate_label_frequencies(self, y):
        """Calculate frequency of each label in the dataset."""
        label_counts = {}
        for labels in y:
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts


def remedial(X, y, **kwargs):
    remedial = REMEDIAL(0.2)
    X_augmented, y_augmented, selected_samples = remedial.fit_transform(X, y)
    return X_augmented, y_augmented, selected_samples
