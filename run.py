import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import (
    LabelPowerset,
    BinaryRelevance,
    ClassifierChain,
)
from skmultilearn.dataset import load_dataset

from util import (
    evaluate_multilabel,
    mmo,
    no_oversample,
    ml_smote,
    ml_ros,
    mmo_smote,
    mmo_mle_nn,
    label_density,
    mean_imbalance_ratio,
)
from sklearn.preprocessing import MinMaxScaler
import time

# Define random seeds for reproducibility
random_seeds = list(range(1, 11))

datasets = [
    "birds",
    "medical",
    "emotions",
    "genbase",
    "enron",
    "scene",
    "yeast",
    "rcv1subset3",
    "rcv1subset1",
    "rcv1subset5",
    "rcv1subset2",
    "rcv1subset4",
    "Corel5k",
    "bibtex",
    "delicious",
    "tmc2007_500",
    "mediamill",
]

oversampling_methods = {
    "none": no_oversample,
    "ml_smote": ml_smote,
    "ml_ros": ml_ros,
    "mmo": mmo,
    "mmo_smote": mmo_smote,
    "mmo_mle_nn": mmo_mle_nn,
}


def process_dataset_with_seed(
    classifier_name, classifier, dataset_name, data, oversampling_methods, seed
):
    dataset_csv_path = (
        f"datasets/{classifier_name}_{dataset_name}_results_seed_{seed}.csv"
    )

    if os.path.exists(dataset_csv_path):
        return dataset_csv_path

    results = []
    X_train = np.asarray(data["X_train"].todense())
    y_train = np.asarray(data["y_train"].todense())
    X_test = np.asarray(data["X_test"].todense())
    y_test = np.asarray(data["y_test"].todense())
    np.random.seed(seed)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for oversample_name, oversample_func in oversampling_methods.items():
        print(
            f"Applying {oversample_name} oversampling for {dataset_name} with seed {seed}."
        )
        start_time_oversampling = time.time()

        kwargs = {"random_state": seed}
        X_resampled, y_resampled = oversample_func(X_train, y_train, **kwargs)
        end_time_oversampling = time.time()

        print(
            f"Training {classifier_name} for {dataset_name} with oversampling {oversample_name} with seed {seed}."
        )
        start_time_training = time.time()
        classifier.fit(X_resampled, y_resampled)
        end_time_training = time.time()

        y_pred = classifier.predict(X_test).toarray()
        metrics = evaluate_multilabel(y_test, y_pred)

        row = {
            "Dataset": dataset_name,
            "Oversampling": oversample_name,
            "Classifier": classifier_name,
            "Seed": seed,
            "Oversampling_Time_ms": (end_time_oversampling - start_time_oversampling)
            * 1000,
            "Training_Time_ms": (end_time_training - start_time_training) * 1000,
            "Train_Set_Increase": (X_resampled.shape[0] - X_train.shape[0])
            / X_train.shape[0],
            "Mean_Imbalance_Ratio_Before": mean_imbalance_ratio(y_train),
            "Mean_Imbalance_Ratio_After": mean_imbalance_ratio(y_resampled),
            "Label_Density_Before": label_density(y_train),
            "Label_Density_After": label_density(y_resampled),
        }
        row.update(metrics)
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(dataset_csv_path, index=False)
    print(f"Results for {dataset_name} with seed {seed} saved to {dataset_csv_path}.")

    return dataset_csv_path


def run_experiment_sequential(data_dict, oversampling_methods, random_seeds):
    dataset_csv_paths = []

    for dataset_name, data in data_dict.items():
        for seed in random_seeds:
            classifier_dict = {
                "BR-RF": BinaryRelevance(
                    RandomForestClassifier(n_estimators=100, random_state=seed)
                ),
                "CC-RF": ClassifierChain(
                    RandomForestClassifier(n_estimators=100, random_state=seed)
                ),
                "LP-RF": LabelPowerset(
                    RandomForestClassifier(n_estimators=100, random_state=seed)
                ),
            }
            for classifier_name, classifier in classifier_dict.items():
                try:
                    dataset_csv_path = process_dataset_with_seed(
                        classifier_name,
                        classifier,
                        dataset_name,
                        data,
                        oversampling_methods,
                        seed,
                    )
                    dataset_csv_paths.append(dataset_csv_path)
                    print(f"Completed processing for {dataset_csv_path}.")

                except Exception as e:
                    print(f"Error processing {dataset_name} with seed {seed}: {e}")

    combined_results = pd.concat([pd.read_csv(path) for path in dataset_csv_paths])
    combined_results.to_csv("datasets/consolidated_results.csv", index=False)
    print("\nConsolidated results saved to 'consolidated_results.csv'.")
    return combined_results


if __name__ == "__main__":
    data_dict = {}
    for dataset in datasets:
        try:
            X_train, y_train, _, _ = load_dataset(dataset, "train")
            X_test, y_test, _, _ = load_dataset(dataset, "test")

            data_dict[dataset] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
            print(f"Successfully loaded {dataset} dataset.")
        except Exception as e:
            print(f"Error loading {dataset} dataset: {e}")

    results_df = run_experiment_sequential(
        data_dict, oversampling_methods, random_seeds
    )
