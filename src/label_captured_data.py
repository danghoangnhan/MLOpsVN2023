import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from problem_config import ProblemConfig, ProblemConst, get_prob_config


def label_captured_data(prob_config: ProblemConfig):
    train_x = pd.read_parquet(prob_config.train_x_path)
    train_y = pd.read_parquet(prob_config.train_y_path)
    ml_type = prob_config.ml_type

    logging.info("Load captured data")
    captured_x = pd.DataFrame()
    for file_path in prob_config.captured_data_dir.glob("*.parquet"):
        captured_data = pd.read_parquet(file_path)
        captured_data = captured_data.dropna()
        captured_data = captured_data[train_x.columns]
        captured_x = pd.concat([captured_x, captured_data])

    logging.info(f"Loaded {len(captured_x)} captured samples, {len(train_x) + len(captured_x)} train + captured")

    # Align features between captured data and training data
    captured_x = captured_x[train_x.columns]

    captured_x = pd.get_dummies(captured_x)
    logging.info("Preprocess the data to handle missing values")
    # Handle missing values in captured data
    imputer = SimpleImputer(strategy="mean")
    captured_x = pd.DataFrame(imputer.fit_transform(captured_x), columns=train_x.columns)
    captured_x = pd.get_dummies(captured_x)

    logging.info("Initialize and fit the clustering model")
    n_cluster = int((len(train_x) + len(captured_x)) / 10) * len(np.unique(train_y))
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=prob_config.random_state
    ).fit(train_x)

    logging.info("Predict the cluster assignments for the new data")
    kmeans_clusters = kmeans_model.predict(captured_x)

    logging.info(
        "Assign new labels to the new data based on the labels of the original data in each cluster"
    )
    new_labels = []
    for i in range(n_cluster):
        mask = kmeans_model.labels_ == i  # mask for data points in cluster i
        cluster_labels = train_y[mask]  # labels of data points in cluster i
        if len(cluster_labels) == 0:
            # If no data points in the cluster, assign a default label (e.g., 0)
            new_labels.append(0)
        else:
            # For a linear regression problem, use the mean of the labels as the new label
            # For a logistic regression problem, use the mode of the labels as the new label
            if ml_type == "regression":
                new_labels.append(np.mean(cluster_labels.values.flatten()))
            else:
                new_labels.append(
                    np.bincount(cluster_labels.values.flatten().astype(int)).argmax()
                )

    approx_label = [new_labels[c] for c in kmeans_clusters]
    approx_label_df = pd.DataFrame(approx_label, columns=[prob_config.target_col])

    captured_x.to_parquet(prob_config.captured_x_path, index=False)
    approx_label_df.to_parquet(prob_config.uncertain_y_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    label_captured_data(prob_config)