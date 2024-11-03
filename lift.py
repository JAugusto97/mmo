"""
Credits to the authors of the paper: "Multi-label learning with Relief-based label-specific
feature selection" (https://doi.org/10.1007/s10489-022-04350-1)

This code was only slighly modified. The original code can be found at: https://data.mendeley.com/datasets/dj4crhrp4w/1.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from collections import Counter

import math
import random
import warnings

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    zero_one_loss,
    coverage_error,
)


warnings.filterwarnings("ignore")


def batchkmeans_cluster(D, clust_num):
    batch_kmeans = MiniBatchKMeans(
        n_clusters=clust_num, max_iter=100, batch_size=D.shape[0] * 10
    )
    batch_kmeans.fit(D)
    centroids = batch_kmeans.cluster_centers_
    return centroids


# calculates distances of each instance from a centroid
def centroid_distance(data, centroid):
    dist = []
    for instance in range(len(data)):
        # dist.append(np.linalg.norm(data[instance,:].values-centroid))  # 范数
        values = data[instance]
        dist.append(np.linalg.norm(values - centroid))

    return dist


# calculates distances of each instance from all centroid
def dist_of_rows_from_centroids(X, centroids):
    # distance_dataframe=pd.DataFrame()
    distance_dataframe2 = []
    i = 0
    for centre in centroids:
        dist = centroid_distance(X, centre)
        # distance_dataframe[i]=dist #this will loop for all centroids
        distance_dataframe2.append(dist)
        i = i + 1
    distance_dataframe2 = list(map(list, zip(*distance_dataframe2)))
    return distance_dataframe2


def SGD_svm(X, Y):
    clf = SGDClassifier(n_jobs=-1)
    clf.fit((X), Y)
    return clf


def clusterDistance(cluster_centers, X_min, cluster_labels, targetCluster=0):
    tmp = [
        index for (index, value) in enumerate(cluster_labels) if value == targetCluster
    ]
    X_label = [X_min[i] for i in tmp]
    distS = []
    for i in X_label:
        dist = np.linalg.norm(cluster_centers[targetCluster] - i)
        distS.append(dist)
    return np.mean(distS), tmp


def cluster(X_max, X_min, y, clust_num, min_all_dict, cluster_matrix):
    # print (X_min)
    kmeans = KMeans(n_clusters=clust_num, random_state=0).fit(X_min)
    cluster_labels = kmeans.labels_  # cluster
    cluster_centers = kmeans.cluster_centers_  # 中心点
    max_labels = kmeans.predict(X_max)
    max_label_counterDict = Counter((max_labels))

    label_counterDict = Counter((cluster_labels))
    label_counter = []
    max_label_counter = []
    cluster_avgDistance = []

    for ii in range(0, clust_num):
        cluster_dis, cluster_index = clusterDistance(
            cluster_centers, X_min, cluster_labels, targetCluster=ii
        )
        cluster_avgDistance.append(cluster_dis)
        label_counter.append(label_counterDict[ii])

        if ii in max_label_counterDict:
            max_label_counter.append(max_label_counterDict[ii])
        else:
            max_label_counter.append(1)

        if len(cluster_index) > 1:
            for i in range(0, len(cluster_index) - 1):
                for j in range(1, len(cluster_index)):
                    ci = cluster_index[i]
                    cj = cluster_index[j]
                    if min_all_dict[ci] in cluster_matrix:
                        v = cluster_matrix[min_all_dict[ci]]
                        v.add(min_all_dict[cj])
                        cluster_matrix[min_all_dict[ci]] = v
                    if min_all_dict[cj] in cluster_matrix:
                        v = cluster_matrix[min_all_dict[cj]]
                        v.add(min_all_dict[ci])
                        cluster_matrix[min_all_dict[cj]] = v
                    if min_all_dict[ci] not in cluster_matrix:
                        v = set()
                        v.add(min_all_dict[cj])
                        cluster_matrix[min_all_dict[ci]] = v
                    if min_all_dict[cj] not in cluster_matrix:
                        v = set()
                        v.add(min_all_dict[ci])
                        cluster_matrix[min_all_dict[cj]] = v

    return (
        cluster_avgDistance,
        label_counter,
        max_label_counter,
        cluster_labels,
        cluster_matrix,
    )  #


def cluster_score_computer(cluster_avgDistance, label_counter, max_label_counterDict):
    # 单拉出来 看看
    SF = []
    for ii in range(0, len(cluster_avgDistance)):
        if cluster_avgDistance[ii] == 0:  #
            SF.append(0)
            continue
        sf = (cluster_avgDistance[ii] / max_label_counterDict[ii]) * label_counter[ii]
        SF.append(sf)
    if np.sum(SF) > 0:
        SF_score = SF / np.sum(SF)
    else:
        SF_score = np.zeros(len(SF))
    return SF_score


def LIFT(X, Y, Xt, Yt, ratio):
    # step-1
    classifiers_for_label = {}  # this will store all classifier functions
    centroids_per_label = {}
    for label in range(Y.shape[1]):
        print(label)
        positive_instances = []
        negative_instances = []
        ylabel = []
        for i in range(0, len(X)):
            if Y[i, label] == 1:
                positive_instances.append(X[i])
                ylabel.append(1)
            else:
                negative_instances.append(X[i])
                ylabel.append(0)

        positive_instances = np.array(positive_instances)
        negative_instances = np.array(negative_instances)
        ylabel = np.array(ylabel)

        clust_num = int(
            ratio * (min(len(positive_instances), len(negative_instances)))
        )  # calculates the number of clusters
        clust_num = max(clust_num, 1)

        centroids = []  # will stores all the centroids

        centroids.extend(batchkmeans_cluster(positive_instances, clust_num))
        centroids.extend(batchkmeans_cluster(negative_instances, clust_num))
        centroids_per_label[str(label)] = centroids
        distance_dataframe = dist_of_rows_from_centroids(
            X, centroids
        )  # it saves distance from instances to each centroids
        # step-2
        classifiers_for_label[str(label)] = SGD_svm(
            distance_dataframe, ylabel
        )  # classifier is trained label wise from the distance matrix and label
    # step-3
    results = pd.DataFrame()
    for label_2b_pred in range(Y.shape[1]):
        Xt_dist_for_label = dist_of_rows_from_centroids(
            Xt, centroids_per_label[str(label_2b_pred)]
        )
        results[str(label_2b_pred)] = classifiers_for_label[str(label_2b_pred)].predict(
            Xt_dist_for_label
        )  # this transforms test set to the distance form upon which all classifiers will act to give labelset
    print("Hamming loss : {}".format(hamming_loss(Yt, results.values)))
    print("zero_one_loss : {}".format(zero_one_loss(Yt, results.values)))
    print("coverage_error : {}".format(coverage_error(Yt, results.values)))
    print(
        "label_ranking_average_precision_score : {}".format(
            label_ranking_average_precision_score(Yt, results.values)
        )
    )

    print(
        str((f1_score(Yt, results.values, average="macro")))
        + "\t"
        + str((f1_score(Yt, results.values, average="micro")))
    )


def LIFT_WEIGHT(X, Y, ratio):
    """
    :param X:
    :param Y:
    :param Xt:
    :param Yt:
    :param ratio:
    :return:
    """

    sample_total_weight = np.zeros((X.shape[0], Y.shape[1]))  # n * q
    label_ir = []
    label_ir_max = []
    cluster_matrix = {}
    for label in range(Y.shape[1]):

        class_1 = np.count_nonzero(Y[:, label] == 1)
        class_0 = np.count_nonzero(Y[:, label] == 0)
        if class_1 > 0:
            ir_label = class_0 / (class_1)
        else:
            ir_label = 0.0  # No generate while not in training part
        # ir_label, _ = mld_metrics.ir_per_label(label, Y)
        if ir_label > 1.0 and not math.isinf(ir_label):  # The Min The More The Rare
            # print (label)
            label_ir.append(ir_label)
            # step-1 cluster
            positive_instances = []
            negative_instances = []
            ylabel = []
            min_all_dict = {}
            for i in range(0, len(X)):
                if Y[i, label] == 1:
                    min_all_dict[len(positive_instances)] = len(
                        positive_instances
                    ) + len(negative_instances)
                    positive_instances.append(X[i])
                    ylabel.append(1)
                else:
                    negative_instances.append(X[i])
                    ylabel.append(0)

            positive_instances = np.array(positive_instances)
            negative_instances = np.array(negative_instances)
            ylabel = np.array(ylabel)

            clust_num = int(
                ratio * (min(len(positive_instances), len(negative_instances)))
            )  # calculates the number of clusters
            clust_num = max(clust_num, 1)
            (
                cluster_avgDistance,
                label_counter,
                max_label_counterDict,
                cluster_labels,
                cluster_matrix,
            ) = cluster(
                negative_instances,
                positive_instances,
                ylabel,
                clust_num,
                min_all_dict,
                cluster_matrix,
            )
            # step-2 label specific score compute
            SF_score = cluster_score_computer(
                cluster_avgDistance, label_counter, max_label_counterDict
            )

            for i, j in min_all_dict.items():
                labelthis = cluster_labels[i]
                sample_total_weight[j, label] = SF_score[
                    labelthis
                ]  # care samples to labels
        else:
            label_ir.append(0)
            label_ir_max.append(ir_label)

    label_ir = label_ir / np.sum(label_ir)  # involved max
    sample_total_weight2 = []

    for i in sample_total_weight:
        tmp = 0.0
        for j in range(0, len(label_ir)):
            tmp += i[j] * label_ir[j]
        sample_total_weight2.append(tmp)
    sample_total_weight2 = sample_total_weight2 / np.sum(sample_total_weight2)
    label_cluster_matrix = cluster_matrix
    return sample_total_weight2, label_cluster_matrix


def get_seed_instance(w):
    seed_index = 0
    limit = random.random() * sum(w)
    temp_sum = 0
    for i in range(len(w)):
        temp_sum += w[i]
        if limit <= temp_sum:
            seed_index = i
            break
    return seed_index


def lift(X, Y, r=0.2, K=3, perc_gen_instances=1):

    sample_total_weight, label_cluster_matrix = LIFT_WEIGHT(X, Y, r)
    min_w = []
    X_min = []
    Y_min = []

    for i in range(0, len(sample_total_weight)):
        if sample_total_weight[i] > 0:
            min_w.append(sample_total_weight[i])
            X_min.append(X[i])
            Y_min.append(Y[i])

    Kn = K
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(
        n_neighbors=Kn + 1, algorithm="brute", metric="euclidean"
    ).fit(X)
    distances, indices = nbrs.kneighbors(X)

    X_train = X
    y_train = Y
    gen_num = int(X.shape[0] * perc_gen_instances)
    X_mlsmote = list(X_train)
    Y_mlsmote = list(y_train)

    X_syn = []
    synset = set()
    for i in range(0, gen_num):
        seed_index = get_seed_instance(sample_total_weight)

        x_seed = X[seed_index]
        y_seed = Y[seed_index]
        synset.add(seed_index)
        # reference_index = indices[seed_index][random.randint(1, Kn)]  # SMOTE依赖的另一个数据
        # x_reference = X[reference_index]
        reference_index = indices[seed_index]
        check_reference = []
        for r in reference_index:
            if r in label_cluster_matrix[seed_index]:
                check_reference.append(r)
        if seed_index in check_reference:
            check_reference.remove(seed_index)

        if len(check_reference) >= 1:
            # print (check_reference)
            tmp = random.randint(1, len(check_reference))
            # print (tmp)
            reference_index = check_reference[tmp - 1]
        else:
            reference_index = indices[seed_index][random.randint(1, Kn)]
        x_reference = X[reference_index]

        x_synthetic = np.zeros(len(x_seed))

        for ii in range(0, len(x_seed)):
            x_synthetic[ii] = x_seed[ii] + (
                random.random() * 1.0 * (x_reference[ii] - x_seed[ii])
            )
        y_synthetic = y_seed
        dist_seed = np.linalg.norm(x_synthetic - x_seed)
        dist_reference = np.linalg.norm(x_synthetic - x_reference)
        cd = dist_seed / (dist_reference)
        if cd > 1:
            y_synthetic = Y[reference_index]

        X_mlsmote.append(x_synthetic)
        Y_mlsmote.append(y_synthetic)

        X_syn.append(x_synthetic)

    return np.asarray(X_mlsmote), np.asarray(Y_mlsmote)
