from typing import Callable, List

import numpy as np
from sklearn.cluster import DBSCAN


class FrameFilter:
    def __init__(self, clustering_eps=0.5, clustering_min_samples=5, filtering_threshold=2.0):
        # The maximum distance between two samples for them to be considered as in the same neighborhood.
        self.__clustering_eps = clustering_eps
        # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        self.__clustering_min_samples = clustering_min_samples
        # The maximum standard deviations above the mean distance for filtering out outlier frames.
        self.__filtering_threshold = filtering_threshold

    def _cluster_features(self, features):
        clustering = DBSCAN(eps=self.__clustering_eps,
                            min_samples=self.__clustering_min_samples,
                            metric='cosine').fit(features)
        return clustering.labels_

    def _get_target_cluster(self, labels):
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return None, []
        unique, counts = np.unique(valid_labels, return_counts=True)
        target_cluster = unique[np.argmax(counts)]
        indices = [i for i, lab in enumerate(labels) if lab == target_cluster]
        return target_cluster, indices

    def _filter_outliers(self, features):
        cluster_mean = np.mean(features, axis=0)
        distances = np.linalg.norm(features - cluster_mean, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        return [i for i, distance in enumerate(distances) if distance < mean_dist + self.__filtering_threshold * std_dist]

    def filter_frames(self, features) -> List[int]:
        labels = self._cluster_features(features)
        target_cluster, indices = self._get_target_cluster(labels)

        if target_cluster is None or not indices:
            cluster_features = features
            original_indices = list(range(len(features)))
        else:
            cluster_features = features[indices]
            original_indices = indices

        filtered_relative = self._filter_outliers(cluster_features)
        filtered_original = [original_indices[i] for i in filtered_relative]
        return filtered_original

