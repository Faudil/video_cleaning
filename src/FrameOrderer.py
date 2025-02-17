from typing import Callable

import networkx as nx
from networkx.algorithms.approximation.traveling_salesman import greedy_tsp, christofides, threshold_accepting_tsp
from sklearn.metrics.pairwise import cosine_distances
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

class FrameOrderer:
    def __init__(self, distance_function: Callable = cosine_distances):
        self.__distance_function = distance_function
        self.methods = [
            greedy_tsp,
            christofides,
        ]

    def _build_graph(self, features):
        """
        Builds a complete graph where each node represents a frame and each edge's weight is the
        distance between feature vectors.
        """
        n = len(features)
        graph = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                dist = 1 - cosine_similarity(features[i], features[j])
                graph.add_edge(i, j, weight=dist)
        return graph

    def _find_shortest_path(self, graph):
            return nx.algorithms.approximation.traveling_salesman_problem(graph, cycle=False)

    def order_frames(self, features):
        graph = self._build_graph(features)
        return self._find_shortest_path(graph)