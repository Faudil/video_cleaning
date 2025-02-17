import numpy as np


def cosine_distances(vectors, reference_vector):
    """
    Computes the cosine distance between a list of vectors and a reference vector.

    Parameters:
    vectors (array-like): List of vectors to compare.
    reference_vector (array-like): The reference vector.

    Returns:
    ndarray: Cosine distances between each vector and the reference vector.
    """
    vectors = np.asarray(vectors)
    reference_vector = np.asarray(reference_vector)

    dot_products = np.dot(vectors, reference_vector)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(reference_vector)

    return 1 - (dot_products / norms)