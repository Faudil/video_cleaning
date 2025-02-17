

# !!!!!!!!
# This code is not used anymore but I let it in the repository to showcase my first experiment and thought process
# !!!!!!!!

import cv2
import numpy as np

from src.encoders.Encoder import Encoder


class SiftEncoder(Encoder):
    """
    This method did not yield any good result even when
    """
    def __init__(self):
        self.__sift = cv2.SIFT_create()
        self.__bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    def encode(self, image):
        _, descriptors = self.__sift.detectAndCompute(image, None)

        return descriptors


    def similarity(self, descriptor_1, descriptor_2) -> float:
        """
        First method to compute the similarity between two images by clustering their descriptors
        This method is quite slow and not worth the performance tradeoff compared to the ORB or VIT method so I decided
        not to use it.
        This code is also not deleted for you to better understand my though process but is used nowere n
        """
        matches = self.__bf_matcher.knnMatch(descriptor_1, descriptor_2, k=2)

        # Apply the ratio test to find good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Calculate similarity score
        similarity = len(good_matches) / max(len(descriptor_1), len(descriptor_2))
        return similarity
