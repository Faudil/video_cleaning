import cv2
import numpy as np


class OrbEncoder:
    def __init__(self):
        self.__orb = cv2.ORB_create()

    def encode(self, frame):
        """
        Extracts a global feature vector from a frame using ORB.
        The function computes ORB descriptors and returns the mean of these descriptors.
        """
        keypoints, descriptors = self.__orb.detectAndCompute(frame, None)
        if descriptors is not None and len(descriptors) > 0:
            #
            return np.mean(descriptors, axis=0)
        else:
            # Return a zero vector if no descriptors were found (ORB descriptor size is typically 32)
            # I don't think this will impact the clustering algorithm performance as it will surely be immediately marked
            # as an outlier
            return np.zeros(32)
