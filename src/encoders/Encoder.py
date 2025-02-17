import abc
from abc import ABC

import cv2


class Encoder(ABC):

    @abc.abstractmethod
    def encode(self, image: cv2.Mat):
        pass
