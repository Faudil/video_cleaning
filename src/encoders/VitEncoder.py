import cv2
import torch
from PIL import Image

from src.encoders.Encoder import Encoder
from transformers import ViTImageProcessor, ViTModel


class VitEncoder(Encoder):
    def __init__(self):
        self.__processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.__model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.__model.eval()


    def encode(self, frame: cv2.Mat):
        """
        TODO: Add batch processing but I didn't do it because my computer doesn't have a GPU so there's no performance improvement
        """
        # Convert frame from BGR to RGB and then to PIL Image
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Preprocess the image and prepare the tensor
        inputs = self.__processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.__model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu().numpy()

