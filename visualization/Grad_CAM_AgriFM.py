import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from Grad_CAM_Code.build_model import build_model


def reshape_transform(tensor, height=None, width=None):

    B, N, C = tensor.shape

    import math
    grid_size = int(math.sqrt(N))

    result = tensor.reshape(B, grid_size, grid_size, C)
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class GradCAMProcessor_AgriFMB:
    def __init__(self, weights_path, image_path, output_filename, model_name, num_classes, device=None, ):
        self.WEIGHTS_PATH = weights_path
        self.IMAGE_PATH = image_path
        self.OUTPUT_FILENAME = output_filename
        self.DEVICE = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_name = model_name
        self.num_classes = num_classes

    def load_model(self, num_classes=123, ):

        self.model = build_model(self.model_name, self.num_classes)
        print(self.model)

        checkpoint = torch.load(self.WEIGHTS_PATH, map_location='cpu', weights_only=False)
        model_state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.DEVICE).eval()

    def process_image(self):

        image_array = np.fromfile(self.IMAGE_PATH, dtype=np.uint8)
        bgr_img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        rgb_img = bgr_img[:, :, ::-1]  # BGR to RGB

        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255.0

        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(self.DEVICE)
        return rgb_img, input_tensor

    def generate_heatmap(self, rgb_img, input_tensor):

        target_layers = [self.model.layers[-1].blocks[-1].norm1]

        cam = GradCAM(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform
        )

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(self.OUTPUT_FILENAME, cam_image)


    def run(self, num_classes=123):

        print(f"Using device: {self.DEVICE}")
        self.load_model(num_classes=num_classes)
        rgb_img, input_tensor = self.process_image()
        self.generate_heatmap(rgb_img, input_tensor)


if __name__ == '__main__':
    WEIGHTS_PATH = r""
    IMAGE_PATH = r""
    OUTPUT_FILENAME = "my_model_gradcam.jpg"
    model_name = "AgriFM-B"
    num_classes = 123
    processor = GradCAMProcessor_AgriFMB(WEIGHTS_PATH, IMAGE_PATH, OUTPUT_FILENAME, model_name, num_classes)
    processor.run()