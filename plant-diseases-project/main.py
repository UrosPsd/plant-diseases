import os
from PIL import Image
import torchvision.transforms.functional as TF

# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

from CNN import CNN

# Import Dataset
transform = transforms.Compose(
    [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
)
dataset = datasets.ImageFolder("Dataset", transform=transform)

# Load Model
targets_size = 39
model = CNN(targets_size)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()

data = pd.read_csv("disease_info.csv", encoding="cp1252")

transform_index_to_disease = dataset.class_to_idx
transform_index_to_disease = dict(
    [(value, key) for key, value in transform_index_to_disease.items()]
)


def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    print("Original : ", image_path[12:-4])
    pred_csv = data["disease_name"][index]
    print(pred_csv)


if __name__ == '__main__':
    img_path = "test_images/tomato_mosaic_virus.JPG"
    prediction(img_path)

