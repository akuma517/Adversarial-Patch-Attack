# We are testing the model on the test dataset.

import numpy as np
import cv2
import torch
import glob as glob
import pandas as pd
import os
import albumentations as A
import time
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torch import topk
from model import build_model
# Define computation device.
device = 'cpu'
# Class names.
sign_names_df = pd.read_csv('/content/Adversarial-Patch-Attack/TrafficSignRecognition/outputs/signnames.csv')
class_names = sign_names_df.SignName.tolist()
# DataFrame for ground truth.
gt_df = pd.read_csv(
    '/content/input/gtsrb-german-traffic-sign/test_labels.csv', 
    delimiter=';'
)
gt_df = gt_df.set_index('Filename', drop=True)
# Initialize model, switch to eval model, load trained weights.
model = build_model(
    pretrained=False,
    fine_tune=False, 
    num_classes=43
).to(device)
model = model.eval()
model.load_state_dict(
    torch.load(
        '/content/drive/MyDrive/Adversarial Patch/trafficSignRecognition/outputs/model.pth', map_location=device
    )['model_state_dict']
)
# Define the transforms, resize => tensor => normalize.
transform = A.Compose([
    A.Resize(224, 224),   # 224 is set by us
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2(),
    ])
counter = 0
# Run for all the test images.
all_images = glob.glob('../input/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/*.png')
correct_count = 0
frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second. 
for i, image_path in enumerate(all_images):
    # Read the image.
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = orig_image.shape
    # Apply the image transforms.
    image_tensor = transform(image=image)['image']
    # Add batch dimension.
    image_tensor = image_tensor.unsqueeze(0)
    # Forward pass through model.
    start_time = time.time()
    outputs = model(image_tensor.to(device))
    end_time = time.time()
    # Get the softmax probabilities.
    probs = F.softmax(outputs).data.squeeze()
    # Get the class indices of top k probabilities.
    class_idx = topk(probs, 1)[1].int()
    # Get the ground truth.
    image_name = image_path.split(os.path.sep)[-1]
    gt_idx = gt_df.loc[image_name].ClassId
    # Check whether correct prediction or not.
    if gt_idx == class_idx:
        correct_count += 1

    counter += 1
    #print(f"Image: {counter}")
    # Get the current fps.
    fps = 1 / (end_time - start_time)
    # Add `fps` to `total_fps`.
    total_fps += fps
    # Increment frame count.
    frame_count += 1
    
print(f"Total number of test images: {len(all_images)}")
print(f"Total correct predictions: {correct_count}")
print(f"Accuracy: {correct_count/len(all_images)*100:.3f}")
# Close all frames and video windows.

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")