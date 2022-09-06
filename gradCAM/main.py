from __future__ import print_function
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
#from grad_cam import grad_cam
from data import data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_grayscale


import argparse
from tkinter.tix import INTEGER
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets
import numpy as np
from data import initialize_data # data.py in the same folder
from model import Net
import torchvision
import pandas as pd
import glob as glob
import time


    

def test_example(image_path,output_folder,model_path):
    
    state_dict = torch.load(model_path)
    model = Net()
    model.load_state_dict(state_dict)
    model.eval()
    #print(model.eval())
    #heatmap_layer = model.conv3
    heatmap_layer = model.localization[3]
    print(heatmap_layer)
    image_label = None
    image = grad_cam(model, image_path, heatmap_layer, image_label)
    plt.imshow(image)
    image_name = image_path.split('/')[-1]
    image_save_path = f'{output_folder}/gradCAM_{image_name}'
    plt.savefig(image_save_path)

if __name__== "__main__":
    #/content/drive/MyDrive/Adversarial Patch/Mini Dataset/Test
   #image_path =  "/content/images/00408.png"
   output_folder = '/content/drive/MyDrive/Adversarial Patch/Mini Dataset/grad_CAM/test_patched_data'
   model_path = '/content/GTSRB_CNN/model/model_40.pth'

   test_dir = '/content/drive/MyDrive/Adversarial Patch/Mini Dataset/patched_test'

   counter = 0
   for image_path in glob.glob(test_dir+'/*.png'):
    test_example(image_path,output_folder,model_path)
    counter = counter +1
  
   print(counter)



