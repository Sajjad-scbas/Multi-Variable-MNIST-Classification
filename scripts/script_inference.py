import numpy as np
import pickle 
import torch

import json 
import glob
from PIL import Image 
import os 

from tqdm import tqdm
import cv2

import matplotlib.pyplot as plt
from model.model_task1 import SampleCNN
from model.model3 import Net


from einops import rearrange

def inference(model_name, file_path, validation:bool =  True):
    # Prepare the test data
    data_path = file_path + 'data/task1_val/' if validation else file_path + 'data/task1_test/'
    labels_path = file_path + 'gt/task1_val.json' if validation else None
    
    target_size = 28
    
    with open(labels_path, 'r') as j:
        labels = json.loads(j.read())
    
    # Prepare the model 
    with open(f'{model_name}_params.pkl', 'rb') as file :
        params = pickle.load(file)
    device = params['device']
    dtype = torch.float32

    model = SampleCNN(params['num_classes'], params['channels'], params['kernels'], params['strides'], params['fc_features']).to(dtype = dtype, device= device)
    model.load_state_dict(torch.load(f"{model_name}.pt"))
    
    if validation : 
        accu = 0
    
    predictions = {}
    
    for im_path in tqdm(sorted(glob.glob(data_path + '*.png'))):
        im_frame = Image.open(im_path)
        image = np.array(im_frame)
        ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        sub_images = []

        
        for idx, cnt in enumerate(contours):
            
            if cv2.contourArea(cnt) < 10 :
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad_height = max(0, (target_size - h)//2)
            pad_width = max(0, (target_size - w)//2)
            
            
            # Taking ROI of the cotour
            sub_images.append(np.pad(
                image[y:y+h, x:x+w],
                ((pad_height, target_size - h - pad_height), (pad_width, target_size - w - pad_width)),
                mode = 'constant' 
                )
            )
            
    

            
        sub_images = torch.Tensor(sub_images) 
        sub_images = rearrange(sub_images, 'b h w -> b 1 h w')  
        _, pred = torch.max(model(sub_images), 1)
        
        
        predictions[os.path.basename(im_path)] = pred.tolist()
        
        if validation : 
            accu += np.sum(pred.tolist() == labels[os.path.basename(im_path)])
                
          
    if validation :
        accu /= 1000
        
    


if __name__=="__main__":
    model_name = "model_N5"
    file_path = '/Users/scbas/Downloads/Auto1/data/'
    inference(model_name, file_path)
    print("Done")
