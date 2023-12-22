import os 
import json 
import glob
import pickle 
from PIL import Image 

import numpy as np

import cv2
import torch

from tqdm import tqdm

from model.model import SampleCNN

from einops import rearrange



def remove_noise_horizontal_salt(image):
    """
    Remove horizontal lines and salt and pepper noise from an image.
    Args:
        image (numpy.ndarray): The input image.
    Returns:
        numpy.ndarray: The denoised image.
    """
    kernel_size = (2,1)
    kernel = np.ones(kernel_size, np.uint8)
    
    border_size = 1
    padded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT)
    
    denoise_image = cv2.morphologyEx(padded_image, cv2.MORPH_OPEN, kernel=kernel)
    denoise_image = denoise_image[border_size:-border_size, border_size:-border_size]

    denoise_image = cv2.medianBlur(denoise_image, 1)
    return denoise_image



def inference(model_name, file_path, output_path, nb_task = 1, validation:bool =  True):
    """
    Generate the predictions for a given model on a set of  
    and save the results in a json file.
    
    Parameters:
        model_name (str): The name of the model to use for inference.
        file_path (str): The path to the directory containing the image data.
        output_path (str, optional): The path to the directory where the predictions will be saved.
        nb_task (int, optional): The task number. Defaults to 1.
        validation (bool, optional): Whether to use the validation set. Defaults to True.
    """
    
    # Prepare the test data
    data_path = f"{file_path}/data/task{nb_task}_val/" if validation else f"{file_path}/data/task{nb_task}_test/"
    
    target_size = 28
    
    # Load the model with its weights
    with open(f'model_weights/{model_name}_params.pkl', 'rb') as file :
        params = pickle.load(file)
    device = params['device']
    dtype = torch.float32
    model = SampleCNN(params['num_classes'], params['channels'], params['kernels'], params['strides'], params['fc_features']).to(dtype = dtype, device= device)
    model.load_state_dict(torch.load(f"model_weights/{model_name}.pt"))
    
    
    # Dictionary for the result of the predictions
    predictions = {}
    
    for im_path in tqdm(sorted(glob.glob(data_path + '*.png'))):
        im_frame = Image.open(im_path)
        image = np.array(im_frame)
        
        # Preprocessing the images (mostly for the 2nd and 3rd task)
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
        if nb_task == 2 :
            thresh = remove_noise_horizontal_salt(thresh)
                            
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        sub_images = []

        for cnt in contours:
            
            if cv2.contourArea(cnt) < 10 :
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad_height = max(0, (target_size - h)//2)
            pad_width = max(0, (target_size - w)//2)
            
            # Taking ROI of the cotour
            sub_images.append(np.pad(
                thresh[y:y+h, x:x+w],
                ((pad_height, target_size - h - pad_height), (pad_width, target_size - w - pad_width)),
                mode = 'constant' 
                )
            )

        sub_images = torch.Tensor(sub_images) 
        sub_images = rearrange(sub_images, 'b h w -> b 1 h w')  
        _, pred = torch.max(model(sub_images), 1)
        
        # Restore the predictions 
        predictions[os.path.basename(im_path)] = pred.tolist()
        
  
    # Convert the dictionary into a json & save it
    predictions = json.dumps(predictions, indent=2)
    
    name_file_output = f"predictions_task{nb_task}_val.json" if validation else f"predictions_task{nb_task}_test.json"
    with open(f"{output_path}/{name_file_output}", 'w') as json_file:
        json_file.write(predictions)
        
    


if __name__=="__main__":
    model_name = "model_N14"
    file_path = '/Users/scbas/Downloads/Auto1/data'
    output_path = '/Users/scbas/Downloads/Auto1/predictions'
    inference(model_name, file_path, output_path, nb_task=2, validation=False)
    print("Done")
