# IMAGE CLASSIFIER COMMAND LINE APPLICATION
#
# Usage:
#   python predict.py 
#                   --data_dir Path to the folder of flower images
#                   --image_file Path to an image file
#
# PROGRAMMER: Chris Mastel
# DATE CREATED: 12-April-2019
# DATE REVISED: 16-April-2019
# PURPOSE: Use a trained network to predict the type of flower in a given input image
#           This function inputs:
#               - a path to the location of the folder containing images
#               - a file name for the desired image
#           This funtions outputs:
#               - Predicted name of the flower in the given image
#               - Probability of the prediction
#
# Import modules
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import json
import pandas as pd
from model_utils import get_predict_args

# Definge global variables
checkpoint_file = 'py_checkpoint.pth'
image_file = '/3/image_06634.jpg'
categories = 'cat_to_name.json'
top_k_classes = 5

def load_checkpoint(filepath):
    print('Running load_checkpoint({})....'.format(filepath))
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    lr = checkpoint['learn_rate']
    model.optimizer = checkpoint['optimizer']
    #model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = nn.NLLLoss()
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model, checkpoint['class_to_idx']

def process_image(image_file):
    # Scales, crops, and normalizes a PIL image for PyTorch model,
    # Returns a NumPy array
    print('Running process_image({})....'.format(image_file))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pil_image = Image.open(image_file)
    
    # Resize the image so that the shortest side is 256 pixels
    
    width, height = pil_image.size
    short_side = min(width, height)
    pil_image = pil_image.resize((int((width/short_side)*256), int((height/short_side)*256)))
    width, height = pil_image.size
    
    # Crop out the centre 224 x 224 pixels of the image
    left = (width-224)/2
    right = (width+224)/2
    top = (height-224)/2
    bottom = (height+224)/2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Make the PIL image into a Numpy array
    np_image = np.array(pil_image)
    
    # Convert the RGB values from a scale of 0 to 255 into a scale of 0.0 to 1.0
    np_image = np_image / 255
    
    # Normalize the image
    np_image = (np_image - mean) / std
    
    # Reorder the array so that the colour channel is first
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
        

def predict(image_file, model, topk):
    print('Running predict()....')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = process_image(image_file)
    
    # Convert the NumPy array into a Tensor
    image_t = torch.from_numpy(image).type(torch.FloatTensor)
    image_t = image_t.unsqueeze(0)
    image_t = image_t.to(device)
    
    model.eval()
    
    with torch.no_grad():
        logits = model.forward(image_t)
        probs, labels = torch.topk(logits, topk)
        probs = probs.exp()
        class_to_idx = model.class_to_idx
        
    probs = probs.cpu().numpy()
    labels = labels.cpu().numpy()
    
    classes_indexed = {model.class_to_idx[i]: i for i in model.class_to_idx}
    classes_list = list()
    
    for label in labels[0]:
        classes_list.append(classes_indexed[label])
        
    return probs[0], classes_list

def show_results(probs, classes):
    print('Running show_results()....')
    with open(categories, 'r') as f:
        cat_to_name = json.load(f)
    flower_names = [cat_to_name[i] for i in classes]
    print('')
    
    df = pd.DataFrame(
        {'Class': pd.Series(data=classes),
         'Flower': pd.Series(data=flower_names),
         'Probability': pd.Series(data=probs)
        })
    print(df)
    true_cat = image_file.split('/')[-2]
    print('The correct flower name is: {}'.format(cat_to_name[true_cat]))
    
def main():
    global image_file, model, class_to_idx, categories, device, top_k_classes

    pred_args = get_predict_args()
    data_dir = pred_args.data_dir
    checkpoint_file = pred_args.checkpoint
    image_file = data_dir + pred_args.image_file
    categories = pred_args.categories
    if (pred_args.to_device == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    top_k_classes = pred_args.top_k_classes
    
    print('Checkpoint file: ', checkpoint_file)
    print('Image file: ', image_file)
    
    model, class_to_idx = load_checkpoint(checkpoint_file)
    probs, classes = predict(image_file, model, top_k_classes)
    show_results(probs, classes)
    
    print('\npredict.py is finished running')
    return

main()