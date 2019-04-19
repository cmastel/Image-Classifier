# model_util.py
#
# PROGRAMMER: Chris Mastel
# DATE CREATED: 11-April-2019
# DATE REVISED: 16-April-2019
# PURPOSE: Utility functions to load get user input information
#
# Import modules
import argparse
import os
import errno

def get_input_args():
    """
    Retrieves and parses the command line arguments provided when the user initiates the program.
    
    Command Line Arguments:
        - File path the folder containing the images: --data_dir
        - Learning rate of the model: --learn_rate with a default value of 0.001
        - Number of Epochs the model will train over: --epochs with a default value of 3
        - Model Architecture: --arch with a choice of densenet121, but a default of VGG16
        - Hidden Units - Level One: --hidden_one are the number of units in hidden layer one, default of 512
        - Hidden Units - Level Two: --hidden_two are the number of units in hidden layer two, default of 256
        
    Returns:
        - parse_args() --> Data structure that stores the Command Line Arguments object
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Command Line Arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers/', 
                        help = 'Path to the folder containing images')
    parser.add_argument('--learn_rate', type = float, default = 0.001, 
                        help = 'Model learning rate')
    parser.add_argument('--epochs', type = int, default = 1, 
                        help = 'Number of Epochs the model will train over')
    parser.add_argument('--arch', type = str, default = 'vgg16', 
                        help = 'Model architecture (vgg16 (default), densenet121)')
    parser.add_argument('--hidden_one', type = int, default = 512,
                        help = 'Units in hidden layer one (default = 512)')
    parser.add_argument('--hidden_two', type = int, default = 256,
                        help = 'Units in hidden layer two (defauly = 256)')
    parser.add_argument('--to_device', type = str, default = 'gpu',
                        help = 'The model can be trained on gpu (default) or cpu')
    
    return parser.parse_args()

def get_predict_args():
    """
    Retrieves and parses the command line arguments provided when the user initiates the program.
    
    Command Line Arguments:
        - File path the folder containing the images: --data_dir
        - File name of the desired image --image_file
        
    Returns:
        - parse_args() --> Data structure that stores the Command Line Arguments object
    """
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Command Line Arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers/test/99/', 
                        help = 'Path to the folder containing images')
        # Alternative default folder: flowers/test/3/
    parser.add_argument('--checkpoint', type = str, default = 'py_checkpoint.pth', 
                        help = 'File name of the checkpoint')
    parser.add_argument('--image_file', type = str, default = 'image_07874.jpg', 
                        help = 'File name of the sample image')
        # Alternative default file: image_06634.jpg
    parser.add_argument('--categories', type = str, default = 'cat_to_name.json',
                        help = 'File name of .json file containing category labels')
    parser.add_argument('--to_device', type = str, default = 'gpu',
                        help = 'The model can be trained on gpu (default) or cpu')
    parser.add_argument('--top_k_classes', type = int, default = 5,
                        help = 'Number of most likely classes to be shown')
    
    return parser.parse_args()
