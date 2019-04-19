# IMAGE CLASSIFIER COMMAND LINE APPLICATION
#
# Usage:
#   python train.py --data_dir --learn_rate 0.001 --epochs 3
#
# PROGRAMMER: Chris Mastel
# DATE CREATED: 11-April-2019
# DATE REVISED: 16-April-2019
# PURPOSE: Will train a neural network on a data set of images, and save the model as a checkpoint.
#           This function inputs:
#               - a path to the location of the folder containing images
#               - Learning Rate (optional, default is 0.001)
#               - Epochs (optional, default is 3)
#           This funtions outputs:
#               - Training Loss
#               - Validation Loss
#               - Validation Accuracy
#               - Creates a Checkpoint file which contains the details of trained model
#
# Import modules
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import json
import time
from model_utils import get_input_args

# Define global variables
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
lr = 0.001
epochs = 3
arch = 'vgg16'
hidden_one = 512
hidden_two = 256
device = 'cuda'
data_transforms = {}
image_datasets = {}

def transformations():
    global data_transforms, image_datasets, dataloaders
    global data_dir, train_dir, valid_dir, test_dir
    print('running transformations()....')
    # Create the transformation for each type of dataset
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['valid'])
    }
    
    # Using the image datasets and the transforms, define the dataloaders
    batch_size = 64
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
    }
    
    return
        
def mapping():
    print('mapping label info to actual names....')
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    # Print out label/name pairs if desired
    #for cat_key, cat_value in sorted(cat_to_name.items(), key=lambda x: x[1]): 
    #    print("{}: {}".format(cat_key, cat_value))

    return

def train_network():
    global model, device, data_transforms, image_datasets, dataloaders
    global epochs, lr, optimizer, criterion, hidden_one, hidden_two
    print('Training model on {} using the {} architecture'.format(device, arch))
    print('Learning rate: ', lr)
    print('Epochs: ', epochs)
    print('training the neural network...')
    print('         (this part can take a while)')
    print('Hidden One: {}, Hidden Two: {}'.format(hidden_one, hidden_two))
          

    start_time = time.time()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the number of input features based on the architecture used
    if (arch == 'vgg16'):
        num_features = model.classifier[0].in_features
    else:
        num_features = model.classifier.in_features
    print('Inpute features: ', num_features)
    
    # Define the classifier, criterion, optimizer
    model.classifier = nn.Sequential(nn.Linear(num_features, hidden_one),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_one, hidden_two),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_two, 102),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device)
    
    # Train the model
    steps = 0
    print_every = 10
    for epoch in range(epochs):
        running_loss = 0
        model.to(device)
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps) # Calculate accuracy
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Step: {}... ".format(steps),
                      "Training Loss: {:.4f}...".format(running_loss/len(dataloaders['train'])),
                      "Validation Loss: {:.4f}...".format(valid_loss/len(dataloaders['valid'])),
                      "Validation Accuracy: {:.4f}...".format(accuracy/len(dataloaders['valid']))
                     )
    
    total_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time//60, total_time%60))
    return

def test_network():
    global model, data_transforms, image_datasets, dataloaders
    print('running test_network()...')
    
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps,labels)
            test_loss += batch_loss.item()
            ps = torch.exp(logps) # Calculate accuracy
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print("Test Loss: {:.4f}...".format(test_loss/len(dataloaders['test'])),
          "Test Accuracy: {:.4f}...".format(accuracy/len(dataloaders['test'])))
          
    return

def save_network():
    global model, optimizer, loss, epochs, lr
    print('running save_network()...')
          
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {'model': model,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'learn_rate': lr,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'criterion': criterion,
                  'class_to_idx': model.class_to_idx
                 }
    torch.save(checkpoint, 'py_checkpoint.pth')
    return
          
def main():
    global data_dir, train_dir, valid_dir, test_dir
    global model, device, epochs, lr, arch, hidden_one, hidden_two
    in_arg = get_input_args()
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    lr = in_arg.learn_rate
    epochs = in_arg.epochs
    arch = in_arg.arch
    if (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif (arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print('\nError, train.py supports either vgg16 or densenet121')
        return
    hidden_one = in_arg.hidden_one
    hidden_two = in_arg.hidden_two
    if (in_arg.to_device == 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
    
    transformations()
    mapping()
    train_network()
    test_network()
    save_network()
    
    return

main()
