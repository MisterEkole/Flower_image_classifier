from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse, json


def load_checkpoint(filepath, cuda=False):
    if not cuda:
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        checkpoint = torch.load(filepath)

    model = models.densenet121(pretrained=True)
    
    
    for params in model.parameters():
        params.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 200)),
        ('relu', nn.ReLU()), 
        ('fc2', nn.Linear(200, 102)),
        ('drop', nn.Dropout(p=0.5)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    classifier.load_state_dict(checkpoint['state_dict'])
    
    
    model.classifier = classifier
    class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = class_to_idx
    model.idx_to_class = inv_map = {v: k for k, v in class_to_idx.items()}
    
    return model

def process_image(image):
 
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
        ])
    
    image = img_transforms(Image.open(image))
    
    return image

def predict_from_checkpoint(image_path, checkpoint, topk=5, category_names=None, cuda=False):
 
    
    # TODO: Implement the code to predict the class from an image file
    model = load_checkpoint(checkpoint, cuda=cuda)
    image_data = process_image(image_path)
    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    model_p = model.eval()
    
    inputs = variable(image_data.unsqueeze(0))

    if cuda:
        inputs = inputs.cuda()
    
    output = model_p(inputs)
    ps = torch.exp(output).data
    
    ps_top = ps.topk(topk)
    idx2class = model.idx_to_class
    probs = ps_top[0].tolist()[0]
    classes = [idx2class[i] for i in ps_top[1].tolist()[0]]


    
    class_names = "Unknown"
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        class_names = [cat_to_name[i] for i in classes]


    
    return probs, classes, class_names

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Predicts the name of a flower from an image with the occurence probability of that name ')
    parser.add_argument('input', type=str, help='path to image')
    parser.add_argument('checkpoint', type=str, help='checkpoint for inference')
    parser.add_argument('--top_k', type=int, help='returns top k most likely classes')
    parser.add_argument('--category_names', type=str, help=' Mapping of categories to real names from a json file')
    parser.add_argument('--gpu', action='store_true', help='GPU for inference if available')

    # Parse and read arguments and assign them to variables if exists 
    args, _ = parser.parse_known_args()

    image_path = args.input
    checkpoint = args.checkpoint

    top_k = 1
    if args.top_k:
        top_k = args.top_k

    category_names = None
    if args.category_names:
        category_names = args.category_names

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("No GPU is available in the machine")

    probs= predict_from_checkpoint(image_path, checkpoint, topk=top_k, category_names=category_names, cuda=cuda)
    
    classes=predict_from_checkpoint(image_path, checkpoint, topk=top_k, category_names=category_names, cuda=cuda)
    
    class_names= predict_from_checkpoint(image_path, checkpoint, topk=top_k, category_names=category_names, cuda=cuda)
 
    print(f"inputted labels = {classes}")
    
    
    
    print(f"confidence probability = {probs}")
    
    
    
    print(f"names of classes = {class_names}")
 
    



if __name__ == '__main__':
    main()