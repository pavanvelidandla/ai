import torch
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict


def transformations():
  
    data_transforms = {"train_transform" : transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),

      "validation_or_test_transform": transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]) }
    
    return data_transforms

def create_datasets(source_dir):
    
    data_dir = source_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = transformations()
    image_datasets = {
        "train_datasets" : datasets.ImageFolder(train_dir, transform=data_transforms["train_transform"]),
        "validation_dataset": datasets.ImageFolder(valid_dir, transform=data_transforms["validation_or_test_transform"]),
        "test_dataset": datasets.ImageFolder(test_dir, transform=data_transforms["validation_or_test_transform"]),
     }
    return image_datasets



    