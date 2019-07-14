# Imports here
import torch
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import time
import argparse
from dataloader import *

parser = argparse.ArgumentParser(description="Train the network to recognize images")
parser.add_argument('source_dir', type=str)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int)
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--epochs', type=int,default=3)


args = parser.parse_args()
print(args)
print(args.source_dir)
print(args.save_dir)
print(args.gpu)
print(args.arch)


image_datasets = create_datasets(args.source_dir)

dataloaders = {
    "train_dataloader": torch.utils.data.DataLoader(image_datasets["train_datasets"],batch_size=64, shuffle=True),
    "valid_dataloader": torch.utils.data.DataLoader(image_datasets["validation_dataset"],batch_size=64,shuffle=True),
    "test_dataloader" : torch.utils.data.DataLoader(image_datasets["test_dataset"],batch_size=64,shuffle=True)
}

device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
print(device)

if (args.arch == 'vgg16'):
    model = models.vgg16(pretrained=True)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 1024)),
                              ('relu', nn.ReLU()),
                              ('dropout', nn.Dropout(0.2)),
                              ('fc3', nn.Linear(1024, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
elif(args.arch == 'densenet121'):
    model = models.densenet121(pretrained=True) 
    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 102),
                                 nn.LogSoftmax(dim=1))
for param in model.parameters():
        param.requires_grad = False
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
model.to(device)
running_loss = 0
print("starting")
epochs = args.epochs
steps = 20
for epoch in range(epochs):
    for ii, (inputs, labels) in enumerate(dataloaders["train_dataloader"]):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        start = time.time()
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (ii + 1 ) % steps == 0:
            print("Running loss after " + str(ii + 1)  + " batches " + str(running_loss))
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                testloader = dataloaders["valid_dataloader"]
                for inputs,labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/steps:.3f}.. "
                  f"Validation loss: {test_loss/len(testloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
    
model.class_to_idx = image_datasets['train_datasets'].class_to_idx
checkpoint = {
    'epochs': 3,
    'class_to_idx' : model.class_to_idx,
    'state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}
torch.save(checkpoint,args.save_dir+args.arch+'_checkpoint.pth')


