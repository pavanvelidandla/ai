from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch import nn , optim
from collections import OrderedDict

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    im_width,im_height = im.size
    ar = im_width / im_height
    resizes = (round(ar*256),256) if ar > 1 else (256,round(ar*256))
    im = im.resize(resizes)
    im_width,im_height = im.size
    im = im.crop((round((im_width - 224) / 2), round((im_height - 224)/2),round((im_width + 224) / 2), round((im_height + 224) / 2)))
    np_im = np.array(im) /255
    np_im = (np_im - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    np_im = np_im.transpose((2,0,1))
    return np_im

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# def view_classify(img_path,model):
#     plt.figure(figsize = (6,10))
#     ax = plt.subplot(2,1,1)
#     # Set up title
#     flower_num = img_path.split('/')[2]
#     title_ = cat_to_name[flower_num]
#     # Plot flower
#     img = process_image(img_path)
#     imshow(img, ax, title = title_);
#     # Make prediction
#     probs, labs, flowers = predict(img_path, model) 
#     # Plot bar chart
#     ax2 = plt.subplot(2,1,2)
#     ax2.set_yticklabels(flowers)
#     y_pos = np.arange(len(flowers))
#     ax2.set_yticks(y_pos)
#     ax2.barh(y_pos, probs, align='center')
#     ax2.invert_yaxis()
#     plt.show()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
#     optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(1024, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epochs']
    return model