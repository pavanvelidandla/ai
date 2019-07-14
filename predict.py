import argparse
import json
import torch
from utils import *

parser = argparse.ArgumentParser("")
parser.add_argument('path_to_image', type=str)
parser.add_argument('path_to_checkpoint', type=str)
parser.add_argument('--top_k', type=int,default=1)
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--gpu ',  action='store_true')

args = parser.parse_args()



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device)
    model.eval()
    processed_img = process_image(image_path)
#     image = np.array(processed_img).transpose((1, 2, 0))
    image_tensor = torch.from_numpy(processed_img)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.float()
    image_tensor.to(device)
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(image_tensor)

    ps = torch.exp(output)
    top_probs, top_labs = ps.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    id_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [id_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[id_to_class[lab]] for lab in top_labs]    
    return top_probs, top_labels, top_flowers

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

probs, labels, flowers = predict(args.path_to_image, load_checkpoint(args.path_to_checkpoint),args.top_k)
for prob,flower in zip(probs,flowers):
    print ("The flower in the picture is predicted to be {} with probability {}".format(flower,prob))

