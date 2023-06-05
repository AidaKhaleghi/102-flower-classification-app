import json

import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch.autograd import Variable

from data import DATA_DIR

path = str(DATA_DIR)
checkpoint_path = path + 'flower_classifier.pth'

# flower categorie name
with open(path + 'cat_to_name.json', 'r') as f:
    category_to_name = json.load(f)


# Loading checkpoint
def load_checkpoint(path):

    # Load in checkpoint
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # Assigning the model
    model = models.densenet161(pretrained=True)

    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Assigning the trained classifier
    model.classifier = checkpoint['classifier']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Move to gpu
    if torch.cuda.is_available():
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def process_image(image):
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485)/(0.229)
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def predict(image, model, topk=5):
    # Implement the code to predict the class from an image file
    image = torch.FloatTensor([process_image(image)])
    model.eval()
    model.to('cpu')
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1]
    top_class = [model.idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class


model, optimizer = load_checkpoint(path=checkpoint_path)

# Creating a idx to class mapping dict
model.idx_to_class = {}
for i, idx in model.class_to_idx.items():
    model.idx_to_class[idx] = category_to_name[i]
