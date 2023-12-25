import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as tf
from PIL import Image 

labels = {
    0: 'vada pav',
    1: 'tandoori chicken',
    2: 'idly',
    3: 'meduvadai',
    4: 'samosa',
    5: 'kathi roll',
    6: 'halwa',
    7: 'biriyani',
    8: 'gulab jamun',
    9: 'dosa'
}

weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT

auto_transforms = weights.transforms()
model = torchvision.models.efficientnet_b1(weights=weights)

for params in model.features.parameters():
    params.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=10, # same number of output units as our number of classes
                    bias=True)
)



def prediction(text:str):
    img = Image.open(text)
    transformed = auto_transforms(img)

    model.load_state_dict(torch.load('transfer.pth',map_location=torch.device('cpu')))

    model.eval()
    with torch.inference_mode():
        logits = model(transformed.unsqueeze(0))
        print(f'logits: {logits}')
        probs = torch.softmax(logits,dim=1)
        print(f'probability: {probs}')
        output = torch.argmax(probs,dim=1)
        index = output.data.item()
        print(f'predicted class: {labels[index]}')

if __name__ == '__main__':
    text = input('enter the loc of image')
    prediction(text)