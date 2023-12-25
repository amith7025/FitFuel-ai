import gradio as gr
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from PIL import Image
import numpy as np

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

test_transforms = T.Compose([
    T.Resize(size=(128,128)),
    T.ToTensor()
])

class FitFuelModel(nn.Module):
    def __init__(self,input_size=3,output_size=len(labels)):
        super().__init__()
        self.conv_blk1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,out_channels=32,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=238144,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=output_size)
        )

    def forward(self,x):
        x = self.conv_blk1(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

model = FitFuelModel()

model.load_state_dict(torch.load('models\prototype(beta_ 7).pth',map_location=torch.device('cpu')))

def prediction(input_image):
    pil_image = Image.fromarray(input_image)
    transformed = test_transforms(pil_image)

    model.eval()
    with torch.inference_mode():
        logits = model(transformed.unsqueeze(0))
        probs = torch.softmax(logits,dim=1)
        pred_labels_and_probs = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
        return pred_labels_and_probs

title = "FitFuel AI (MINGO AI)"
description = "a Narrow AI  to predict 10 different classes of image with CNN architecture"
article = "Created by Sharun,Vishnu,Amith"

# Create the Gradio demo
demo = gr.Interface(fn=prediction, # mapping function from input to output
                    inputs=gr.Image(), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions")], # our fn has two outputs, therefore we have two outputs 
                    title=title,
                    description=description,
                    article=article)


if __name__ == "__main__":
    demo.launch(share=True)