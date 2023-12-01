import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
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

def prediction(text:str):
    img = Image.open(text)
    transformed = test_transforms(img)

    model.load_state_dict(torch.load('models\prototype(beta_ 7).pth',map_location=torch.device('cpu')))

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
