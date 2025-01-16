import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self,num_class = 2):
        super(SimpleCNN,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*28*28,1024),
            nn.Linear(1024,64),
            nn.Dropout(p=0.5),
            nn.Linear(64,num_class)
        )

    def forward(self,x):
        x = self.model(x)
        return x

def simplecnn(nc):
    return SimpleCNN(nc)

if __name__ == '__main__':
    xiaodai = SimpleCNN()
    input = torch.ones((64,3,224,224))
    output = xiaodai(input)
    print(output.shape)