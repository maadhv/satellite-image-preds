
import torch.nn as nn

class network(nn.Module):

  ''' replicates tinyvgg model architecture'''

  def __init__(self):
    super().__init__()

    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels = 3,out_channels = 10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=10,out_channels=10,
                  kernel_size=3,stride=1,padding=1, bias=False),
        nn.MaxPool2d(2)
    )

    self.block_3 = nn.Sequential(
        nn.Conv2d(in_channels = 10,out_channels = 10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2,stride=2))

    self.block_4 = nn.Sequential(
        nn.Conv2d(in_channels = 10,out_channels = 10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=10,out_channels=10,kernel_size=3,
                  stride=1,
                  padding=1, bias=False),
        nn.MaxPool2d(kernel_size=2,stride=2))

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=10*16,out_features=4))

  def forward(self,x):

    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    x= self.block_4(x)
    x = self.classifier(x)
    return x
