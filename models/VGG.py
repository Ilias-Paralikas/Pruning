import torch 
import torch.nn as nn


class myVGG(nn.Module):
    def __init__(self):
        super(myVGG, self).__init__()  # Call the parent class's initializer
        block1 = self.create_block(3, 64)  # Create the first block
        block2 = self.create_block(64, 64)  # Create the second block
        block3 = self.create_block(64, 128)
        block4 = self.create_block(128, 128)
        self.net = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            nn.Sequential(
            nn.Flatten(),
            nn.Linear(73728, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
            )
        )

    def forward(self, x):
        return self.net(x)

    def create_block(self, in_channels, filters, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size, padding),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )
        
        