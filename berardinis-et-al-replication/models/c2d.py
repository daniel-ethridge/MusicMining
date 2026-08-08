import torch
import torch.nn as nn
 


class C2DNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d()