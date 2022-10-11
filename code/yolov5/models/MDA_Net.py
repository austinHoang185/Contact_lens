from PIL import Image
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class discriminator1(nn.Module):
    def __init__(self):
        super(discriminator1, self).__init__()
        self.GRL = GradientReversal()
        self.conv1 = nn.Conv2d(384,256,stride=2,padding=1,kernel_size=3 )
        self.SiLU1 = nn.SiLU()
        self.conv2 = nn.Conv2d(256, 1,stride=1,padding=0,kernel_size=3 )
        self.SiLU2 = nn.SiLU()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(324,1)
           
    def forward(self, x):
        x = self.GRL(x)
        x = self.conv1(x)
        x = self.SiLU1(x)
        x = self.conv2(x)
        x = self.SiLU2(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.l1(x))
        return x

class discriminator2(nn.Module):
    def __init__(self):
        super(discriminator2, self).__init__()
        self.GRL = GradientReversal()
        self.conv1 = nn.Conv2d(384,256,stride=2,padding=1,kernel_size=3 )
        self.SiLU1 = nn.SiLU()
        self.conv2 = nn.Conv2d(256, 1,stride=1,padding=1,kernel_size=3 )
        self.SiLU2 = nn.SiLU()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(100,1)
           
    def forward(self, x):
        x = self.GRL(x)
        x = self.conv1(x)
        x = self.SiLU1(x)
        x = self.conv2(x)
        x = self.SiLU2(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.l1(x))
        return x

class discriminator3(nn.Module):
    def __init__(self):
        super(discriminator3, self).__init__()
        self.GRL = GradientReversal()
        self.conv1 = nn.Conv2d(768,512,stride=2,padding=1,kernel_size=3 )
        self.SiLU1 = nn.SiLU()
        self.conv2 = nn.Conv2d(512, 1,stride=1,padding=1,kernel_size=3 )
        self.SiLU2 = nn.SiLU()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(100,1)
           
    def forward(self, x):
        x = self.GRL(x)
        x = self.conv1(x)
        x = self.SiLU1(x)
        x = self.conv2(x)
        x = self.SiLU2(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.l1(x))
        return x




