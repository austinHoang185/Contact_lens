#model
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from utils import GrayscaleToRgb, GradientReversal
import torchvision.models as model



class DANet(nn.Module):
    def __init__(self, input: int = 1024):
        super(DANet, self).__init__()
        self.GRL =GradientReversal()
        self.l1 = nn.Linear(input, 512)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(512, 50)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(50,1)
           
    def forward(self, x):
        x = self.GRL(x)
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        x = self.l3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=False )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 3, dropout: float = 0.4) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def ModelInit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Efficientnet().to(device)
    model
    return model

if __name__=="__main__":
    model = ModelInit()
    print(model)