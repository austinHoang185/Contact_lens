import warnings
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from GoogleNet.utils import load_state_dict_from_url


__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _GoogLeNetOutputs set here for backwards compat
_GoogLeNetOutputs = GoogLeNetOutputs


class GoogLeNet(nn.Module):
    # __constants__ = ['transform_input']

    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False, init_weights=True,
                 blocks=None, in_channels = 3):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception]
        assert len(blocks) == 2
        conv_block = blocks[0]
        inception_block = blocks[1]
        
        self.feature_extractor = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            conv_block(64, 64, kernel_size=1),
            conv_block(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),

            inception_block(192, 64, 96, 128, 16, 32, 32),
            inception_block(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),

            inception_block(480, 192, 96, 208, 16, 48, 64),
            inception_block(512, 160, 112, 224, 24, 64, 64),
            inception_block(512, 128, 128, 256, 24, 64, 64),
            inception_block(512, 112, 144, 288, 32, 64, 64),
            inception_block(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            
            inception_block(832, 256, 160, 320, 32, 128, 128),
            inception_block(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # @torch.jit.unused
    # def eager_outputs(self, x):
    #         return x
    
    # def forward(self, x):
    #     # type: (Tensor) -> GoogLeNetOutputs
    #     x = self._forward(x)
    #     aux_defined = self.training and False
    #     if torch.jit.is_scripting():
    #         if not aux_defined:
    #             warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
    #         return GoogLeNetOutputs(x)
    #     else:
    #         return self.eager_outputs(x)


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                 conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)





class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
