from pyexpat import model
import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize 
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.transforms.transforms import CenterCrop


class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, stride=1):
        super().__init__()

        self.conv_layer  = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding, stride=stride),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return self.conv_layer(x)
    

class UNET(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.conv_downs = nn.ModuleList()
        self.conv_ups = nn.ModuleList() 

        # Module list U down part
        for out_channels in features:
            self.conv_downs.append(UnetConv(in_channels=in_channels, out_channels=out_channels))
            self.conv_downs.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

        # Module list U up part
        for out_channels in reversed(features):
            in_channels = out_channels*2 # skip connection, add dim
            self.conv_ups.append(UnetConv(in_channels=in_channels, out_channels=out_channels))
            if out_channels != features[0]:
                self.conv_ups.append(nn.ConvTranspose2d(in_channels=out_channels, out_channels=int(out_channels/2), kernel_size=2, stride=2))

        
        self.bottleneck_layer = nn.Sequential(
            UnetConv(in_channels=features[-1], out_channels=features[-1]*2),
            nn.ConvTranspose2d(in_channels=features[-1]*2, out_channels=features[-1], kernel_size=2, stride=2)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=features[0], out_channels=3, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
    
    def forward(self, x: Tensor, target=None):
        out_features = []

        for module in self.conv_downs:
            x = module(x)
            if isinstance(module, UnetConv):
                out_features.append(x)

        x = self.bottleneck_layer(x)

        out_features = out_features[::-1]
        i = 0
        for module in self.conv_ups:
            # resize out_feature value to x same shape
            if isinstance(module, UnetConv):
                x = torch.concat([x, out_features[i]], dim=1)
                i+=1    
            x = module(x)
    
        model_out = self.output_layer(x)
        loss = None

        if target is not None:
            loss = binary_cross_entropy_with_logits(model_out, target)

        return model_out, loss
                

def main():
    x = torch.rand((1, 3, 256, 256))
    conv = UnetConv(in_channels=3, out_channels=64) # 3 channel
    print(f"UnetConv{x.shape} -> unet_conv -> {conv(x).size()}")

    model = UNET(in_channels=3)

    print(f"UNET MODEL {x.size()} -> model(x) -> {model(x)[0].size()}")


if __name__ == '__main__':
    print('Checking..')
    main()
    