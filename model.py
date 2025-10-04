import torch
from torch import Tensor
import torch.nn as nn
from torchvision.transforms import Resize 

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, stride=1):
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
    def __init__(self, features=[64, 128, 256, 512]):
        super().__init__()
        self.conv_downs = nn.ModuleList()
        self.conv_ups = nn.ModuleList()
        in_channels = 1

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
            nn.Conv2d(in_channels=features[0], out_channels=1, kernel_size=1, stride=1),
            nn.LeakyReLU()
        )
    
    def forward(self, x: Tensor):
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
                height, width = x.shape[-2], x.shape[-1]
                resized_features = Resize((height, width))(out_features[i])
                i+=1
                assert x.shape[1] == resized_features.shape[1] and x.shape[-1] == resized_features.shape[-1], "size should match"
                x = torch.concat([x, resized_features], dim=1)    
            x = module(x)

        return self.output_layer(x)
                

def main():
    x = torch.rand((1, 1, 572, 572))
    conv = UnetConv(1, 64)
    print(f"UnetConv{x.shape} -> unet_conv -> {conv(x).size()}")

    model = UNET()

    print(f"UNET MODEL {x.size()} -> model(x) -> {model(x).size()}")


if __name__ == '__main__':
    print('Checking..')
    main()
    