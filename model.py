from os import name
from sympy import riemann_xi
import torch
from torch import Tensor, mode
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
        in_channels = 3

        for out_channels in features:
            self.conv_downs.append(UnetConv(in_channels=in_channels, out_channels=out_channels))
            self.conv_downs.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels

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
            print("down shape", x.shape)
            if isinstance(module, UnetConv):
                out_features.append(x)
        
        x = self.bottleneck_layer(x)
        print(x.shape,'bottleneck')
        out_features = out_features[::-1]

        i = 0
        for module in self.conv_ups:
            # import code; code.interact(local=locals())
            # resize out_feature value to x same shape
            if isinstance(module, UnetConv):
                height, width = x.shape[-2], x.shape[-1]
                resized_features = Resize((height, width))(out_features[i])
                i+=1
                print(x.shape)
                print(resized_features.shape)
                assert x.shape[1] == resized_features.shape[1] and x.shape[-1] == resized_features.shape[-1], "size should match"
                x = torch.concat([x, resized_features], dim=1)
                print(f'x after concat {x.shape}')
                
            x = module(x)
            print("up", x.shape)
        
        return self.output_layer(x)
                


        


    





def main():
    x = torch.rand((1, 3, 572, 572))
    conv = UnetConv(3, 64)
    print(f"UnetConv{x.shape} -> unet_conv -> {conv(x).size()}")

    model = UNET()

    print(f"{x.size()} -> model(x) -> {model(x).size()}")


if __name__ == '__main__':
    print('Checking..')
    main()
    