import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(DoubleConv, self).__init__()
#         print(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False), # bias = False becaise BatchNorm2d is set
            nn.BatchNorm2d(out_channels), # BatchNorm2d were not known when paper came out
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features = [64, 128, 256, 512]):
        super(UNET, self).__init__()
                
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # keep size the same
        

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
#         print(self.ups)
        
            

    def forward(self, x):
        
        def print_sizes(model, input_tensor):
            output = input_tensor
            for m in model.children():
                output = m(output)
                print(m, output.shape)
            return output
#         print('start forward')
                        
        x = x.unsqueeze(1)
        skip_connections = []
        
#         print(x.shape)

        for down in self.downs:
#             print('going down')
#             print(x.shape)
            x = down(x)
#             print(x.shape)
            skip_connections.append(x)
            x = self.pool(x)
#             print('after pooling', x.shape)
#             print(x.shape)
       
        
#         print('before bottleneck', x.shape)
        x = self.bottleneck(x)
#         print('after bottleneck', x.shape)
        skip_connections = skip_connections[::-1] # reverse list
#         print(x.shape)

        # the upsampling
        for idx in range(0, len(self.ups), 2): # step of 2 because we want up - double column - up - double column
#             print('going up')
#             print(x.shape, idx)
            x = self.ups[idx](x)
#             print(x.shape, idx//2)
            skip_connection = skip_connections[idx//2] # //2 because we want still steps of one
#             print(x.shape)

            # if statement because we can put in shapes that are not divisble by two around 19:00 of video
            if x.shape != skip_connection.shape: 
                x = TF.resize(x, size=skip_connection.shape[2:]) # hopefully does not impact accuracy too much
#             print(x.shape, idx+1)
            concat_skip = torch.cat((skip_connection, x), dim=1)
#             print('concat shape', concat_skip.shape)
            x = self.ups[idx+1](concat_skip)
#             print(x.shape)
#             print(x.shape)
#         print('before final conv', x.shape)
        x = self.final_conv(x)
#         print('after final conv', x.shape)
        return x