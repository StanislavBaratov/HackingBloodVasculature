import torch
import torch.nn as nn

class DoubleConv2d(nn.Module):
    """Сверточный блок"""
    def __init__(self, 
                 input_channels=1, 
                 output_channels=32, 
                 kernel_size=3
                ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, 
                      output_channels, 
                      kernel_size, 
                      padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, 
                      output_channels, 
                      kernel_size, 
                      padding='same'),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """Модель архитектуры UNet"""
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        #==============Encoder block===========================
        self.conv1 = DoubleConv2d(1, 32, kernel_size=3) 
        self.conv2 = DoubleConv2d(32, 64, kernel_size=3)
        self.conv3 = DoubleConv2d(64, 128, kernel_size=3)
        self.conv4 = DoubleConv2d(128, 256, kernel_size=3)
        self.conv5 = DoubleConv2d(256, 512, kernel_size=3)
        #======================================================
        
        #=============Decoder block============================
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv2d(512, 256, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv2d(256, 128, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv2d(128, 64, kernel_size=3)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv2d(64, 32, kernel_size=3)
        self.gap = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        #======================================================
        
        
    def forward(self, x):
        #================Convolution===========================
        down1 = self.conv1(x) # 32 * H * W
        down2 = self.pool(down1) # 32 * H/2 * W/2
        down3 = self.conv2(down2) # 64 * H/2 * W/2
        down4 = self.pool(down3) # 64 * H/4 * W/4
        down5 = self.conv3(down4) # 128 * H/4 * W/4
        down6 = self.pool(down5) # 128 * H/8 * W/8
        down7 = self.conv4(down6) # 256 * H/8 * W/8
        down8 = self.pool(down7) # 256 * H/16 * W/16
        down9 = self.conv5(down8) # 512 * H/16 * W/16
        #======================================================
        
        #===============Deconcolution==========================
        up1 = self.deconv1(down9) # 256 * H/8 * W/8
        x = self.upconv1(torch.cat([down7, up1], dim=1)) # 256 * H/8 * W/8
        up2 = self.deconv2(x) # 128 * H/4 * W/4
        x = self.upconv2(torch.cat([down5, up2], dim=1)) # 128 * H/4 * W/4
        up3 = self.deconv3(x) # 64 * H/2 * W/2
        x = self.upconv3(torch.cat([down3, up3], dim=1)) # 64 * H/2 * W/2
        up4 = self.deconv4(x) # 32 * H * W
        x = self.upconv4(torch.cat([down1, up4], dim=1)) # 32 * H * W
        x = self.gap(x) # 1 * H * W
        #=======================================================
        return x
