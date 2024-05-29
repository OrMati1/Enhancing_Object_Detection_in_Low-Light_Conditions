import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3UNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV3UNet, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.mobilenet = mobilenet_v3_large(weights=weights)
        self.encoder_features = self.mobilenet.features

        # Define the decoder
        self.dec4 = self.conv_block(960 + 112, 256)  # Corrected channel sum (x4 upsampled + x3)
        self.dec3 = self.conv_block(256 + 40, 128)   # (Previous decoder output + x2)
        self.dec2 = self.conv_block(128 + 24, 64)    # (Previous decoder output + x1)
        self.dec1 = self.conv_block(64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def pad_and_concat(self, upsampled, bypass):
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x):
        x1 = self.encoder_features[0:3](x)   # Outputs 24 channels
        x2 = self.encoder_features[3:7](x1)  # Outputs 40 channels
        x3 = self.encoder_features[7:13](x2) # Outputs 112 channels
        x4 = self.encoder_features[13:](x3)  # Outputs 960 channels

        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pad_and_concat(x, x3)
        x = self.dec4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pad_and_concat(x, x2)
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.pad_and_concat(x, x1)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.dec1(x)

        out = self.final_conv(x)
        return out

def test_model():
    model = MobileNetV3UNet(num_classes=3, pretrained=True)
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)

    try:
        with torch.no_grad():
            output = model(input_tensor)
        print("Test successful!")
        print("Input shape:", input_tensor.shape)
        print("Output shape:", output.shape)
    except Exception as e:
        print("Test failed!")
        print("Error:", e)

if __name__ == "__main__":
    test_model()
