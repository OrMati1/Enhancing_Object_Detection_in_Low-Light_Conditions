import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MobileNetV3SmallUNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV3SmallUNet, self).__init__()
        # Load the pretrained MobileNetV3 small model
        self.mobilenet_encoder = models.mobilenet_v3_small(pretrained=pretrained).features

        # Decoder
        self.dec4 = self.conv_block(576 + 40, 256)  # Concatenating x4 and x3
        self.dec3 = self.conv_block(256 + 24, 128)  # Concatenating dec4 and x2
        self.dec2 = self.conv_block(128 + 16, 64)   # Concatenating dec3 and x1
        self.dec1 = self.conv_block(64 + 3, 64)    # Concatenating dec2 and original input

        # Final Classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def pad_and_concat(self, upsampled, bypass):
        """Pad upsampled to the size of bypass and concatenate along channel dimension."""
        diffY = bypass.size()[2] - upsampled.size()[2]
        diffX = bypass.size()[3] - upsampled.size()[3]

        upsampled = F.pad(upsampled, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x):
        # Encoder using MobileNetV3
        x0 = x  # original
        x1 = self.mobilenet_encoder[:2](x)  # Output channels: 16
        x2 = self.mobilenet_encoder[2:4](x1)  # Output channels: 24
        x3 = self.mobilenet_encoder[4:7](x2)  # Output channels: 40
        x4 = self.mobilenet_encoder[7:](x3)  # Output channels: 576

        # Decoder with skip connections
        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pad_and_concat(x, x3)
        x = self.dec4(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pad_and_concat(x, x2)
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pad_and_concat(x, x1)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.pad_and_concat(x, x0)
        x = self.dec1(x)

        # Final output
        out = self.final_conv(x)
        return out

# Create an instance of the model
model = MobileNetV3SmallUNet(num_classes=3)

# Test the forward pass with an example input
input_tensor = torch.randn(1, 3, 256, 256)  # Example input tensor
try:
    output = model(input_tensor)
    print("Output shape:", output.shape)
except Exception as e:
    print("Error during forward pass:", str(e))
