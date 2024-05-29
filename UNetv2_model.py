import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class UNetv2(nn.Module):
    def __init__(self):
        super(UNetv2, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Output layer
        self.output = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def pad_and_concat(self, upsampled, bypass):
        diff_y = bypass.size(2) - upsampled.size(2)
        diff_x = bypass.size(3) - upsampled.size(3)

        upsampled = F.pad(upsampled, (diff_x // 2, diff_x - diff_x // 2,
                                      diff_y // 2, diff_y - diff_y // 2))
        return torch.cat((upsampled, bypass), dim=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        # Decoder
        dec4 = self.dec4(self.pad_and_concat(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4))
        dec3 = self.dec3(self.pad_and_concat(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3))
        dec2 = self.dec2(self.pad_and_concat(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2))
        dec1 = self.dec1(self.pad_and_concat(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1))

        return torch.sigmoid(self.output(dec1))


if __name__ == '__main__':
    model = UNetv2()
    x = torch.randn((1, 3, 256, 256))  # Example input
    out = model(x)
    print(out.shape)
