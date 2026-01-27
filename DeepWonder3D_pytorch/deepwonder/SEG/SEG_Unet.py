import torch.nn as nn
import torch
from torch import autograd


class DoubleConv(nn.Module):
    """
    Two-layer 2D convolutional block with BatchNorm and ReLU.

    This module is a common building block for U-Net style architectures:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the double convolution block.

        Args:
            input (torch.Tensor): Input tensor of shape
                ``(batch, in_ch, height, width)``.

        Returns:
            torch.Tensor: Output tensor of shape
            ``(batch, out_ch, height, width)``.
        """
        return self.conv(input)


class Unet(nn.Module):
    """
    Classic 2D U-Net for segmentation.

    This is a 5-level U-Net encoder-decoder with skip connections and a
    final sigmoid activation for binary segmentation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
    """

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_ch, height, width)``.

        Returns:
            torch.Tensor: Sigmoid-activated logits of shape
            ``(batch, out_ch, height, width)``.
        """
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


class Unet4(nn.Module):
    """
    4-level 2D U-Net with configurable feature width.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        f_num (int): Base number of feature channels at the first level.
    """

    def __init__(self, in_ch, out_ch, f_num):
        super(Unet4, self).__init__()
        self.conv1 = DoubleConv(in_ch, f_num)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(f_num, f_num * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(f_num * 2, f_num * 2 * 2)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(f_num * 2 * 2, f_num * 2 * 2 * 2)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(f_num * 2 * 2 * 2, f_num * 2 * 2 * 2 * 2)
        self.up6 = nn.ConvTranspose2d(f_num * 2 * 2 * 2 * 2, f_num * 2 * 2 * 2, 2, stride=2)
        self.conv6 = DoubleConv(f_num * 2 * 2 * 2 * 2, f_num * 2 * 2 * 2)
        self.up7 = nn.ConvTranspose2d(f_num * 2 * 2 * 2, f_num * 2 * 2, 2, stride=2)
        self.conv7 = DoubleConv(f_num * 2 * 2 * 2, f_num * 2 * 2)
        self.up8 = nn.ConvTranspose2d(f_num * 2 * 2, f_num * 2, 2, stride=2)
        self.conv8 = DoubleConv(f_num * 2 * 2, f_num * 2)
        self.up9 = nn.ConvTranspose2d(f_num * 2, f_num, 2, stride=2)
        self.conv9 = DoubleConv(f_num * 2, f_num)
        self.conv10 = nn.Conv2d(f_num, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 4-level U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_ch, height, width)``.

        Returns:
            torch.Tensor: Raw logits of shape
            ``(batch, out_ch, height, width)``.
        """
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = c10
        return out


class Unet3(nn.Module):
    """
    3-level 2D U-Net variant used in temporal-spatial networks.

    Compared to ``Unet4``, this architecture has one fewer encoder/decoder
    stage and uses a symmetric layout for skip connections.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        f_num (int): Base number of feature channels at the first level.
    """

    def __init__(self, in_ch, out_ch, f_num):
        super(Unet3, self).__init__()
        self.conv1 = DoubleConv(in_ch, f_num)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(f_num, f_num * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(f_num * 2, f_num * 2 * 2)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(f_num * 2 * 2, f_num * 2 * 2)
        self.up7 = nn.ConvTranspose2d(f_num * 2 * 2, f_num * 2 * 2, 2, stride=2)
        self.conv7 = DoubleConv(f_num * 2 * 2 * 2, f_num * 2 * 2)
        self.up8 = nn.ConvTranspose2d(f_num * 2 * 2, f_num * 2, 2, stride=2)
        self.conv8 = DoubleConv(f_num * 2 * 2, f_num * 2)
        self.up9 = nn.ConvTranspose2d(f_num * 2, f_num, 2, stride=2)
        self.conv9 = DoubleConv(f_num * 2, f_num)
        self.conv10 = nn.Conv2d(f_num, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 3-level U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_ch, height, width)``.

        Returns:
            torch.Tensor: Sigmoid-activated logits of shape
            ``(batch, out_ch, height, width)``.
        """
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        up_7 = self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out