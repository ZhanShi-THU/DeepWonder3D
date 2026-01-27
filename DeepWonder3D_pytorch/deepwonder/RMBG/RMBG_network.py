from .RMBG_3DUnet import UNet3D, UNet3D_squeeze
import torch.nn as nn


class Network_3D_Unet(nn.Module):
    """
    Wrapper network that exposes either a standard 3D U-Net or a squeezed 3D U-Net.

    This module initializes one of the RMBG U-Net generator architectures and
    forwards the input volume through it.

    Args:
        UNet_type (str): Type of underlying U-Net backbone.
            - ``'3DUNet'`` uses ``UNet3D``.
            - ``'UNet3D_squeeze'`` uses ``UNet3D_squeeze``.
        in_channels (int): Number of channels in the input volume.
        out_channels (int): Number of channels in the output volume.
        f_maps (int): Base number of feature maps in the network.
        final_sigmoid (bool): If True, final activation in generator is sigmoid,
            otherwise softmax along the channel dimension.
        down_num (int): Number of down-sampling stages for ``UNet3D_squeeze``.
    """

    def __init__(
        self,
        UNet_type: str = "3DUNet",
        in_channels: int = 1,
        out_channels: int = 1,
        f_maps: int = 64,
        final_sigmoid: bool = True,
        down_num: int = 4,
    ):
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid

        if UNet_type == "3DUNet":
            self.Generator = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                final_sigmoid=final_sigmoid,
            )
        elif UNet_type == "UNet3D_squeeze":
            self.Generator = UNet3D_squeeze(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                final_sigmoid=final_sigmoid,
                down_num=down_num,
            )
        else:
            raise ValueError(f"Unsupported UNet_type '{UNet_type}'.")

    def forward(self, x):
        """
        Forward pass of the RMBG generator network.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_channels, depth, height, width)``.

        Returns:
            torch.Tensor: Output tensor of shape
            ``(batch, out_channels, depth, height, width)`` produced by the
            selected U-Net generator.
        """
        fake_x = self.Generator(x)
        return fake_x
