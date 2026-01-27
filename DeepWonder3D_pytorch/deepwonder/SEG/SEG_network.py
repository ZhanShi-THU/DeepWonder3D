from .SEG_3DUnet import UNet3D, T_conv_net, TS_UNet3D
from .SEG_3DUnet import Unet4_single_no_end_3D, Unet4_no_end_3D
import torch.nn as nn
import torch.nn.functional as F
import torch
from .SEG_Unet import Unet, Unet3, Unet4


class SEG_Network_3D_Unet(nn.Module):
    """
    Wrapper for several 3D U-Net style segmentation backbones.

    Depending on ``UNet_type``, this module instantiates and forwards inputs
    through one of several 3D architectures for volumetric segmentation.

    Args:
        UNet_type (str): Backbone type. Supported values:
            - ``'3DUNet'`` -> ``UNet3D``.
            - ``'TS_UNet3D'`` -> ``TS_UNet3D`` (temporal-spatial variant).
            - ``'Unet4_single_no_end_3D'`` -> ``Unet4_single_no_end_3D``.
            - ``'Unet4_no_end_UNet3D'`` -> ``Unet4_no_end_3D``.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        frame_num (int): Temporal depth (number of frames) in the input.
        final_sigmoid (bool): If True, use sigmoid in backbone, otherwise
            softmax along the channel dimension where applicable.
        f_maps (int): Base number of feature maps in the network.
    """

    def __init__(
        self,
        UNet_type: str = "3DUNet",
        in_channels: int = 1,
        out_channels: int = 1,
        frame_num: int = 64,
        final_sigmoid: bool = True,
        f_maps: int = 64,
    ):
        super(SEG_Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid

        if UNet_type == "3DUNet":
            self.Generator = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                frame_num=frame_num,
                final_sigmoid=final_sigmoid,
                f_maps=f_maps,
            )
        elif UNet_type == "TS_UNet3D":
            self.Generator = TS_UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                frame_num=frame_num,
                final_sigmoid=final_sigmoid,
                f_maps=f_maps,
            )
        elif UNet_type == "Unet4_single_no_end_3D":
            self.Generator = Unet4_single_no_end_3D(
                in_ch=in_channels,
                out_ch=out_channels,
                frame_num=frame_num,
                f_num=f_maps,
            )
        elif UNet_type == "Unet4_no_end_UNet3D":
            print("network ---> ", UNet_type)
            self.Generator = Unet4_no_end_3D(
                in_ch=in_channels,
                out_ch=out_channels,
                frame_num=frame_num,
                f_num=f_maps,
            )
        else:
            raise ValueError(f"Unsupported UNet_type '{UNet_type}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 3D segmentation network.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_channels, depth, height, width)``.

        Returns:
            torch.Tensor: Segmentation logits of shape
            ``(batch, out_channels, depth', height', width')`` depending on
            the selected backbone.
        """
        fake_x = self.Generator(x)
        return fake_x


class Network_TSnet(nn.Module):
    """
    Two-stage temporal-spatial segmentation network.

    Stage 1 (``T_conv_net``) encodes temporal information across frames.
    Stage 2 (``Unet3``) performs 2D U-Net segmentation on the aggregated
    temporal features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        frame_num (int): Temporal depth (number of frames) in the input.
        f_maps (int): Base number of feature maps in the temporal conv net.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        frame_num: int = 64,
        f_maps: int = 64,
    ):
        super(Network_TSnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.T_conv_net = T_conv_net(
            in_channels=in_channels,
            frame_num=frame_num,
            tc_f_maps=f_maps,
        )
        self.Unet = Unet3(
            in_ch=f_maps,
            out_ch=out_channels,
            f_num=f_maps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the temporal-spatial segmentation network.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, in_channels, depth, height, width)``.

        Returns:
            torch.Tensor: 2D segmentation logits of shape
            ``(batch, out_channels, height, width)``.
        """
        pred_patch = self.T_conv_net(x)
        input_u = torch.squeeze(pred_patch, dim=2)
        output_u = self.Unet(input_u)
        return output_u
