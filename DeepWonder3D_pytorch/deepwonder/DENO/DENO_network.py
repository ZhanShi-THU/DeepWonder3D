from deepwonder.DENO.DENO_model_3DUnet import UNet3D
import torch.nn as nn

class Network_3D_Unet(nn.Module):
    """
    3D U-Net network wrapper for denoising tasks.

    Wraps the UNet3D model to provide a consistent interface for denoising
    operations on 3D image volumes.
    """
    def __init__(self, UNet_type = '3DUNet', in_channels=1, out_channels=1, f_maps=64, final_sigmoid = True):
        """
        Initialize the 3D U-Net network.

        Args:
            UNet_type (str, optional): Type of U-Net architecture. Default is '3DUNet'.
            in_channels (int, optional): Number of input channels. Default is 1.
            out_channels (int, optional): Number of output channels. Default is 1.
            f_maps (int, optional): Number of feature maps in the first layer. Default is 64.
            final_sigmoid (bool, optional): Whether to apply sigmoid activation at the end.
                Default is True.
        """
        super(Network_3D_Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_sigmoid = final_sigmoid

        if UNet_type == '3DUNet':
            self.Generator = UNet3D( in_channels = in_channels,
                                     out_channels = out_channels,
                                     f_maps = f_maps, 
                                     final_sigmoid = final_sigmoid)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channels, D, H, W)

        Returns:
            torch.Tensor: Denoised output tensor with shape (batch, out_channels, D, H, W)
        """
        fake_x = self.Generator(x)
        return fake_x
