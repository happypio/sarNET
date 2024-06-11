import torch.nn as nn


class sarNET(nn.Module):
    """
    A semi-dense autoregressive network for image reconstruction.

    Parameters
    ----------
    layers : list of SemiDense and activation functions
        The layers of the neural network.
    recon_loss : function
        The reconstruction loss function to be used.
    image_dim : int
        The dimension of the input/output image (assumed to be square).

    Attributes
    ----------
    layers : nn.Module
        The layers of the neural network.
    recon_loss : function
        The reconstruction loss function.
    image_dim : int
        The dimension of the input/output image.
    """

    def __init__(self, layers, recon_loss, image_dim):
        """
        Initialize the sarNET model with layers, reconstruction loss, and image dimensions.

        Parameters
        ----------
        layers : list of SemiDense and activation functions
            The layers of the neural network.
        recon_loss : function
            The reconstruction loss function to be used.
        image_dim : int
            The dimension of the input/output image (assumed to be square).
        """

        super().__init__()
        self.layers = layers
        self.recon_loss = recon_loss
        self.image_dim = image_dim

    def forward(self, x):
        """
        Perform the forward pass of the network.
        """
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = x.view(x.size(0), -1, self.image_dim, self.image_dim)
        return x

    def loss(self, outputs, targets):
        """
        Compute the reconstruction loss.
        """
        return self.recon_loss(outputs, targets)

    def to(self, *args, **kwargs):
        """
        Move the model and its parameters to a specified device.
        """
        super().to(*args, **kwargs)
        for c in self.layers:
            c.to(args[0])
