import torch
import torch.nn as nn


class sarNet_VAE(nn.Module):
    """
    A variational autoencoder (VAE) with a semi-dense autoregressive network as a decoder for image reconstruction.

    Parameters
    ----------
    layers : list of SemiDense and activation functions
        The layers of the neural network.
    recon_loss : function
        The reconstruction loss function to be used.
    latent_dim : int
        The dimension of the latent space.
    image_dim : int
        The dimension of the input image (assumed to be square).

    Attributes
    ----------
    latent_dim : int
        The dimension of the latent space.
    encoder : nn.Sequential
        The encoder part of the VAE, consisting of convolutional layers.
    fc_mu : nn.Linear
        Fully connected layer to compute the mean of the latent space.
    fc_logvar : nn.Linear
        Fully connected layer to compute the log variance of the latent space.
    decoder : latent_sarNET
        The decoder part of the VAE, utilizing a semi-dense autoregressive network.
    """

    def __init__(self, layers, recon_loss, latent_dim, image_dim):
        """
        Initialize the sarNet_VAE model with layers, reconstruction loss, latent dimension, and image dimensions.

        Parameters
        ----------
        layers : list of SemiDense and activation functions
            The layers of the neural network.
        recon_loss : function
            The reconstruction loss function to be used.
        latent_dim : int
            The dimension of the latent space.
        image_dim : int
            The dimension of the input/output image (assumed to be square).
        """

        super().__init__()
        self.latent_dim = latent_dim

        # Convolutional encoder: Converts the input image to a latent representation
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7),
        )

        # Fully connected layers for mean and log variance of the latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: Uses a semi-dense autoregressive network to reconstruct the image autoregressively from the latent space
        self.decoder = latent_sarNET(layers, recon_loss, image_dim)

    def encode(self, x):
        """
        Encode the input image into the latent space.
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterize the latent space to sample z from the latent space distribution.
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z):
        """
        Decode the latent vector z to reconstruct the image autoregressively.
        """
        h = z
        h = h.view(h.size(0), -1)
        return self.decoder(x, h)

    def forward(self, x):
        """
        Perform the forward pass of the network.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(x, z), mu, logvar

    def to(self, *args, **kwargs):
        """
        Move the model and its parameters to a specified device.
        """
        super().to(*args, **kwargs)
        for c in self.children():
            c.to(args[0])


class latent_sarNET(nn.Module):
    """
    A semi-dense autoregressive network with latent vectors for image reconstruction.

    Parameters
    ----------
    layers : list of SemiDense and activation functions
        The layers of the neural network.
    recon_loss : function
        The reconstruction loss function to be used.
    image_dim : int
        The dimension of the input image (assumed to be square).

    Attributes
    ----------
    layers : nn.ModuleList
        A list of neural network layers.
    recon_loss : function
        The reconstruction loss function.
    image_dim : int
        The dimension of the input/output image.
    """

    def __init__(self, layers, recon_loss, image_dim):
        """
        Initialize the latent_sarNET model with layers, reconstruction loss, and image dimensions.

        Parameters
        ----------
        layers : list of SemiDense and activation functions
            The layers of the neural network. Each SemiDense layer should be on even positin in the list.
        recon_loss : function
            The reconstruction loss function to be used.
        image_dim : int
            The dimension of the input/output image (assumed to be square).
        """
        super().__init__()
        self.layers = layers
        self.recon_loss = recon_loss
        self.image_dim = image_dim

    def forward(self, x, z):
        """
        Perform the forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        z : torch.Tensor
            The latent space tensor to be concatenated at each even layer.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the network layers.
        """
        x = x.view(x.size(0), -1)

        for i, l in enumerate(self.layers):
            if i % 2 == 0:
                x = torch.concat([x, z], dim=1)
            x = self.layers[i](x)

        # Reshape the output to the original image dimensions
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
        # Move all layers to the specified device
        for c in self.layers:
            c.to(args[0])
