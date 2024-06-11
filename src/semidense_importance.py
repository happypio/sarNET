import torch
import torch.nn as nn


class SemiDenseImportance(nn.Module):
    """
    A module to calculate the importance of patches in a semi-dense autoregressive network.

    Parameters
    ----------
    in_dim : int
        The input dimension of the image (height/width).
    out_dim : int
        The output dimension of the image (height/width).
    in_channels : int
        The number of input channels (e.g., 3 for RGB, 1 for black and white).
    out_channels : int
        The number of output channels.
    patch_dim_in : int
        The dimension of input patches.
    patch_dim_out : int
        The dimension of output patches.
    latent_dim : int
        The dimension of the latent space.
    inspected_layer: SemiDense
        The layer to calculate the importance of patches
    device : torch.device, optional
        The device to use for the model (e.g., 'cpu' or 'cuda').
    dtype : torch.dtype, optional
        The data type for the model parameters.
    shift : bool, optional
        Whether to shift the patches during processing
        (if true an i-th patch looks at 0,...i-1 patches, if false, it looks at 0,...,i patches).

    Attributes
    ----------
    in_features : int
        The number of input features for each patch.
    out_features : int
        The number of output features for each patch.
    weight : torch.nn.Parameter
        The learnable weights of the linear transformation.
    bias : torch.nn.Parameter
        The learnable bias of the linear transformation.
    weight_mask : torch.Tensor
        The mask to apply to the weights to enforce the semi-dense connectivity.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        in_channels,
        out_channels,
        patch_dim_in,
        patch_dim_out,
        latent_dim,
        inspected_layer,
        device=None,
        dtype=None,
        shift=True,
    ):
        """
        Initialize the SemiDense layer with the given parameters.

        - patch_dim_in should be a divisor of in_dim
        - patch_dim_out should be a divisor of out_dim
        - out_dim should be divisible by (in_dim / patch_dim_in) - so that
            each in_patch will have the same number of neurons in an out_patch
        - the number of input patches should match the number of output patches -
            i.e. in_dim / patch_dim_in == out_dim / patch_dim_out

        """

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_features = in_dim * in_dim * in_channels + latent_dim
        self.out_features = out_dim * out_dim * out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_dim_in = patch_dim_in
        self.patch_dim_out = patch_dim_out

        # Initialize weight and bias parameters
        self.inspected_layer = inspected_layer

        # Create a weight mask to enforce semi-dense connectivity
        self.weight_mask = torch.zeros_like(self.inspected_layer.weight)

        # Block connections from all neurons from latent dimensions
        if latent_dim > 0:
            self.weight_mask[:, -latent_dim:] = 0

        in_patch_idx = 0
        idxs_in_patches = []

        self.max_in_patch_idx_for_out_patch = {}

        self.idxs_of_in_patches = {}
        self.idxs_of_out_patches = {}
        # Iterate over output patches to set up connections
        # The number of output patches should be the same as the number of input patches
        for out_patch_idx in range(
            0,
            (self.out_dim * self.out_dim)
            // (self.patch_dim_out * self.patch_dim_out),
        ):

            # If shift is false, we want to append indexes of in patches in each iteration
            # If shift is true, we want to have the same indexes of in patches for the 0-th and 1-st out patch.
            # For the 2,...,n-th patches we want to append indexes of in patches.
            if not shift or out_patch_idx != 1:
                # get the row idx of patch
                row = in_patch_idx // (self.in_dim // self.patch_dim_in)
                # get the col idx of patch
                col = in_patch_idx % (self.in_dim // self.patch_dim_in)

                # get idxs of in neurons inside the patch (square: patch_dim_in x patch_dim_in)
                neurons_in_patch = [
                    (row * self.patch_dim_in + p_y) * self.in_dim
                    + col * self.patch_dim_in
                    + p_x
                    for p_x in range(self.patch_dim_in)
                    for p_y in range(self.patch_dim_in)
                ]

                # store idxs of neurons inside the previous patches and the current one
                idxs_in_patches += neurons_in_patch

                context_neurons_in_patch = [
                    n_i_p + ch * (self.in_dim * self.in_dim)
                    for n_i_p in neurons_in_patch
                    for ch in range(self.in_channels)
                ]
                self.idxs_of_in_patches[in_patch_idx] = (
                    context_neurons_in_patch
                )

            self.max_in_patch_idx_for_out_patch[out_patch_idx] = in_patch_idx

            # If shift is true, the 0-th out patch should look at 0-th in patch
            # and the i>0 i-th out patch should look at the 0,....,(i-1)-th in patches
            # If shift is false, the i-th out patch should look at 0,...1-th in patches
            if shift:
                if out_patch_idx != 0:
                    in_patch_idx += 1
            else:
                in_patch_idx += 1

            # get the row idx of patch
            row = out_patch_idx // (self.out_dim // self.patch_dim_out)
            # get the col idx of patch
            col = out_patch_idx % (self.out_dim // self.patch_dim_out)

            # get idxs of out neurons inside the patch (square: patch_dim_out x patch_dim_out)
            neurons_out_patch = [
                (row * self.patch_dim_out + p_y) * self.out_dim
                + col * self.patch_dim_out
                + p_x
                for p_x in range(self.patch_dim_out)
                for p_y in range(self.patch_dim_out)
            ]

            # get idxs of all out neurons across the out channels (each out channel has a block of out_dim * out_dim neurons)
            neurons_out_patch = [
                n_o_p + ch * (self.out_dim * self.out_dim)
                for n_o_p in neurons_out_patch
                for ch in range(self.out_channels)
            ]

            self.idxs_of_out_patches[out_patch_idx] = neurons_out_patch

    def to(self, *args, **kwargs):
        """Move the module and its parameters to a specified device."""
        super().to(*args, **kwargs)
        self.weight_mask = self.weight_mask.to(device=args[0])

    def check_importance(self, out_patch_idx, top_k, image):
        """
        Check the importance of input patches for a given output patch index.

        Parameters
        ----------
        out_patch_idx : int
            The index of the output patch to evaluate.
        top_k : int
            The number of top important patches to return.
        image : torch.Tensor
            The input image tensor.

        Returns
        -------
        list of tuple
            A list of tuples containing the importance score and input patch index, sorted by importance.
        """

        neurons_out_patch = self.idxs_of_out_patches[out_patch_idx]
        max_in_patch_idx = self.max_in_patch_idx_for_out_patch[out_patch_idx]
        image = image.view(1, -1)

        idxs = []
        for in_patch_idx in range(max_in_patch_idx + 1):
            context_neurons = self.idxs_of_in_patches[in_patch_idx]
            for n in neurons_out_patch:
                self.weight_mask[n, context_neurons] = 1

            activations = (
                image @ (self.inspected_layer.weight * self.weight_mask).T
            )
            idxs.append((sum(abs(activations.flatten())).item(), in_patch_idx))

            for n in neurons_out_patch:
                self.weight_mask[n, context_neurons] = 0

        idxs.sort(reverse=True)
        return idxs[:top_k]
