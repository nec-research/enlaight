"""Module with detection probability functions."""

# Right now, I don't think that we need a functional support for these classes - Sascha.

from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class SquashingBasedDetectionProbability(Module, ABC):
    r"""Base class for squashing-based detection probability functions.

    This abstract class should be used to implement squashing-based detection
    probability functions. Define the forward pass in the child class.
    """

    @abstractmethod
    def forward(self, inputs: Tensor, components: Tensor) -> Tensor:
        r"""Forward pass.

        :param inputs: Input tensor of the data.
        :param components: Input tensor of the components.
        :return: Squashed values.
        """


class ReLUSquashedCosineSimilarity(SquashingBasedDetectionProbability):
    r"""Activate an arbitrary cosine similarity with ReLU.

    By the ReLU activation, a proper similarity measure (i.e., outputs are in
    :math:`\left[0,1\right]`) is artificially produced. The function is the following:

    .. math::
        \max \{ \mathrm{cosine\_similarity}(x_1, x_2), 0 \}.

    :param cosine_similarity: The cosine similarity to be used.

    :Example:

    >>> inputs = torch.rand(64, 4)
    >>> components = torch.rand(12, 4)
    >>> func = detection_probability.ReLUSquashedCosineSimilarity(
    ...     cosine_similarity=distance.CosineSimilarity(eps=0,)
    ... )
    >>> output = func(inputs, components)
    """

    def __init__(self, cosine_similarity: Module) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.cosine_similarity = cosine_similarity

    def forward(self, inputs: Tensor, components: Tensor) -> Tensor:
        r"""Forward pass.

        Note that the input and output shapes are defined by the
        ``self.cosine_similarity`` method since this class-method applies an
        element-wise operation.

        :param inputs: Input tensor of the data.
        :param components: Input tensor of the components.
        :return: ``detection_probability``: Squashed cosine similarity.

        :Shapes-In:
            - ``inputs``: (*number_of_samples*, *number_of_features*).
            - ``components``: (*number_of_components*, *number_of_features*).

        :Shapes-Out:
            - ``detection_probability``: (*number_of_samples*, *number_of_components*)
              .
        """
        detection_probability = torch.relu(self.cosine_similarity(inputs, components))
        return detection_probability


class ExponentiallySquashedNegativeDistance(SquashingBasedDetectionProbability):
    r"""Activate an arbitrary distance with and rbf-like kernel.

    By this activation, the distance is transformed into a proper similarity measure
    (i.e., outputs are in :math:`\left[0,1\right]`). The transformation function is the
    following:

    .. math::
        \mathrm{exp} \left( -\frac{\mathrm{distance}(x_1, x_2)}{\sigma} \right).

    If ``sigma`` is not chosen correctly, the network can be hard to train because of
    vanishing gradients. The idea is to compute by how much the distances between the
    points vary. It is recommended to set the ``sigma`` by the following approach:
    Assume that the mean distance is denoted by ``mean`` and standard deviation distance
    by ``std`` for the distance function given the dataspace. Moreover, the minimum
    similarity defined formally as lower bound for the expected decision similarity
    should be :math:`p_0`. Then, :math:`\sigma` should be set as

    .. math::
        \sigma = -\frac{mean + std}{\mathrm{ln}(p_0)}.

    For example, in case of MNIST, we can normalize the data space to
    :math:`\left[0,1\right]^{784}`. Then, initial :math:`\sigma \approx 7.61` value
    for the non-squared Euclidean distance determined by the above equation when
    we assume that :math:`p_0=0.01`.

    The parameter ``sigma`` supports broadcasting and can be a trainable tensor so
    that it gets updated during training. For broadcasting, make sure that the size
    is suitable and that the broadcasting works as expected. The output shape of the
    distance function is usually (*batches*, *number_of_components*). Consequently, to
    have trainable sigma values for each component, use an initial ``sigma`` tensor of
    the shape (1, *number_of_components*).

    If the sigma is trainable, the ``sigma`` is activated by an absolute value
    function to avoid negative values and the division is further stabalized by a
    small epsilon. If the sigma is not trainable, the equation from above is applied
    without any activation. Consequently, negative sigma values will not be converted
    to positive values.

    :param distance: The distance to be used (e.g., squared Euclidean distance).
    :param sigma: The temperature of the kernel. Usually a positive number.
    :param trainable_sigma: Whether to have trainable ``sigma`` parameters.
    :param eps: A small value to avoid division by zero.

    :Example:

    >>> inputs = torch.rand(64, 4)
    >>> components = torch.rand(12, 4)
    >>> func = detection_probability.ExponentiallySquashedNegativeDistance(
    ...     distance=distance.EuclideanDistance(squared=False),
    ...     sigma=6.08,
    ...     trainable_sigma=True,
    ... )
    >>> output = func(inputs, components)
    """

    def __init__(
        self,
        distance: Module,
        sigma: Union[float, Tensor] = 1,
        trainable_sigma: bool = False,
        eps: float = 1.0e-7,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.distance = distance
        self.eps = eps
        self.trainable_sigma = trainable_sigma

        if not isinstance(sigma, Tensor):
            sigma = torch.tensor(sigma).float()

        if self.trainable_sigma:
            self._sigma = Parameter(sigma)
        else:
            # register tensor as buffer to ensure movement to correct device
            self.register_buffer("_sigma", sigma)

    def forward(self, inputs: Tensor, components: Tensor) -> Tensor:
        r"""Forward pass.

        Note that the input and output shapes are defined by the
        ``self.distance`` method since this class-method applies an element-wise
        operation.

        :param inputs: Input tensor of the data.
        :param components: Input tensor of the components.
        :return: ``detection_probability``: Squashed distance.

        :Shapes-In:
            - ``inputs``: (*number_of_samples*, *number_of_features*).
            - ``components``: (*number_of_components*, *number_of_features*).

        :Shapes-Out:
            - ``detection_probability``: (*number_of_samples*, *number_of_components*).
        """
        detection_probability = torch.exp(
            -self.distance(inputs, components) / self.sigma
        )
        return detection_probability

    @property
    def sigma(self) -> Tensor:
        r"""Get the sigma.

        Applies the stabilization by the absolute value and the eps if trainable.
        """
        if self.trainable_sigma:
            sigma = torch.abs(self._sigma) + self.eps
        else:
            sigma = self._sigma
        return sigma
