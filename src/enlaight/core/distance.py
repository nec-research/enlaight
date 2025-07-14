"""Module with distance functions (similarities or dissimilarities)."""

# DEVELOPER NOTE: Each distance computation that uses the identity to the dot products
# by the Pythagorean theorem must stabilize the computation by a ReLU activation because
# the distance can become negative because of numerical instabilities when the distance
# is almost zero.

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: import as non-lowercase okay
from torch.nn.functional import cosine_similarity as _cosine_similarity
from torch.nn.functional import pairwise_distance as _lp_distance
from torch.nn.modules import Module
from torch.nn.modules.distance import CosineSimilarity as _CosineSimilarity
from torch.nn.modules.distance import PairwiseDistance as _LpDistance

__all__ = [
    "CosineSimilarity",
    "EuclideanDistance",
    "LpDistance",
    "TangentDistance",
    "cosine_similarity",
    "euclidean_distance",
    "lp_distance",
    "tangent_distance",
    "squared_euclidean_distance",
    "squared_tangent_distance",
    "_affine_subspaces_dimension_and_size_check",
]


def _tensor_dimension_and_size_check(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Check the sizes and dimensions of the tensors for distance functions.

    To compute the distance between vectors or set of vectors (provided as matrix), the
    following must hold: Input vectors can be 1 or 2-dimensional. If 2-dimensional,
    the first dimension is considered as batch dimension.

    This function checks that feature dimensions of the vectors have the same size, that
    they are of the :class:`torch.Tensor`, have suitable dimensions, and that no
    dimension is empty (size of 0).

    If the checks are successful, the tensors are reshaped to be always in the
    2-dimensional format. Thus, a vector of size (4,)  becomes a tensor (matrix) of
    size (1, 4).

    :param x: First input tensor.
    :param y: Second input tensor.
    :raises ValueError:
        - If ``x`` or ``y`` are not of :class:`torch.Tensor`.
        - If ``x`` or ``y`` has a dimension greater than 2.
        - If ``x`` or ``y`` has a dimension of size 0.
        - If ``x`` and ``y`` do not have the same feature dimension
          ``x.shape[-1] != y.shape[-1]``.
    :return:
        - ``x``: Tensor ``x`` in 2-dimensional format.
        - ``y``: Tensor ``y`` in 2-dimensional format.

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``y``: (*batch_2*, *features*) or (*features*,).

    :Shapes-Out:
        - ``x``: (*batch_1*, *features*)
        - ``y``: (*batch_2*, *features*), where *batch_1* or *batch_2* is 1 if ``x`` or
          ``y`` was 1-dimensional.
    """
    # If provided first dimension is the batch dimension
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise ValueError(
            f"The inputs must be of class Tensor. "
            f"Provided type(x)={type(x)} and type(y)={type(y)}."
        )

    if not 0 < x.dim() <= 2 or not 0 < y.dim() <= 2:
        raise ValueError(
            f"The dimensions of the tensors must be 1 or 2. "
            f"Provided x.dim()={x.dim()} and y.dim()={y.dim()}."
        )

    if not x.shape[-1] == y.shape[-1] or not x.numel() or not y.numel():
        raise ValueError(
            f"The size of the tensors must be equal at the feature dimension (i.e., "
            f"last dimension) and must be greater than 0 in each dimension. "
            f"Provided x.shape[-1]={x.shape[-1]} and y.shape[-1]={y.shape[-1]}."
        )

    if x.dim() == 1:
        x = x.unsqueeze(0)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    return x, y


def _affine_subspaces_dimension_and_size_check(
    translations: Tensor, tangents: Tensor
) -> Tuple[Tensor, Tensor]:
    r"""Check the dimension and size of affine subspaces.

    :param translations: Translations of the affine subspaces.
    :param tangents: Tangents of the affine subspaces.
    :raises ValueError:
        - If ``tangents`` or ``translations`` is not of
          :class:`torch.Tensor`.
        - If ``tangents`` or ``translations`` has sizes of size 0.
        - If ``tangents`` has not the dimension 2 or 3.
        - If ``translations`` has not the dimension 1 or 2.
        - If ``tangents`` has not the same feature size as
          translations.
        - If ``tangents`` has a batch size different from the
            translations.
        - If feature size is less than the subspace size.
    :return:
        - ``translations``: Tensor of dimension 2 containing the translations.
        - ``tangents``: Tensor of dimension 3 containing the tangents.

    :Shape-In:
        - ``translations``: (*features*,) or (*batch*, *features*).
        - ``tangents``: (*features*, *subspace_size*) or
          (*batch*, *features*, *subspace_size*).

    :Shape-Out:
        - ``translations``: (*batch_1*, *features*),
        - ``tangents``: (*batch_2*, *features*, *subspace_size*), where *batch_1* or
          *batch_2* is 1 if ``translations`` was 1-dimensional or ``tangents`` was
          2-dimensional.
    """
    if not isinstance(translations, Tensor) or not isinstance(tangents, Tensor):
        raise ValueError(
            f"The inputs must be of class Tensor. "
            f"Provided type(translations)={type(translations)} and "
            f"type(tangents)={type(tangents)}."
        )

    if not translations.numel() or not tangents.numel():
        raise ValueError(
            f"Each dimension must be greater than 0. "
            f"Provided translations.shape={translations.shape} and "
            f"tangents.shape={tangents.shape}."
        )

    if not 2 <= tangents.dim() <= 3:
        raise ValueError(
            f"The dimension of the tangent tensor must be 2 or 3. "
            f"Provided tangents.dim()={tangents.dim()}."
        )
    elif tangents.dim() == 2:
        tangents = tangents.unsqueeze(0)

    if not 1 <= translations.dim() <= 2:
        raise ValueError(
            f"The dimension of the translation tensor must be 1 or 2. "
            f"Provided translations.dim()={translations.dim()}."
        )
    elif translations.dim() == 1:
        translations = translations.unsqueeze(0)

    if not translations.shape[:2] == tangents.shape[:2]:
        raise ValueError(
            f"The size of the affine_subspaces elements must be equal at the number of "
            f"prototypes dimension (dimension 0) and the feature dimension "
            f"(dimension 1) and must be greater than 0 in each dimension. "
            f"Provided translations.shape={translations.shape} and "
            f"tangents.shape={tangents.shape}."
        )

    if not translations.shape[1] >= tangents.shape[2]:
        raise ValueError(
            f"The subspace dimension must be less than or equal to the feature "
            f"dimension."
            f"Provided translations.shape[1]={translations.shape[1]} and "
            f"tangents.shape[2]={tangents.shape[2]}."
        )

    return translations, tangents


class EuclideanDistance(Module):
    r"""Computes the Euclidean distance.

    Input vectors can be 1 or 2-dimensional. If 2-dimensional, the first dimension
    is considered as batch dimension. Dimensions of the returned tensor are not
    squeezed to ensure a consistent dimension of 2. Thus, if two vectors are
    provided, the returned tensor has the size (1, 1).

    If the normal (non-squared) Euclidean distance is used, a small epsilon is added to
    stabilize the gradient computation for distance values close to 0:

    .. math::
        d(x,y) = \sqrt{(x-y)^T (x-y) + eps}

    The implementation is memory and computational efficiently realized by the
    dot-product formulation. A comparison of a naive implementation and the
    optimized implementation can be found in the body of the main function
    :func:`.squared_euclidean_distance`.

    :param squared: If ``True``, squared Euclidean distance. Otherwise, normal
        Euclidean distance.
    :param eps: Small epsilon that is added when normal Euclidean distance is used.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(30, 128)
    >>> distance_func = distance.EuclideanDistance(squared=True, eps=1e-6)
    >>> output = distance_func(x, y)
    """

    def __init__(self, squared: bool = False, eps: float = 1.0e-7) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.squared = squared
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Return (squared) Euclidean distance.

        :param x: First tensor of input vectors.
        :param y: Second tensor of input vectors.
        :return: ``distances``: Tensor of distances where the element ``distances[i,j]``
            is the distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the
            j-th vector in ``y`` (i.e., ``y[j,:]``).

        :Shapes-In:
            - ``x``: (*batch_1*, *features*) or (*features*,).
            - ``y``: (*batch_2*, *features*) or (*features*,).

        :Shapes-Out:
            - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1
              if ``x`` or ``y`` was 1-dimensional.
        """
        if self.squared:
            distances = squared_euclidean_distance(x, y)
        else:
            distances = euclidean_distance(x, y, self.eps)

        return distances


class TangentDistance(Module):
    r"""Computes the tangent distance.

    The tangent distance is defined as the shortest Euclidean distance between a
    point and an affine subspace. A :math:`d`-dimensional affine subspace :math:`w` is
    defined as

    .. math::
        w = \{t + W\theta \mid \theta \in \mathbb{R}^d\},

    where :math:`t` is the translation and :math:`W` is the tangent basis or basis
    representation of the subspace. For the computation, it is assumed that :math:`W`
    is orthogonal (i.e., :math:`W^T W = I`). Otherwise, the interpretation as the
    shortest Euclidean distance is not valid. **Note that the orthogonality is not
    checked by the class!**

    Given a vector :math:`x` and affine subspace :math:`w`, the tangent distance is
    determined by

    .. math::
        d(x, w) = (x - t)^T (I - W W^T) (x - t).

    This class computes this equation with batch support so that, given a set of input
    vectors and a set of affine subspaces, the returned distance matrix yields the
    values of the i-th input vector to the j-th affine subspace.

    If the non-squared version is used, the integration of ``eps`` follows the approach
    applied in :class:`EuclideanDistance`.

    The implementation is memory and computational efficiently realized by projecting
    the data to low dimensional subspace before computing the Euclidean distance by
    the dot-product formulation. A comparison of a naive implementation and the
    optimized implementation can be found in the body of the main function
    :func:`.squared_tangent_distance`.

    :param squared: If ``True``, squared tangent distance. Otherwise, normal tangent
        distance.
    :param eps: Small epsilon that is added when non-squared tangent distance is used.

    :Example:

    >>> x = torch.randn(100, 64)
    >>> t = torch.randn(20, 64)
    >>> # orthogonal basis
    >>> W, _ = torch.linalg.qr(torch.randn(20, 64, 12))
    >>> affine_subspaces = (t, W)
    >>> distance_func = distance.TangentDistance(squared=False)
    >>> output = distance_func(x, affine_subspaces)
    """

    def __init__(self, squared: bool = False, eps: float = 1.0e-7) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.squared = squared
        self.eps = eps

    def forward(self, x: Tensor, affine_subspaces: Tuple[Tensor, Tensor]) -> Tensor:
        r"""Return (squared) tangent distance.

        :param x: Input tensor of vectors.
        :param affine_subspaces: Tuple of (``translations``, ``tangents``) to describe
            the affine subspaces.
        :return: ``distance``: Tensor of distances where the element ``distances[i,j]``
            is the distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the
            j-th affine subspace (i.e., ``translation[j,:]`` and ``tangents[j,:,:]``).

        :Shapes-In:
            - ``x``: (*batch_1*, *features*) or (*features*,).
            - ``translations``: Either (*batch_2*, *features*) or (*features*,).
            - ``tangents``: Either (*features*, *subspace_size*) or
              (*batch_2*, *features*, *subspace_size*).

        :Shapes-Out:
            - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1
              if ``translations`` was 1-dimensional or ``tangents`` was 2-dimensional.
        """
        if self.squared:
            distances = squared_tangent_distance(x=x, affine_subspaces=affine_subspaces)
        else:
            distances = tangent_distance(
                x=x, affine_subspaces=affine_subspaces, eps=self.eps
            )

        return distances


class CosineSimilarity(_CosineSimilarity):
    r"""Computes the cosine similarity.

    Uses the :mod:`torch` routine and modifies the input such that it follows the input
    output style of other distance functions.

    Given two vectors :math:`x` and :math:`y`, the cosine similarity is computed by:

    .. math::
        d(x,y) = \dfrac{x^T y}{\max(\Vert x \Vert _2 \Vert y \Vert _2, \epsilon)}.

    The function supports batches in ``x`` and ``y``.

    :param eps: Small value to avoid division by zero.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(12, 128)
    >>> distance_func = distance.CosineSimilarity(eps=1e-6)
    >>> output = distance_func(x, y)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        r"""Initialize an object of the class."""
        super().__init__(dim=2)
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Return the cosine similarity.

        :param x: First input tensor.
        :param y: Second input tensor.
        :return: ``similarities``: Tensor of similarities where the element
            ``output[i,j]`` is the similarity between the i-th vector in ``x`` (i.e.,
            ``x[i,:]``) and the j-th vector in ``y`` (i.e., ``y[j,:]``).

        :Shapes-In:
            - ``x``: (*batch_1*, *features*) or (*features*,).
            - ``y``: (*batch_2*, *features*) or (*features*,).

        :Shapes-Out:
            - ``similarities``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is
              1 if ``x`` or ``y`` was 1-dimensional.
        """
        x, y = _tensor_dimension_and_size_check(x, y)

        similarities = super().forward(x1=x.unsqueeze(1), x2=y.unsqueeze(0))

        return similarities


class LpDistance(_LpDistance):
    r"""Computes Lp-distance.

    **Imported from** :mod:`torch` **for completeness.**

    **Note that this function does not follow the general input-output layout of the
    other distance functions since no cross-batch computation is performed.**

    Distances are computed using ``p``-norm, with constant ``eps`` added to avoid
    division by zero if ``p`` is negative, i.e.:

    .. math::
        \mathrm{dist}\left(x, y\right) = \left\Vert x-y + eps \right\Vert_p,

    and the ``p``-norm is given by:

    .. math::
        \Vert x \Vert _p = \left( \sum_{i=1}^n  \vert x_i \vert ^ p \right) ^ {1/p}.

    :param p: The norm degree. Can be negative.
    :param eps: Small value to avoid division by zero.
    :param keepdim: Determines whether to keep the vector dimension.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(100, 128)
    >>> distance_func = distance.LpDistance(p=2)
    >>> output = distance_func(x, y)
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Return the Lp-distance.

        :param x: First input tensor.
        :param y: Second input tensor.
        :return: ``lp_distances``: Tensor of Lp-distances.

        :Shapes-In:
            - ``x``: (*batch*, *features*) or (*features*,).
            - ``y``: (*batch*, *features*) or (*features*,) but same shape as ``x``.

        :Shapes-Out:
            - ``lp_distances``: (*batch*) or () based on input dimension. If ``keepdim``
              is ``True``, then (*batch*, 1) or (1,) based on input dimension.
        """
        lp_distances = super().forward(x1=x, x2=y)
        return lp_distances


def cosine_similarity(x: Tensor, y: Tensor, eps: float = 1.0e-7) -> Tensor:
    r"""Compute cosine similarity.

    Uses the :mod:`torch` routine and modifies the input such that it follows the input
    output style of other distance functions.

    Functional implementation of the :class:`.CosineSimilarity`. See this class for
    further information.

    :param x: First input tensor.
    :param y: Second input tensor.
    :param eps: Small value to avoid division by zero.
    :return: ``similarities``: Tensor of similarities where the element ``output[i,j]``
        is the similarity between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the
        j-th vector in ``y`` (i.e., ``y[j,:]``).

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``y``: (*batch_2*, *features*) or (*features*,).

    :Shapes-Out:
        - ``similarities``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1 if
          ``x`` or ``y`` was 1-dimensional.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(12, 128)
    >>> output = distance.cosine_similarity(x, y)
    """
    x, y = _tensor_dimension_and_size_check(x, y)

    similarities = _cosine_similarity(
        x1=x.unsqueeze(1), x2=y.unsqueeze(0), dim=2, eps=eps
    )

    return similarities


def lp_distance(
    x: Tensor, y: Tensor, p: float = 2.0, eps: float = 1e-06, keepdim: bool = False
) -> Tensor:
    r"""Compute Lp-distance.

    **Imported from** :mod:`torch` **for completeness.**

    **Note that this function does not follow the general input-output layout of the
    other distance functions since no cross-batch computation is performed.**

    Functional implementation of the :class:`.LpDistance`. See this class for further
    information.

    :param x: First input tensor.
    :param y: Second input tensor.
    :param p: The norm degree. Can be negative.
    :param eps: Small value to avoid division by zero.
    :param keepdim: Determines whether to keep the vector dimension.
    :return: ``lp_distances``: Tensor of Lp-distances.

    :Shapes-In:
        - ``x``: (*batch*, *features*) or (*features*,).
        - ``y``: (*batch*, *features*) or (*features*,) but same shape as ``x``.

    :Shapes-Out:
        - ``lp_distances``: (*batch*) or (,) based on input dimension. If ``keepdim`` is
          ``True``, then (*batch*, 1) or (1,) based on input dimension.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(100, 128)
    >>> output = distance.lp_distance(x, y, p=1, keepdim=False)
    """
    lp_distances = _lp_distance(x1=x, x2=y, p=p, eps=eps, keepdim=keepdim)

    return lp_distances


def euclidean_distance(x: Tensor, y: Tensor, eps: float = 1.0e-7) -> Tensor:
    r"""Euclidean distance function.

    Functional implementation of the :class:`.EuclideanDistance`. See this class for
    further information.

    :param x: First tensor of input vectors.
    :param y: Second tensor of input vectors.
    :param eps: Epsilon added after summation to stabilize gradient computations for
        distances close to 0.
    :return: ``distances``: Tensor of distances where the element ``output[i,j]`` is the
        distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the j-th vector
        in ``y`` (i.e., ``y[j,:]``).

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``y``: (*batch_2*, *features*) or (*features*,).

    :Shapes-Out:
        - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1 if
          ``x`` or ``y`` was 1-dimensional.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(30, 128)
    >>> output = distance.euclidean_distance(x, y, eps=1e-6)
    """
    distances = torch.sqrt(squared_euclidean_distance(x, y) + eps)

    return distances


def squared_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    r"""Squared Euclidean distance function.

    Functional implementation of the **squared** :class:`.EuclideanDistance`. See this
    class for further information.

    :param x: First tensor of input vectors.
    :param y: Second tensor of input vectors.
    :return: ``distances``: Tensor of distances where the element ``output[i,j]`` is the
        distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the j-th vector
        in ``y`` (i.e., ``y[j,:]``).

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``y``: (*batch_2*, *features*) or (*features*,).

    :Shapes-Out:
        - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1 if
          ``x`` or ``y`` was 1-dimensional.

    :Example:

    >>> x = torch.randn(100, 128)
    >>> y = torch.randn(30, 128)
    >>> output = distance.squared_euclidean_distance(x, y)
    """
    x, y = _tensor_dimension_and_size_check(x, y)

    distances = torch.relu(
        torch.sum(x**2, dim=1, keepdim=True)
        - 2 * x @ y.T
        + torch.sum(y.T**2, dim=0, keepdim=True)
    )

    return distances


def _squared_tangent_distance(
    *, x: Tensor, affine_subspaces: Tuple[Tensor, Tensor]
) -> Tuple[Tensor, Tensor]:
    r"""Squared tangent distance function.

    Functional implementation of the **squared** :class:`.TangentDistance`. See this
    class for further information.

    The return is a Tuple of distances tensors where the element ``output[i,j]`` is the
    distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the j-th
    affine subspace (i.e., ``translation[j,:]`` and ``tangents[j,:,:]``). The first
    element are the squared tangent distances, the second the Euclidean distances,
    and the third the projected distances.

    :param x: Input vectors.
    :param affine_subspaces: Tuple of (``translations``, ``tangents``) to describe
        the affine subspaces.
    :return:
        - ``squared_tangent_distances``: Tensor containing the sqaured tangent
          distances.
        - ``naive_distances``: Tensor containing the Euclidean distances.
        - ``projected_distances``: Tensor containing the projected distances.

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``translations``: (*batch_2*, *features*) or (*features*,).
        - ``tangents``: (*features*, *subspace_size*) or
          (*batch_2*, *features*, *subspace_size*).

    :Shapes-Out:
        - ``squared_tangent_distances: (*batch_1*, *batch_2*).
        - ``naive_distances``: (*batch_1*, *batch_2*).
    """
    translations, tangents = affine_subspaces

    x, translations = _tensor_dimension_and_size_check(x, translations)

    translations, tangents = _affine_subspaces_dimension_and_size_check(
        translations, tangents
    )

    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)

    squared_tangent_distances = torch.sum(
        (
            (x.unsqueeze(1) - translations).unsqueeze(2)
            @ (eye - tangents @ tangents.transpose(1, 2))
        )
        ** 2,
        (2, 3),
    )

    naive_distances = squared_euclidean_distance(x, translations)

    return squared_tangent_distances, naive_distances


def squared_tangent_distance(
    *, x: Tensor, affine_subspaces: Tuple[Tensor, Tensor]
) -> Tensor:
    r"""Squared tangent distance function.

    Functional implementation of the **squared** :class:`.TangentDistance`. See this
    class for further information.

    :param x: Input vectors.
    :param affine_subspaces: Tuple of (``translations``, ``tangents``) to describe
        the affine subspaces.
    :return: ``distances``: Tensor of distances where the element ``output[i,j]`` is the
        distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the j-th affine
        subspace (i.e., ``translation[j,:]`` and ``tangents[j,:,:]``).

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``translations``: (*batch_2*, *features*) or (*features*,).
        - ``tangents``: (*features*, *subspace_size*) or
          (*batch_2*, *features*, *subspace_size*).

    :Shapes-Out:
        - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1 if
          ``translations`` was 1-dimensional or ``tangents`` was 2-dimensional.

    :Example:

    >>> x = torch.randn(100, 64)
    >>> t = torch.randn(20, 64)
    >>> W, _ = torch.linalg.qr(torch.randn(20, 64, 12))  # orthogonal basis
    >>> output = distance.squared_tangent_distance(x=x, affine_subspaces=(t, W))
    """
    distances = _squared_tangent_distance(x=x, affine_subspaces=affine_subspaces)[0]

    return distances


def tangent_distance(
    *, x: Tensor, affine_subspaces: Tuple[Tensor, Tensor], eps: float = 1.0e-7
) -> Tensor:
    r"""Tangent distance function.

    Functional implementation of the :class:`.TangentDistance`. See this
    class for further information.

    :param x: Input vectors.
    :param affine_subspaces: Tuple of (``translations``, ``tangents``) to describe
        the affine subspaces.
    :param eps: Epsilon added after summation to stabilize gradient computations for
        distances close to 0.
    :return: ``distances``: Tensor of distances where the element ``output[i,j]`` is the
        distance between the i-th vector in ``x`` (i.e., ``x[i,:]``) and the j-th affine
        subspace (i.e., ``translation[j,:]`` and ``tangents[j,:,:]``).

    :Shapes-In:
        - ``x``: (*batch_1*, *features*) or (*features*,).
        - ``translations``: (*batch_2*, *features*) or (*features*,).
        - ``tangents``: (*features*, *subspace_size*) or
          (*batch_2*, *features*, *subspace_size*).

    :Shapes-Out:
        - ``distances``: (*batch_1*, *batch_2*), where *batch_1* or *batch_2* is 1 if
          ``translations`` was 1-dimensional or ``tangents`` was 2-dimensional.

    :Example:

    >>> x = torch.randn(100, 64)
    >>> t = torch.randn(20, 64)
    >>> W, _ = torch.linalg.qr(torch.randn(20, 64, 12))  # orthogonal basis
    >>> output = distance.tangent_distance(x=x, affine_subspaces=(t, W))
    """
    distances = torch.sqrt(
        squared_tangent_distance(x=x, affine_subspaces=affine_subspaces) + eps
    )

    return distances
