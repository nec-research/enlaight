"""Module with utility functions."""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.nn.utils.parametrizations
from torch import Tensor
from torch.nn import Module, Parameter

from enlaight.core.distance import (  # noqa: private import okay
    _affine_subspaces_dimension_and_size_check,
)


class TrainableSource(Module):
    r"""Source with trainable parameters.

    This module holds a tensor of trainable parameters called ``weight``. This module
    can be used to create a set of trainable prototypes or components. If the forward
    method is called, the method returns its trainable parameters.

    Follows the PyTorch naming convention by using the name ``weight`` for the
    parameters.

    If the input is a tensor of zeros, the trainable parameters are initialized by
    zeros. Note that the module can be set non-trainable. This is useful if expert
    knowledge is used for initialization.

    Note that the optional ``weight_constraint`` is only applied on the ``init_weight``
    tensor if ``apply_constraints_on_init`` is ``True``.

    :param init_weight: Tensor that initializes the trainable parameters.
    :param weight_constraint: Callable constraint on the ``weight`` parameter. Usually
        called after a training step to realize projected stochastic gradient descent.
    :param apply_constraints_on_init: If ``True``, the specified constraints are applied
        on the ``init_weight`` tensor.

    :Example:

    >>> init_weight = torch.randn((12, 11))
    >>> weight_constraint = lambda weight: torch.clamp(weight, 0, 1)
    >>> source = utils.TrainableSource(
    ...     init_weight=init_weight,
    ...     weight_constraint=weight_constraint,
    ...     apply_constraints_on_init=True,
    ... )
    >>> weight = source(torch.randn(1))  # weight is clamped
    """

    def __init__(
        self,
        init_weight: Tensor,
        weight_constraint: Optional[Callable[[Tensor], Tensor]] = None,
        apply_constraints_on_init: bool = False,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.weight = Parameter(init_weight)
        self.weight_constraint = weight_constraint

        if apply_constraints_on_init:
            # call constraints with dummy values to ensure values are correct
            self.constraints(torch.tensor(0), torch.tensor(0), 0)

    def forward(self, x: Tensor) -> Tensor:
        r"""Return the trainable parameters.

        We accept the argument ``x`` for convenience as this is the usual signature
        of the forward step.

        :param x: Input tensor.
        :return: ``self.weight``: Trainable parameters initialized by ``init_weight``.

        :Shapes-In:
            - ``x``: (*).

        :Shapes-Out:
            - ``self.weight``: (*).
        """
        return self.weight

    def constraints(
        self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, batch_idx: int
    ) -> None:
        r"""Apply the constraint on the weight parameter.

        Follows the signature of PyTorch-Lightning ``on_train_batch_end`` in case a
        constraint needs these arguments. Constraints are only applied if the
        parameters are trainable (i.e., ``requires_grad == True``).

        :param outputs: The outputs of the training_step.
        :param batch: Batched training data.
        :param batch_idx: Index of the batch.
        """
        if self.weight_constraint is not None and self.weight.requires_grad:
            self.weight.data = self.weight_constraint(self.weight.data)


# Alias
Prototypes = TrainableSource
Components = TrainableSource


class TrainableSourcePair(Module):
    r"""Source with a pair of trainable parameters.

    This module holds a pair of tensors as trainable parameters called ``weight`` and
    ``bias``. This module can be used to create a set of trainable affine subspaces
    where ``weight`` is the basis representation and ``bias`` is the translation. If
    the forward method is called, the method returns its trainable parameters.

    See :class:`TrainableSource` for further information.

    :param init_weight: Tensor that initializes the trainable parameters ``weight``.
    :param init_bias: Tensor that initializes the trainable parameters ``bias``.
    :param weight_constraint: Callable constraint on the ``weight`` parameter. Usually
        called after a training step.
    :param bias_constraint: Callable constraint on the ``bias`` parameter. Usually
        called after a training step.
    :param apply_constraints_on_init: If ``True``, the specified constraints are applied
        on the initialization values ``init_weight`` and/or ``init_bias``.

    :Example:

    >>> init_weight = torch.rand((12, 11))
    >>> init_bias = torch.rand((12,))
    >>> source = utils.TrainableSourcePair(
    ...     init_weight=init_weight,
    ...     init_bias=init_bias,
    ...     apply_constraints_on_init=False,
    ... )
    >>> bias, weight = source(torch.randn(1))
    """

    def __init__(
        self,
        *,
        init_weight: Tensor,
        init_bias: Tensor,
        weight_constraint: Optional[Callable[[Tensor], Tensor]] = None,
        bias_constraint: Optional[Callable[[Tensor], Tensor]] = None,
        apply_constraints_on_init: bool = False,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.weight = Parameter(init_weight)
        self.weight_constraint = weight_constraint

        self.bias = Parameter(init_bias)
        self.bias_constraint = bias_constraint

        if apply_constraints_on_init:
            # call constraints with dummy values to ensure values are correct
            self.constraints(torch.tensor(0), torch.tensor(0), 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Return the trainable parameters.

        We accept the argument ``x`` for convenience as this is the usual signature
        of the forward step.

        :param x: Input tensor.
        :return:
            - ``self.bias``: Trainable bias.
            - ``self.weight``: Trainable parameters initialized by ``init_weight``.


        :Shapes-In:
            - ``x``: (*).

        :Shapes-Out:
            - ``self.bias``: (*).
            - ``self.weight``: (*).
        """
        return self.bias, self.weight

    def constraints(
        self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, batch_idx: int
    ) -> None:
        r"""Apply the constraints on the bias and weight parameters.

        Follows the signature of PyTorch-Lightning ``on_train_batch_end`` in case a
        constraint needs these arguments. Constraints are only applied if the
        parameters are trainable (i.e., ``requires_grad == True``).

        :param outputs: The outputs of the training_step.
        :param batch: Batched training data.
        :param batch_idx: Index of the batch.
        """
        if self.weight_constraint is not None and self.weight.requires_grad:
            self.weight.data = self.weight_constraint(self.weight.data)

        if self.bias_constraint is not None and self.bias.requires_grad:
            self.bias.data = self.bias_constraint(self.bias.data)


def AffineSubspaces(  # noqa: we use camel-case on purpose to imitate a class
    *,
    init_weight: Tensor,
    init_bias: Tensor,
    projected_sgd: bool = False,
    orthogonalize_init_weight: bool = True,
) -> Module:
    r"""Return a :class:`TrainableSourcePair` as affine subspaces.

    In order to fulfill the tangent distance assumption that the basis (the
    ``weight``) parameters are orthogonal, we need to apply an orthogonal
    parametrization or a respective constraint to the weights. This function takes
    the ``init_weight`` and ``init_bias`` tensor and returns an instance of
    :class:`TrainableSourcePair` with:

    - an orthogonal parametrization of the weight parameters by Householder matrices if
      ``projected_sgd`` is ``False``. Thus, this parametrizes the ``weight`` so that
      updates of stochastic gradient descent are performed in the space of orthogonal
      matrices.
    - an orthogonal constraint of the ``weight`` parameters by a QR decomposition if
      ``projected_sgd`` is ``True``. During training, the constraint is applied after
      each update so that projected stochastic gradient descent is performed.

    The orthogonal parametrization is useful if the updates should be performed in
    the space of orthogonal matrices and, therefore, with precise gradients. However,
    this comes at the cost of increased memory consumption and slower run time.
    Therefore, ``projected_sgd`` might be preferred with the drawback of not
    performing precise updates, which can lead to learning instabilities. Note that
    the basis representation completely changes between two update steps when using
    ``projected_sgd``. This means that the new basis representation is not
    necessarily the closest basis representation in terms of the applied update (see
    notes in :func:`orthogonal_constraint`).

    :param init_weight: Tensor that initializes the trainable parameters ``weight``.
    :param init_bias: Tensor that initializes the trainable parameters ``bias``.
    :param projected_sgd: If ``True``, uses a :class:`TrainableSourcePair` with an
        orthogonal constraint on the ``weight`` (tangents) based on QR decomposition.
        If ``False``, uses an orthogonal parametrization based on Householder matrices.
    :param orthogonalize_init_weight: If ``True``, the initial weight matrix is
        orthogonalized. This option is only active if ``projected_sgd`` is ``True``.
    :return: ``affine_subspaces``: Instance of :class:`TrainableSourcePair` with
        orthogonal parametrization.

    :Shapes-In:
        - ``init_weight``: (*batch*, *features*, *subspace_size*).
        - ``init_bias``: (*batch*, *features*).

    :Example:

    >>> init_weight = torch.rand((12, 11))
    >>> init_bias = torch.rand((12,))
    >>> source = utils.AffineSubspaces(init_weight=init_weight, init_bias=init_bias)
    >>> bias, weight = source(torch.randn(1))  # weight is orthonormal
    """
    init_bias, init_weight = _affine_subspaces_dimension_and_size_check(
        init_bias, init_weight
    )

    if projected_sgd:
        affine_subspaces = TrainableSourcePair(
            init_weight=init_weight,
            init_bias=init_bias,
            weight_constraint=orthogonal_constraint,
            apply_constraints_on_init=orthogonalize_init_weight,
        )

    else:
        affine_subspaces = orthogonal_parametrization(
            TrainableSourcePair(
                init_weight=init_weight,
                init_bias=init_bias,
            ),
        )

    return affine_subspaces


def orthogonal_parametrization(module: Module) -> Module:
    r"""Apply an orthogonal parametrization to the weight tensor of the module.

    This function is used to handle the parameter setting of
    :func:`torch.nn.utils.parametrizations.orthogonal` centrally.

    Operation is performed in-place.

    :param module: Module with ``weight`` parameter to be orthogonalized.
    :return: ``module``: Module with orthogonal parametrization.
    """
    module = torch.nn.utils.parametrizations.orthogonal(
        module, name="weight", orthogonal_map="householder", use_trivialization=True
    )

    return module


def orthogonal_constraint(weight: Tensor) -> Tensor:
    r"""Orthogonalize the weight tensor.

    Uses the QR method from PyTorch. Therefore, do not use this method if the basis
    representation is important, meaning if the orthogonalization should return the
    closest orthogonal weight tensor. In this case use an orthogonalization scheme
    based on the polar decomposition (which is slower than QR).

    :param weight: Weight tensor to be orthogonalized.
    :return: ``orthogonal_weight``: Orthogonalized ``weight`` tensor.

    :Shapes-In:
        - ``weight``: (*).

    :Shapes-In:
        - ``orthogonal_weight``: ``weight.shape``.
    """
    # Requires contiguous(): Otherwise the following warning is raised:
    #   UserWarning: grad and param do not obey the gradient layout contract.
    #   This is not an error, but may impair performance.
    orthogonal_weight = torch.linalg.qr(weight, mode="reduced")[0].contiguous()

    return orthogonal_weight
