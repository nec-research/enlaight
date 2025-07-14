"""Module of GTLVQ algorithms."""

import warnings
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.utilities.parsing import is_picklable
from torch.nn.modules import Module
from torch.nn.utils.parametrizations import _Orthogonal  # noqa: Used for type check
from torch.nn.utils.parametrize import is_parametrized

from enlaight.core.distance import TangentDistance
from enlaight.core.loss import GLVQLoss
from enlaight.core.utils import TrainableSourcePair, orthogonal_parametrization
from enlaight.models.GLVQ import GLVQ


class GTLVQ(GLVQ):
    r"""GTLVQ algorithm as Pytorch Lightning module.

    **GTLVQ** ... Generalized Tangent Learning Vector Quantization

    See the following paper for information about the method:

    `"Fast Adversarial Robustness Certification of Nearest Prototype Classifiers for
    Arbitrary Seminorms" by Saralajew et al., 2020. <https://proceedings.neurips.cc/
    paper/2020/file/9da187a7a191431db943a9a5a6fec6f4-Paper.pdf>`_

    Generalization of GLVQ by assuming that prototypes are affine subspaces instead of
    vectors in the data space; see :class:`enlaight.models.GLVQ` for basic information
    about the GLVQ algorithm.

    The distance function is the so-called tangent distance that computes the smallest
    Euclidean distance from the given data point to the affine subspace; see
    :class:`enlaight.core.distance.TangentDistance` for more information about
    the tangent distance and affine subspaces.

    The parametrization of the affine subspaces is assumed to have orthogonal bases.
    This is required by the tangent distance computation. **Note that the
    orthogonality is not checked!** Because this would add a huge computational
    overhead. If this model is used without a proper parametrization of the affine
    subspaces, the results will be incorrect and the training is likely producing
    NaNs (so the training will crash). Please, consider this carefully if you do not
    use the provided affine subspaces module
    :func:`enlaight.core.utils.AffineSubspaces`.

    The method is called "tangent learning" because it can be shown that the tangents
    (the basis vectors of the subspace) align at the position of the translations so
    that they approximate class variations. Consequently, they make a local
    approximation of the data variations at the position of the translation and, thus,
    approximate the unknown class-specific data manifold.

    Trainable ``affine_subspaces`` (i.e., ``prototypes``) with an orthogonal
    parametrization or constraint can be initialized by the
    :func:`enlaight.core.utils.AffineSubspaces`. The ``affine_subspaces`` are assumed
    to be parametrized by a tuple of tensors (``translations``, ``tangents``).

    See the **GTLVQ example notebook** for an introduction.

    :param prototypes: Module that returns the ``affine_subspaces`` as a tuple of
        tensors (``translations``, ``tangents``).
    :param prototype_labels: Tensor of prototype labels.
    :param loss: The loss function. Must be a loss that follows the base class pattern
        :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.
    :param loss_activation: Activation function of the loss.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param squared: If ``True``, squared tangent distance. Otherwise, normal tangent
        distance.
    :param eps: Small epsilon that is added when non-squared tangent distance is used.

    :Shape:

        - ``translations``: (*number_of_prototypes*, *number_of_features*).
        - ``tangents``: (*number_of_prototypes*, *number_of_features*, *subspace_size*)
          with *number_of_features* greater than or equal to *subspace_size*.
        - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
    """

    _distance = TangentDistance

    def __init__(
        self,
        *,
        prototypes: Module,
        prototype_labels: torch.Tensor,
        loss: Module = GLVQLoss(),
        loss_activation: Optional[Module] = None,
        learning_rate: float = 1.0e-3,
        squared: bool = True,
        eps: float = 1.0e-8,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__(
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            distance=self._distance(squared=squared, eps=eps),
            loss=loss,
            loss_activation=loss_activation,
            learning_rate=learning_rate,
        )

        # saving of function:`enlaight.core.utils.AffineSubspaces` doesn't work and will
        # raise a warning; we handle the saving manually via callbacks and inform the
        # user about this by the following warning.
        if not is_picklable(self.prototypes):
            if self._is_prototypes_orthogonal_trainable_source_pair():
                warnings.warn(
                    "`self.prototypes` is not pickle so that complete checkpoints "
                    "cannot be created automatically by PyTorch-Lightning. However, "
                    "it is recognized that `self.prototypes` is of "
                    "class:`TrainableSourcePair` with an orthogonal parametrization. "
                    "This case can be handled during checkpoint creation so that the "
                    "checkpoint files will be correct.",
                    UserWarning,
                )

    def _is_prototypes_orthogonal_trainable_source_pair(self) -> bool:
        r"""Return true if prototypes is an orthogonalized TrainableSourcePair.

        Return ``True`` if the ``self.prototypes`` is a module of
        :func:`enlaight.core.utils.AffineSubspaces`

        :return: ``True`` if above condition is valid. False otherwise.
        """
        if isinstance(self.prototypes, TrainableSourcePair):
            if is_parametrized(self.prototypes):
                if isinstance(self.prototypes.parametrizations.weight[0], _Orthogonal):
                    return True

        return False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""Store orthogonal parametrization manually if needed.

        If ``self.prototypes`` is a module of
        :func:`enlaight.core.utils.AffineSubspaces`, the original
        :class:`enlaight.core.utils.TrainableSourcePair` is manually added to the
        checkpoint. Additionally, a flag is stored that indicates that
        ``self.prototypes`` was a module of
        :func:`enlaight.core.utils.AffineSubspaces`.

        :param checkpoint: Checkpoint dictionary.
        """
        checkpoint["_is_prototypes_orthogonal_trainable_source_pair"] = False

        if "prototypes" not in checkpoint["hyper_parameters"].keys():
            if self._is_prototypes_orthogonal_trainable_source_pair():
                if not hasattr(self.prototypes, "constraint_values"):
                    checkpoint["hyper_parameters"]["prototypes"] = TrainableSourcePair(
                        init_weight=self.prototypes.weight,
                        init_bias=self.prototypes.bias,
                    )

                checkpoint["_is_prototypes_orthogonal_trainable_source_pair"] = True

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""Restore orthogonal parametrization of affine subspaces if needed.

        If ``self.prototypes`` was a module of
        :func:`enlaight.core.utils.AffineSubspaces`, the stored
        :class:`enlaight.core.utils.TrainableSourcePair` is manually transformed into
        a module of this function. Thus, the orthogonalization is restored.

        :param checkpoint: Checkpoint dictionary.
        """
        if checkpoint["_is_prototypes_orthogonal_trainable_source_pair"]:
            orthogonal_parametrization(self.prototypes)
