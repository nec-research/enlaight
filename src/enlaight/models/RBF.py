"""Module of the RBF algorithm."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.parsing import is_picklable
from torch import optim
from torch.nn.modules import Module
from torch.nn.utils.parametrizations import _Orthogonal  # noqa: Used for type check
from torch.nn.utils.parametrize import is_parametrized

from enlaight.core.detection_probability import ExponentiallySquashedNegativeDistance
from enlaight.core.distance import EuclideanDistance
from enlaight.core.utils import TrainableSourcePair, orthogonal_parametrization


class RBF(pl.LightningModule):
    r"""RBF networks as Pytorch Lightning module.

    **RBF** ...  Radial Basis Function networks

    The method is rather generic but some computations assume class labels
    (classification); for instance, the accuracy computation. Overall, by defining an
    appropriate loss, the module also works for regression even if some logs might be
    incorrect/useless.

    The implementation supports layer normalization (weights sum unit-wise to one) and
    kernel normalization, where the output of the kernels is normalized to one.

    Note that any loss function is supported. If it is a loss function from the PyTorch
    library, it could be that the mean loss is returned. As a consequence, the loss
    activation could have no effect on the learning.

    See the **RBF example notebook** for an introduction.

    :param components: Components of the kernel computation.
    :param number_outputs: Number of output units (e.g., number of classes).
    :param number_components: Number of kernels (or equivalently the number of
        components).
    :param kernel_normalization: Whether to normalize the kernel outputs.
    :param layer_normalization: Whether to apply layer normalization.
    :param kernel: The kernel function.
    :param loss: The loss function.
    :param loss_activation: Activation function of the loss.
    :param encoder: Encoder used to encode the data before applying the RBF computation.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param eps: Small epsilon that is added to the normalization of the kernels.

    :Shape:

        - ``components``: (*number_of_components*, *number_of_features*).
    """

    def __init__(
        self,
        *,
        components: Module,
        number_outputs: int,
        number_components: int,
        kernel_normalization: bool = False,
        layer_normalization: bool = False,
        kernel: Module = ExponentiallySquashedNegativeDistance(
            distance=EuclideanDistance(squared=True), trainable_sigma=True, sigma=1.0
        ),
        loss: Module = nn.CrossEntropyLoss(),
        loss_activation: Optional[Module] = None,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-3,
        eps: float = 1e-9,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        # outputs are always unnormalized (no softmax)
        # number_of_components cannot be inferred  since components could be more
        #  complex and could only produce the real prototype structure after feeding an
        #  input. Hence, we require number of components.

        self.components = components
        self.kernel_normalization = kernel_normalization
        self.layer_normalization = layer_normalization
        self.kernel = kernel

        # We use this to have the option to define self.loss() as a method in
        # inherited classes by providing the argument 'loss=None' in the __init__.
        # We do not indicate via typehints that this is possible as it could break the
        # class definition. To have a more flexible handling of inputs we also allow
        # loss inputs that are callable, which is the minimum requirement.
        if isinstance(loss, Module) or callable(loss):
            self.loss = loss

        self.loss_activation = loss_activation
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.eps = eps
        self.number_outputs = number_outputs
        self.number_components = number_components

        self.weights = nn.Parameter(
            torch.empty(self.number_outputs, self.number_components)
        )
        nn.init.xavier_uniform_(self.weights)

        # turn-off logger to avoid saving of the modules to the .yml (data is stored
        # in checkpoint file)
        self.save_hyperparameters(logger=False)

        # saving of function:`enlaight.core.utils.AffineSubspaces` doesn't work and will
        # raise a warning; we handle the saving manually via callbacks and inform the
        # user about this by the following warning.
        if not is_picklable(self.components):
            if self._is_components_orthogonal_trainable_source_pair():
                warnings.warn(
                    "`self.components` is not pickle so that complete checkpoints "
                    "cannot be created automatically by PyTorch-Lightning. However, "
                    "it is recognized that `self.components` is of "
                    "class:`TrainableSourcePair` with an orthogonal parametrization. "
                    "This case can be handled during checkpoint creation so that the "
                    "checkpoint files will be correct.",
                    UserWarning,
                )

    def _is_components_orthogonal_trainable_source_pair(self) -> bool:
        r"""Return true if components is an orthogonalized TrainableSourcePair.

        Return ``True`` if the ``self.components`` is a module of
        :func:`enlaight.core.utils.AffineSubspaces`

        :return: ``True`` if above condition is valid. False otherwise.
        """
        if isinstance(self.components, TrainableSourcePair):
            if is_parametrized(self.components):
                if isinstance(self.components.parametrizations.weight[0], _Orthogonal):
                    return True

        return False

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""Store orthogonal parametrization manually if needed.

        If ``self.components`` is a module of
        :func:`enlaight.core.utils.AffineSubspaces`, the original
        :class:`enlaight.core.utils.TrainableSourcePair` is manually added to the
        checkpoint. Additionally, a flag is stored that indicates that
        ``self.components`` was a module of
        :func:`enlaight.core.utils.AffineSubspaces`.

        :param checkpoint: Checkpoint dictionary.
        """
        checkpoint["_is_components_orthogonal_trainable_source_pair"] = False

        if "components" not in checkpoint["hyper_parameters"].keys():
            if self._is_components_orthogonal_trainable_source_pair():
                if not hasattr(self.components, "constraint_values"):
                    checkpoint["hyper_parameters"]["components"] = TrainableSourcePair(
                        init_weight=self.components.weight,
                        init_bias=self.components.bias,
                    )

                checkpoint["_is_components_orthogonal_trainable_source_pair"] = True

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        r"""Restore orthogonal parametrization of affine subspaces if needed.

        If ``self.components`` was a module of
        :func:`enlaight.core.utils.AffineSubspaces`, the stored
        :class:`enlaight.core.utils.TrainableSourcePair` is manually transformed into
        a module of this function. Thus, the orthogonalization is restored.

        :param checkpoint: Checkpoint dictionary.
        """
        if checkpoint["_is_components_orthogonal_trainable_source_pair"]:
            orthogonal_parametrization(self.components)

    def _shared_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared forward step.

        :param x: Input data tensor suitable for the kernel function.
        :return:
            - ``prediction``: Output logit values.
            - ``detection_probabilities``: Kernel responses.

        :Shapes-In:
            - ``x``: (*batch*, \*)

        :Shapes-Out:
            - ``prediction``: (*batch*, *number_outputs*).
            - ``detection_probabilities``: (*batch*, *number_components*).
        """
        if self.encoder is None:
            data_encoded = x
            components_encoded = self.components(x)
        else:
            data_encoded = self.encoder(x)
            components_encoded = self.encoder(self.components(x))

        detection_probabilities = self.kernel(data_encoded, components_encoded)

        if self.kernel_normalization:
            detection_probabilities = detection_probabilities / (
                torch.sum(detection_probabilities, dim=1, keepdim=True) + self.eps
            )

        weights = self._decode_weights

        prediction = detection_probabilities @ weights.T

        return prediction, detection_probabilities

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the output logits.

        Applies the encoder before the RBF computation.
        Note that we forward the input vector to the ``components`` module in case the
        module expects an input tensor.

        :param x: Input data tensor.
        :return: ``prediction``: Tensor of logit values.

        :Shapes-In:
            - ``x``: (*batch*, \*)

        :Shapes-Out:
            - ``prediction``: (*batch*, *number_outputs*).
        """
        prediction, _ = self._shared_forward(x)

        return prediction

    @property
    def _decode_weights(self) -> torch.Tensor:
        r"""Decode weights (with normalization if applied).

        :return: ``weights``: Decoded weights.

        :Shapes-Out:
            - ``weights``: (*number_outputs*, *number_components*).
        """
        if self.layer_normalization:
            weights = torch.softmax(self.weights, dim=1)
        else:
            weights = self.weights

        return weights

    @property
    def decoded_weights(self) -> torch.Tensor:
        r"""Get decoded weights (with normalization if applied).

        The returned tensor is a detached clone.

        :return: ``weights``: Decoded weights.

        :Shapes-Out:
            - ``weights``: (*number_outputs*, *number_components*).
        """
        weights = self._decode_weights.detach().clone()

        return weights

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prediction step used for inference.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        :return: Tuple of (``predicted_labels``, ``loss``. ``output_logits``).

        :Shape-In:
            - ``batch``: (*batch*, \*).

        :Shape-Out:
            - ``predicted_labels``: (*batch*).
            - ``loss``: (*batch*) or float.
            - ``output_logits``: (*batch*, *number_classes*).
        """
        output_logits = self.forward(batch)
        predicted_labels = torch.argmax(output_logits, dim=1)

        loss = self.loss(output_logits, predicted_labels)

        return predicted_labels, loss, output_logits

    def _shared_training_validation_test_predict_step(
        self,
        name: str,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step.

        :param name: Name of the step. Used for logging.
        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Tuple of (``mean_of_loss_values``, ``accuracy``).
        """
        data, data_labels = batch

        output_logits = self.forward(data)
        predicted_labels = torch.argmax(output_logits, dim=1)

        loss = self.loss(output_logits, data_labels)

        if self.loss_activation is None:
            activated_loss = loss
        else:
            activated_loss = self.loss_activation(loss)

        mean_activated_loss = torch.mean(activated_loss)

        accuracy = (predicted_labels == data_labels).float().sum() / len(data_labels)

        self.log(
            f"{name}_loss",
            mean_activated_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{name}_acc",
            accuracy,
            prog_bar=True,
            logger=True,
        )

        return (
            mean_activated_loss,
            accuracy,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        r"""Training step.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Mean loss value.
        """
        mean_activated_loss, _ = self._shared_training_validation_test_predict_step(
            "train", batch, batch_idx
        )

        return mean_activated_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        r"""Implement validation step.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        """
        mean_activated_loss, _ = self._shared_training_validation_test_predict_step(
            "val", batch, batch_idx
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        r"""Test step.

        The return value is required to compute the precise final loss value and
        accuracy, which is inaccurate with the internal method as equally sized batch
        sizes are assumed.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Dictionary containing ``{'loss': mean_value_of_loss,
            'accuracy': accuracy, 'batch_size': batch_size}``.
        """
        (
            mean_activated_loss,
            accuracy,
        ) = self._shared_training_validation_test_predict_step("test", batch, batch_idx)

        return {
            "loss": mean_activated_loss,
            "accuracy": accuracy,
            "batch_size": batch[0].size()[0],
        }

    def configure_optimizers(self) -> optim.Optimizer:
        r"""Configure ADAM optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_batch_end(
        self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, batch_idx: int
    ) -> None:
        r"""Call constraints of components if available.

        :param outputs: The outputs of the training_step.
        :param batch: Batched training data.
        :param batch_idx: Index of the batch.
        """
        if hasattr(self.components, "constraints"):
            getattr(self.components, "constraints")(outputs, batch, batch_idx)


class RobustRBF(RBF):
    r"""Robust RBF networks as Pytorch Lightning module.

    **RBF** ...  Robust Radial Basis Function networks

    The module realize an RBF network where the loss provably optimizes the robustness.
    See the linked paper in :class:`enlaight.models.StableCBC`. This paper introduces
    the loss function where the provably RobustRBF is a special case.

    Since this class is an RBF with a different loss, the class comes without a
    dedicated tutorial. See the **RBF example notebook** for an introduction.

    :param components: Components of the kernel computation.
    :param number_outputs: Number of output units (e.g., number of classes).
    :param number_components: Number of kernels (or equivalently the number of
        components).
    :param distance: A module that measures the distances between the components and
        the inputs. Note that this must be a proper metric.
    :param sigma: The temperature of the kernel. Usually a positive number.
    :param trainable_sigma: Whether to have trainable ``sigma`` parameters.
    :param margin: The margin value if the loss should optimize for a minimal
        robustness value.
    :param negative_loss_weight: Scaling of the negative loss part.
    :param encoder: Encoder used to encode the data before applying the RBF computation.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param eps: Small epsilon that is added to the normalization of the kernels.

    :Shape:

        - ``components``: (*number_of_components*, *number_of_features*).
    """

    def __init__(
        self,
        *,
        components: Module,
        number_outputs: int,
        number_components: int,
        distance: Module = EuclideanDistance(squared=True),
        trainable_sigma: bool = True,
        sigma: Union[float, torch.Tensor] = 1.0,
        negative_loss_weight: float = 1.0,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-3,
        eps: float = 1e-9,
        margin: Optional[float] = None,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__(
            components=components,
            number_outputs=number_outputs,
            number_components=number_components,
            kernel_normalization=False,
            layer_normalization=True,
            kernel=ExponentiallySquashedNegativeDistance(
                distance=distance,
                trainable_sigma=trainable_sigma,
                sigma=sigma,
            ),
            loss=None,  # noqa: We use this to surpass the loss definition attribute
            loss_activation=None,
            encoder=encoder,
            learning_rate=learning_rate,
            eps=eps,
        )
        self.margin = margin
        self.negative_loss_weight = negative_loss_weight

        if not hasattr(self.kernel.distance, "squared"):
            distance_name = self.kernel.distance.__class__.__name__
            warnings.warn(
                f"Distance function {distance_name} has no attribute 'squared'. "
                f"Therefore, it cannot be inferred if the distance is or is not "
                f"squared. Consequently, the default 'non-squared' is assumed. "
                f"Make sure that this is appropriate."
            )

    def loss(
        self,
        output_logits: torch.Tensor,
        data_labels: torch.Tensor,
        max_distances: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the robust stable CBC loss function.

        :param output_logits: Tensor of logit values.
        :param data_labels: The labels of the data one-hot encoded.
        :param max_distances: The maximum distance in the prediction.
        :return: Tensor of loss values for each sample.

        :Shape-In:
            - ``output_logits``: (*number_of_samples*, *number_outputs*).
            - ``data_labels``: (*number_of_samples*).

        :Shape-Out:
            - ``loss_per_sample``: (*number_of_samples*,).
        """
        data_labels = torch.nn.functional.one_hot(
            data_labels.to(torch.int64),
            num_classes=self.number_outputs,
        ).float()

        # In the following, the minus is required to have a minimization problem:
        # best_correct -> highest_correct_probability
        best_correct = torch.sum(data_labels * output_logits, dim=1)

        # best_incorrect -> highest_incorrect_probability
        best_incorrect, _ = torch.max(output_logits - data_labels, dim=1)

        sigma = torch.min(self.kernel.sigma)

        # only works for one sample per class!
        loss = sigma * torch.log(torch.sqrt(best_correct / best_incorrect))

        distance_func = self.kernel.distance
        distance_name = distance_func.__class__.__name__

        # We check if the distance is squared. If the attribute doesn't exist, we
        # always assume the non-squared case.
        if not hasattr(distance_func, "squared"):
            squared_distance = False
        else:
            squared_distance = distance_func.squared

        if squared_distance:
            max_distances_ = torch.sqrt(max_distances)
            loss_tmp = torch.where(
                loss > 0,
                -max_distances_ / 3
                + torch.sqrt(max_distances / 9 + torch.abs(loss) / 3),
                loss * self.negative_loss_weight,
            )
            loss = loss_tmp

        # If we recognize that the distance is a tangent distance, we divide by two.
        if distance_name == "TangentDistance":
            loss = loss / 2

        if self.margin is not None:
            loss = torch.relu(self.margin - loss)
        else:
            # negate to have a minimization problem
            loss = -loss

        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prediction step used for inference.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        :return: Tuple of (``predicted_labels``, ``loss``. ``output_logits``).

        :Shape-In:
            - ``batch``: (*batch*, \*).

        :Shape-Out:
            - ``predicted_labels``: (*batch*).
            - ``loss``: (*batch*).
            - ``output_logits``: (*batch*, *number_classes*).
        """
        output_logits, detection_probabilities = self._shared_forward(batch)

        # reverse detection probabilities to distances and compute the max
        max_distances = torch.max(
            -self.kernel.sigma * torch.log(detection_probabilities),
            dim=1,
        )[0]

        predicted_labels = torch.argmax(output_logits, dim=1)

        loss = self.loss(output_logits, predicted_labels, max_distances)

        return predicted_labels, loss, output_logits

    def _shared_training_validation_test_predict_step(
        self,
        name: str,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step.

        :param name: Name of the step. Used for logging.
        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Tuple of (``mean_of_loss_values``, ``accuracy``).
        """
        data, data_labels = batch

        output_logits, detection_probabilities = self._shared_forward(data)

        predicted_labels = torch.argmax(output_logits, dim=1)

        # reverse detection probabilities to distances and compute the max
        # We only compute it if squared is True; otherwise we use a dummy value.
        distance_func = self.kernel.distance

        # We check if the distance is squared. If the attribute doesn't exist, we
        # always assume the non-squared case.
        if not hasattr(distance_func, "squared"):
            squared_distance = False
        else:
            squared_distance = distance_func.squared

        if squared_distance:
            max_distances = torch.max(
                -self.kernel.sigma * torch.log(detection_probabilities),
                dim=1,
            )[0]
        else:
            # dummy value
            max_distances = torch.tensor(1.0)

        loss = self.loss(output_logits, data_labels, max_distances)

        mean_activated_loss = torch.mean(loss)

        accuracy = (predicted_labels == data_labels).float().sum() / len(data_labels)
        self.log(
            f"{name}_loss",
            mean_activated_loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{name}_acc",
            accuracy,
            prog_bar=True,
            logger=True,
        )

        return mean_activated_loss, accuracy
