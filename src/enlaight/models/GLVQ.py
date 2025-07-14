"""Module of GLVQ algorithms."""

from typing import Any, Dict, Optional, Tuple, Union

import lightning as pl
import torch
from torch import optim
from torch.nn.modules import Module

from enlaight.core.distance import EuclideanDistance
from enlaight.core.loss import GLVQLoss


class GLVQ(pl.LightningModule):
    r"""GLVQ algorithm as Pytorch Lightning module.

    **GLVQ** ... Generalized Learning Vector Quantization

    See the following paper for information about the method:

        `"Generalized Learning Vector Quantization" by Sato and Yamada, 1995.
        <https://proceedings.neurips.cc/paper/1995/file/
        9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf>`_

    The algorithm is generic and accepts an arbitrary number of prototypes and
    distribution of the prototypes. The loss function is not fixed to the GLVQ loss
    function but must be a compatible loss, see
    :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`. For instance,
    it is also possible to use a :class:`enlaight.core.loss.MarginLoss`.

    Input data and prototypes must be processable by the provided ``distance``
    function (size and type checks should be performed by the distance function). The
    ``prototypes`` can be any (trainable or non-trainable) module. The
    ``prototype_labels`` define the class labels of the prototypes (one-hot coded).
    As a consequence, the number of labels must agree with the number of prototypes.
    The loss function checks the size and type of the input tensors. If provided,
    the ``loss_activation`` is applied after computing the loss function. For
    example, the loss values can be squashed by tanh. Before computing the distance
    function an arbitrary ``encoder`` can be applied to both the prototypes and the
    input data. With this, Siamese networks can be realized.

    Note that the loss function expects flat data tensors and prototypes.
    Additionally, the ``prototype_labels`` are used during the distance computation and,
    therefore, must have a data type that allows tensor multiplication.

    The loss function is minimized by the ADAM optimizer.

    See the **tutorial example notebook** for an introduction.

    :param prototypes: Module that returns the prototypes.
    :param prototype_labels: Tensor of prototype labels.
    :param distance: Distance function used to measure the dissimilarity.
    :param loss: The loss function. Must be a loss that follows the base class pattern
        :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.
    :param loss_activation: Activation function of the loss.
    :param encoder: Encoder used to encode the data before distance computation.
    :param learning_rate: Learning rate of the Adam optimizer.

    :Shape:

        - ``prototypes``: (*number_of_prototypes*, *number_of_features*).
        - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
    """

    def __init__(
        self,
        *,
        prototypes: Module,
        prototype_labels: torch.Tensor,
        distance: Module = EuclideanDistance(),
        loss: Module = GLVQLoss(),
        loss_activation: Optional[Module] = None,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-3,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()

        self.prototypes = prototypes
        self.distance = distance
        self.loss = loss
        self.loss_activation = loss_activation
        self.encoder = encoder
        self.learning_rate = learning_rate

        # register input tensor as buffer to ensure movement to correct device
        self.register_buffer("prototype_labels", prototype_labels)

        # turn-off logger to avoid saving of the modules to the .yml (data is stored
        # in checkpoint file)
        self.save_hyperparameters(logger=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the distance tensor.

        Compute the distance between each sample and prototype. Applies the encoder
        before the distance computation.

        Note that we forward the input vector to the ``prototypes`` module in case the
        module expects an input tensor.

        :param x: Input data tensor.
        :return: ``distances``: Tensor of distance values.

        :Shapes-In:
            - ``x``: (*number_of_samples*, \*)

        :Shapes-Out:
            - ``distances``: (*number_of_samples*, *number_of_prototypes*).
        """
        if self.encoder is None:
            data_encoded = x
            prototypes_encoded = self.prototypes(x)
        else:
            data_encoded = self.encoder(x)
            prototypes_encoded = self.encoder(self.prototypes(x))

        distances = self.distance(data_encoded, prototypes_encoded)

        return distances

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prediction step used for inference.

        Besides the predicted class labels, the function returns the loss values as
        they are a measure for stability of the classification in case of the GLVQ loss.
        Moreover, the indices of the closest prototypes are returned.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        :return:
            - ``predicted_class_labels``: The predicted labels depending on the winner
              prototypes - one-hot encoded.
            - ``loss_values``: The loss value per sample - **not** activated by
              ``loss_activation``.
            - ``closest_prototypes``: The winner prototypes.

        :Shapes-In:
            - ``batch``: (*number_of_samples*, \*).

        :Shapes-Out:
            - ``predicted_class_labels``: (*number_of_samples*, *number_of_classes*).
            - ``loss_values``: (*number_of_samples*,).
            - ``closest_prototypes``: (*number_of_samples*,).
        """
        # lazy solution as it is not run time critical (3-times torch.min computation)
        distances = self.forward(batch)

        _, closest_prototype = torch.min(distances, 1)
        predicted_labels = self.prototype_labels[closest_prototype]

        loss = self.loss(
            distances,
            self.prototype_labels,
            predicted_labels,
        )

        return predicted_labels, loss, closest_prototype

    def _shared_training_validation_test_step(
        self, name: str, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step.

        :param name: Name of the step. Used for logging.
        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return:
            - ``mean_of_loss_values``: The mean of the (activated) loss values (if
              ``self.loss_activation`` is not None).
            - ``accuracy``: The accuracy for the batch.

        :Shapes-In:
            - ``batch``: (`samples`, `classes`).

        :Shapes-Out:
            - ``mean_of_loss_values``: (,).
            - ``accuracy``: (,).
        """
        data, data_labels = batch

        distances = self.forward(data)

        loss = self.loss(
            distances,
            self.prototype_labels,
            data_labels,
        )

        if self.loss_activation is None:
            activated_loss = loss
        else:
            activated_loss = self.loss_activation(loss)

        mean_activated_loss = torch.mean(activated_loss)
        self.log(
            f"{name}_loss",
            mean_activated_loss,
            prog_bar=True,
            logger=True,
        )

        _, closest_prototype = torch.min(distances, 1)
        predicted_labels = self.prototype_labels[closest_prototype]

        accuracy = (
            torch.sum(
                torch.argmax(predicted_labels, dim=1)
                == torch.argmax(data_labels, dim=1)
            )
            / data_labels.shape[0]
        )
        self.log(
            f"{name}_acc",
            accuracy,
            prog_bar=True,
            logger=True,
        )

        return mean_activated_loss, accuracy

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        r"""Training step.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: ``mean_activated_loss``: The mean of the (activated) loss values (if
              ``self.loss_activation`` is not None).

        :Shapes-In:
            - ``batch``: (`samples`, `classes`).

        :Shapes-Out:
            - ``mean_activated_loss``: (,).
        """
        mean_activated_loss, _ = self._shared_training_validation_test_step(
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
        mean_activated_loss, _ = self._shared_training_validation_test_step(
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
        :return: ``results``: Dictionary containing ``{'loss': mean_value_of_loss,
            'accuracy': accuracy, 'batch_size': batch_size}``.

        :Shapes-In:
            - ``batch``: (`samples`, `classes`)

        :Shapes-Out:
            - ``results``: {\*}.
        """
        mean_activated_loss, accuracy = self._shared_training_validation_test_step(
            "test", batch, batch_idx
        )

        return {
            "loss": mean_activated_loss,
            "accuracy": accuracy,
            "batch_size": batch[0].shape[0],
        }

    def configure_optimizers(self) -> optim.Optimizer:
        r"""Configure ADAM optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_batch_end(
        self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, batch_idx: int
    ) -> None:
        r"""Call constraints of prototypes if available.

        :param outputs: The outputs of the training_step.
        :param batch: Batched training data.
        :param batch_idx: Index of the batch.
        """
        if hasattr(self.prototypes, "constraints"):
            getattr(self.prototypes, "constraints")(outputs, batch, batch_idx)
