"""Classification by components models."""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import lightning as pl
import torch
from lightning.pytorch.utilities.parsing import is_picklable
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.utils.parametrizations import _Orthogonal  # noqa: Used for type check
from torch.nn.utils.parametrize import is_parametrized

from enlaight.core.detection_probability import ExponentiallySquashedNegativeDistance
from enlaight.core.loss import MarginLoss, robust_stable_cbc_loss
from enlaight.core.reasoning import cbc_reasoning, stable_cbc_reasoning
from enlaight.core.utils import TrainableSourcePair, orthogonal_parametrization


class CBC(pl.LightningModule):
    r"""CBC algorithm as Pytorch Lightning module.

    **CBC** ...  Classification-by-Components

    See the following paper for information about the method:

        `"Classification-by-Components: Probabilistic Modeling of Reasoning over a Set
        of Components" by Saralajew et al., 2019.
        <https://proceedings.neurips.cc/paper_files/paper/2019/hash/
        dca5672ff3444c7e997aa9a2c4eb2094-Abstract.html>`_

    Please note that there is a difference in the wording between the paper and
    the implementation. In the paper, we call the requiredness probabilities reasoning
    probabilities. However, here, reasoning probabilities are the probabilities that
    are multiplied with the detection and model the classification output.

    The algorithm is generic and accepts an arbitrary number of components and
    definition of the reasoning labels. The loss function is not fixed but must be a
    compatible loss, see :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.

    Input data and components must be processable by the provided
    ``detection_probability`` function (size and type checks should be performed by
    the function). The ``components`` can be any (trainable or non-trainable) module.
    The ``reasoning_labels`` define the class labels of the reasoning concepts
    (one-hot coded). As a consequence, the number of labels must agree with the number
    of reasoning concepts. The loss function checks the size and type of the input
    tensors. If provided, the ``loss_activation`` is applied after computing the loss
    function. For example, the loss values can be squashed by tanh. Before computing
    the detection probability function an arbitrary ``encoder`` can be applied to both
    the components and the input data. With this, Siamese networks can be realized.

    The ``init_requiredness_probabilities`` is used to initialize the trainable
    tensor of requiredness probabilities: ``requiredness_probabilities``. The tensor
    must consist of values of the unit interval. However, these values cannot be
    directly interpreted as the requiredness probabilities as a breaking chopstick
    decoding is applied to derive the requiredness probabilities. Therefore,
    to get the requiredness probabilities consisting of positive, indefinite,
    and negative requiredness use the function
    :meth:`.decoded_requiredness_probabilities`. The provided
    ``init_requiredness_probabilities`` is clamped to the unit interval to ensure
    this constraint. The decoding works as follows: Assume that the tensor is denoted
    as :math:`q\in\left[0,1\right]^2`. Then :math:`q_0` is the positive requiredness
    probability. The negative requiredness probability is given by :math:`(1 - q_0) *
    q_1`. Consequently, the indefinite requiredness probability is determined by
    :math:`1 - q_0 - (1 - q_0) * q_1`. After each update step, the trainable
    requiredness probability vector is clamped to the unit interval. Consequently,
    we perform a projected stochastic gradient descent.

    The ``init_component_probabilities`` is used to initialize the trainable tensor
    of component probabilities (the prior): ``component_probabilities``. If the
    provided value is ``None``, it is assumed that the vector is a non-trainable,
    uniform probability tensor. In this case, the tensor will cancel out during the
    computation so that the vector can be ignored. If a tensor is provided, the tensor
    defines the logit values of the trainable probability vector. This means that the
    provided tensor will be squashed by a softmax function during runtime to produce
    the probability vector.

    Note that the loss function expects flat data tensors and components.
    Additionally, the ``reasoning_labels`` are used during the distance computation and,
    therefore, must have a data type that allows tensor multiplication.

    The loss function is minimized by the ADAM optimizer.

    See the **CBC example notebook** for an introduction.

    :param components: Module that returns the components.
    :param reasoning_labels: Tensor of reasoning labels that define the class mapping
        of the reasoning concepts (one-hot encoded).
    :param detection_probability: A module that measure the detection probability
        given the components and an input. The module must support batching with
        respect to both inputs and components.
    :param init_requiredness_probabilities: The initial tensor used to define the
        trainable parameters ``requiredness_probabilities``. The tensor is assumed to be
        encoded as a breaking chopstick encoding. Consequently, all values must be in
        the unit interval.
    :param init_component_probabilities: The initial tensor used to define the
        trainable parameters ``component_probabilities``, which is the component prior.
        If ``None``, it is assumed that each component is equally important.
    :param loss: The loss function. Must be a loss that follows the base class pattern
        :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.
    :param loss_activation: Activation function of the loss.
    :param encoder: Encoder used to encode the data before applying the CBC computation.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param eps: Small epsilon that is added to stabilize the reasoning.

    :Shape:

        - ``components``: (*number_of_components*, *number_of_features*).
        - ``reasoning_labels``: (*number_of_reasoning_concepts*, *number_of_classes*).
        - ``init_requiredness_probabilities``: (2, *number_of_components*,
          *number_of_reasoning_concepts*).
        - ``init_component_probabilities``: (*number_of_components*,).
    """

    def __init__(
        self,
        *,
        components: Module,
        reasoning_labels: torch.Tensor,
        detection_probability: Module,
        init_requiredness_probabilities: Tensor,
        init_component_probabilities: Optional[Tensor] = None,
        loss: Module = MarginLoss(margin=0.3, similarity=True),
        loss_activation: Optional[Module] = None,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-2,
        eps: float = 1.0e-7,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()

        self.components = components
        self.detection_probability = detection_probability
        self.loss_activation = loss_activation
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.eps = eps

        # We use this to have the option to define self.loss() as a method in
        # inherited classes by providing the argument 'loss=None' in the __init__.
        # We do not indicate via typehints that this is possible as it could break the
        # class definition. To have a more flexible handling of inputs we also allow
        # loss inputs that are callable, which is the minimum requirement.
        if isinstance(loss, Module) or callable(loss):
            self.loss = loss

        # register input tensor as buffer to ensure movement to correct device
        self.register_buffer("reasoning_labels", reasoning_labels)

        # we use this condition to bypass the Parameter definition for StableCBC and
        # RobustStableCBC.
        if init_requiredness_probabilities is not None:
            init_requiredness_probabilities = torch.clamp(
                init_requiredness_probabilities, min=0, max=1
            )
            self.requiredness_probabilities = Parameter(init_requiredness_probabilities)
        else:
            self.requiredness_probabilities = None

        if init_component_probabilities is not None:
            # we apply the log because during decoding we apply softmax
            self.component_probabilities = Parameter(
                torch.log(init_component_probabilities)
            )
        else:
            self.component_probabilities = None

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

    @property
    def _decode_requiredness_probabilities(self) -> Tensor:
        r"""Decode the requiredness probabilities.

        The output is a tensor with the positive and negative requiredness
        probabilities.

        :return: Tensor with the positive and negative requiredness probabilities.

        :Shapes-Out:
            - ``decoded_requiredness_probabilities``:
              (2, *number_of_components*, *number_of_reasoning_concepts*).
        """
        # Breaking chopstick implementation with clamp.
        negative_requiredness_probabilities = (
            1 - self.requiredness_probabilities[0]
        ) * self.requiredness_probabilities[1]

        decoded_requiredness_probabilities = torch.cat(
            [
                torch.unsqueeze(self.requiredness_probabilities[0], 0),
                torch.unsqueeze(negative_requiredness_probabilities, 0),
            ]
        )

        return decoded_requiredness_probabilities

    @property
    def _decode_component_probabilities(self) -> Optional[Tensor]:
        r"""Decode the component probabilities.

        The output is a tensor with the component probabilities if not ``None``,
        otherwise the function will return ``None``.

        :return: ``component_probabilities``: Tensor with the component probabilities if
            not ``None``, otherwise ``None``.

        :Shapes-Out:
            - ``component_probabilities``: (*number_of_components*,).
        """
        if self.component_probabilities is not None:
            component_probabilities = torch.softmax(self.component_probabilities, dim=0)
        else:
            component_probabilities = None

        return component_probabilities

    @property
    def decoded_requiredness_probabilities(self) -> Tensor:
        r"""Get decoded requiredness probabilities.

        The output is a tensor ``p`` with the decoded requiredness probabilities,
        where ``p[0]`` is the probability of positive requiredness, ``p[1]`` is the
        probability of indefinite requiredness, and ``p[2]`` is the probability of
        negative requiredness. Note that the sum over the first dimension of the
        tensor is 1. The returned tensor is a detached clone.

        :return: ``decoded_requiredness_probabilities``: Decoded requiredness
            probabilities.

        :Shapes-Out:`
            - ``decoded_requiredness_probabilities``:
              (3, *number_of_components*, *number_of_reasoning_concepts*).
        """
        p = self._decode_requiredness_probabilities.data
        probs = [
            torch.unsqueeze(p[0], 0),
            1 - torch.unsqueeze(p[0], 0) - torch.unsqueeze(p[1], 0),
            torch.unsqueeze(p[1], 0),
        ]

        decoded_requiredness_probabilities = torch.cat(probs).detach().clone()
        return decoded_requiredness_probabilities

    @property
    def decoded_component_probabilities(self) -> Tensor:
        r"""Get decoded component probabilities.

        The output is a tensor with the decoded component probabilities. In case that
        no component probabilities were specified, the method returns a probability
        tensor where each component is equally likely, which is equivalent to not
        specifying component probabilities.
        The returned tensor is a detached clone.

        :return: ``probabilities``: Decoded component probabilities.

        :Shape-Out:
            - ``probabilities``: (*number_of_components*,).
        """
        p = self._decode_component_probabilities
        if p is not None:
            probabilities = p.data.detach().clone()
        else:
            number_components = self.requiredness_probabilities.shape[1]
            probabilities = torch.ones(number_components) / number_components

        return probabilities

    def _reasoning(
        self,
        *,
        detection_probabilities: Tensor,
        full_report: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        r"""Wrap the reasoning to realize that the class can be used as a base class.

        :param detection_probabilities: Tensor of probabilities of detected components
            in the inputs.
        :param full_report: If ``False``, only the agreement probability is computed
            (the class probability). If ``True``, all internal probabilities of the
            reasoning process are returned as *detached* and *cloned* tensors. See
            "return" for the returned probabilities.
        :return:
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i]`` is the probability for agreement ``i`` if
              ``detection_probabilities.dim()==1``.
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i,j]`` is the probability for agreement ``j`` given
              the input ``i`` from the batch of ``detection_probabilities`` if
              ``detection_probabilities.dim()==2``.
            - ``report``: If ``full_report==True``, for each probability tensor
              returned in the dictionary, the specification for ``full_report==False``
              is correct. The dictionary holds the following probability tensors:
              *agreement probability* (key 'agreement'),
              *disagreement probability* (key 'disagreement'),
              *detection probability* (key 'detection'; returned for completeness),
              *positive agreement probability* (key 'positive agreement'),
              *negative agreement probability* (key 'negative agreement'),
              *positive disagreement probability* (key 'positive disagreement'),
              *negative disagreement probability* (key 'negative disagreement'),
              *positive reasoning probability*  (key 'positive reasoning'),
              *negative reasoning probability* (key 'negative reasoning'),
              *positive requiredness probability* (key 'positive requiredness'),
              *indefinite requiredness probability* (key 'indefinite requiredness'),
              *negative requiredness probability* (key 'negative requiredness'),
              *component prior probability* (key 'component prior').

        :Shapes-In:
            - ``detection_probabilities``: Either (*number_of_components*,) or
              (*batch*, *number_of_components*).

        :Shapes-Out:
            - ``probabilities``: (*number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==1``.
            - ``probabilities``: (*batch*, *number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==2``.
            - ``report``: {*} If ``full_report==True``, dictionary of tensors with the
              format specified for ``full_report==False``.
        """
        probabilities = cbc_reasoning(
            detection_probabilities=detection_probabilities,
            requiredness_probabilities=self._decode_requiredness_probabilities,
            component_probabilities=self._decode_component_probabilities,
            eps=self.eps,
            full_report=full_report,
        )

        return probabilities

    def _shared_forward(
        self, x: Tensor, full_report: bool
    ) -> Union[Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        r"""Shared forward step.

        :param x: Input data tensor suitable for the detection function.
        :param full_report: If ``False``, only the agreement probabilities and the
            detection probabilities are computed. If ``True``, all internal
            probabilities of the reasoning process are returned as *detached* and
            *cloned* tensors. See 'return' for the returned probabilities.
        :return:
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i]`` is the probability for agreement ``i`` if
              ``detection_probabilities.dim()==1``.
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i,j]`` is the probability for agreement ``j`` given
              the input ``i`` from the batch of ``detection_probabilities`` if
              ``detection_probabilities.dim()==2``.
            - ``report``: If ``full_report==True``, for each probability tensor
              returned in the dictionary, the specification for ``full_report==False``
              is correct. The dictionary holds the following probability tensors:
              *agreement probability* (key 'agreement'),
              *disagreement probability* (key 'disagreement'),
              *detection probability* (key 'detection'; returned for completeness),
              *positive agreement probability* (key 'positive agreement'),
              *negative agreement probability* (key 'negative agreement'),
              *positive disagreement probability* (key 'positive disagreement'),
              *negative disagreement probability* (key 'negative disagreement'),
              *positive reasoning probability*  (key 'positive reasoning'),
              *negative reasoning probability* (key 'negative reasoning'),
              *positive requiredness probability* (key 'positive requiredness'),
              *indefinite requiredness probability* (key 'indefinite requiredness'),
              *negative requiredness probability* (key 'negative requiredness'),
              *component prior probability* (key 'component prior').

        :Shapes-In:
            - ``x``: (*batch*, \*)

        :Shapes-Out:
            - ``probabilities``: (*number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==1``.
            - ``probabilities``: (*batch*, *number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==2``.
            - ``report``: {\*} If ``full_report==True``, dictionary of tensors with the
              format specified for ``full_report==False``.
        """
        if self.encoder is None:
            data_encoded = x
            components_encoded = self.components(x)
        else:
            data_encoded = self.encoder(x)
            components_encoded = self.encoder(self.components(x))

        detection_probabilities = self.detection_probability(
            data_encoded, components_encoded
        )

        probabilities = self._reasoning(
            detection_probabilities=detection_probabilities,
            full_report=full_report,
        )

        if full_report:
            return probabilities
        else:
            return probabilities, detection_probabilities

    def forward(self, x: Tensor) -> Tensor:
        r"""Compute the class probability (agreement probability) tensor.

        Applies the encoder before the CBC computation.
        Note that we forward the input vector to the ``components`` module in case the
        module expects an input tensor.

        :param x: Input data tensor.
        :return: ``class_probabilities``: Tensor of class probabilities.

        :Shapes-In:
            - ``x``: (*batch*, \*)

        :Shapes-Out:
            - ``class_probabilities``: (*batch*, *number_of_reasoning_concepts*).
        """
        class_probabilities, _ = self._shared_forward(x, full_report=False)

        return class_probabilities

    def all_reasoning_probabilities(self, x: Tensor) -> Dict[str, Tensor]:
        r"""Compute all internal reasoning probabilities.

        Applies the encoder before the CBC computation.
        Note that we forward the input tensor to the ``components`` module in case the
        module expects an input tensor. All probabilities of the reasoning process are
        returned as *detached* and *cloned* tensors.

        :param x: Input data tensor.
        :return: ``probabilities``:  If ``x.dim()==1`` each tensor in the dict will be
            one-dimensional, where ``output[i]`` is the probability for agreement ``i``.
            If ``x.dim()==2`` each tensor in the dict will be two-dimensional, where
            ``output[i,j]`` is the probability for agreement ``j`` given the input
            ``i`` from the batch.
            The dictionary holds the following probability tensors:
            *agreement probability* (key 'agreement'),
            *disagreement probability* (key 'disagreement'),
            *detection probability* (key 'detection'; returned for completeness),
            *positive agreement probability* (key 'positive agreement'),
            *negative agreement probability* (key 'negative agreement'),
            *positive disagreement probability* (key 'positive disagreement'),
            *negative disagreement probability* (key 'negative disagreement'),
            *positive reasoning probability*  (key 'positive reasoning'),
            *negative reasoning probability* (key 'negative reasoning'),
            *positive requiredness probability* (key 'positive requiredness'),
            *indefinite requiredness probability* (key 'indefinite requiredness'),
            *negative requiredness probability* (key 'negative requiredness'),
            *component prior probability* (key 'component prior').

        :Shapes-In:
            - ``x``: (*batch*, \*)

        :Shapes-Out:
            - ``probabilities``: {``str``, (*Different tensor shapes*)}
              Dictionary of tensors with the respective format (see the description of
              the tensors in different functions).
        """
        probabilities: Dict[str, Tensor] = self._shared_forward(x, full_report=True)  # type: ignore[assignment]  # noqa: Assignment is correct

        return probabilities

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int = 0, dataloader_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Prediction step used for inference.

        Besides the predicted class labels, the function returns the loss values as
        they are a measure for stability of the classification in case of the
        margin loss. Moreover, the indices of the highest probable reasoning concepts
        are returned (highest agreement probability).

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        :return: Tuple of (``predicted_class_labels``, ``loss_values``,
            ``highest_agreement``). The ``predicted_class_labels`` are one-hot
            coded. The ``loss_values`` are not activated by ``loss_activation``.

        :Shape-In:
            - ``batch``: (*batch*, \*).

        :Shape-Out:
            - ``predicted_class_labels``: (*batch*, *number_of_classes*).
            - ``loss_values``: (*batch*,).
            - ``highest_agreement``: (*batch*,).
        """
        class_probabilities = self.forward(batch)

        _, highest_agreement = torch.max(class_probabilities, 1)
        predicted_labels = self.reasoning_labels[highest_agreement]

        loss = self.loss(class_probabilities, self.reasoning_labels, predicted_labels)

        return predicted_labels, loss, highest_agreement

    def _shared_training_validation_test_step(
        self, name: str, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step.

        :param name: Name of the step. Used for logging.
        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Tuple of (``mean_of_loss_values``, ``accuracy``).
        """
        data, data_labels = batch

        class_probabilities = self.forward(data)

        loss = self.loss(class_probabilities, self.reasoning_labels, data_labels)

        (
            mean_activated_loss,
            accuracy,
        ) = self._shared_training_validation_test_step_post_processing(
            name=name,
            data_labels=data_labels,
            loss=loss,
            class_probabilities=class_probabilities,
        )

        return mean_activated_loss, accuracy

    def _shared_training_validation_test_step_post_processing(
        self,
        name: str,
        data_labels: torch.Tensor,
        loss: torch.Tensor,
        class_probabilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step post-processing.

        Averages the loss, computes the accuracy, and does the logging.

        :param name: Name of the step. Used for logging.
        :param data_labels: Tensor of data labels.
        :param loss: Tensor of loss values.
        :param class_probabilities: Tensor of class probabilities.
        :return: Tuple of (``mean_of_loss_values``, ``accuracy``).
        """
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

        _, highest_probable_reasoning = torch.max(class_probabilities, 1)
        predicted_labels = self.reasoning_labels[highest_probable_reasoning]

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
        :return: Mean loss value.
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
        :return: Dictionary containing ``{'loss': mean_value_of_loss,
            'accuracy': accuracy, 'batch_size': batch_size}``.
        """
        mean_activated_loss, accuracy = self._shared_training_validation_test_step(
            "test", batch, batch_idx
        )

        return {
            "loss": mean_activated_loss,
            "accuracy": accuracy,
            "batch_size": batch[0].shape[0],
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        r"""Configure ADAM optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def on_train_batch_end(
        self, outputs: Union[torch.Tensor, Dict[str, Any]], batch: Any, batch_idx: int
    ) -> None:
        r"""Call constraints of components if available and clamp requiredness.

        :param outputs: The outputs of the training_step.
        :param batch: Batched training data.
        :param batch_idx: Index of the batch.
        """
        self.requiredness_probabilities.data = torch.clamp(
            self.requiredness_probabilities.data, min=0, max=1
        )
        if hasattr(self.components, "constraints"):
            getattr(self.components, "constraints")(outputs, batch, batch_idx)


class StableCBC(CBC):
    r"""Stable CBC algorithm as Pytorch Lightning module.

    **StableCBC** ...  Stable Classification-by-Components

    See the following paper for information about the method:

        `"A Robust Prototype-Based Network with Interpretable RBF Classifier
        Foundations" by Saralajew et al., 2025.
        <https://arxiv.org/abs/2412.15499>`_

    The algorithm is an extension of :class:`enlaight.models.CBC`. The difference is
    with respect to the performed reasoning process, see
    :func:`enlaight.core.reasoning.stable_cbc_reasoning` for more information.

    The algorithm is generic and accepts an arbitrary number of components and
    definition of the reasoning labels. The loss function is not fixed but must be a
    compatible loss, see :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.

    Input data and components must be processable by the provided
    ``detection_probability`` function (size and type checks should be performed by
    the function). The ``components`` can be any (trainable or non-trainable) module.
    The ``reasoning_labels`` define the class labels of the reasoning concepts
    (one-hot coded). As a consequence, the number of labels must agree with the number
    of reasoning concepts. The loss function checks the size and type of the input
    tensors. If provided, the ``loss_activation`` is applied after computing the loss
    function. For example, the loss values can be squashed by tanh. Before computing
    the detection probability function an arbitrary ``encoder`` can be applied to both
    the components and the input data. With this, Siamese networks can be realized.

    If provided, the ``init_component_probabilities`` and
    ``init_requiredness_probabilities`` are only used to initialize the reasoning
    probabilities. Internally, these two init probabilities are not used for training.
    The training is performed over the joint probability and the component and
    requiredness probabilities can be derived by marginalization. The class provides
    the option to initialize the reasoning probabilities by the component and
    requiredness probabilities to have an easy way to induce expert knowledge.

    Note that the loss function expects flat data tensors and components.
    Additionally, the ``reasoning_labels`` are used during the distance computation and,
    therefore, must have a data type that allows tensor multiplication.

    The loss function is minimized by the ADAM optimizer.

    See the **StableCBC example notebook** for an introduction.

    :param components: Module that returns the components.
    :param reasoning_labels: Tensor of reasoning labels that define the class mapping
        of the reasoning concepts (one-hot encoded).
    :param detection_probability: A module that measure the detection probability
        given the components and an input. The module must support batching with
        respect to both inputs and components.
    :param init_requiredness_probabilities: The initial requiredness probabilities
        used to initialize the reasoning probabilities or None.
    :param init_component_probabilities: The initial component probabilities used
        to initialize the reasoning probabilities or None.
    :param init_reasoning_probabilities: The initial reasoning probabilities
        used to initialize the reasoning probabilities or None.
    :param loss: The loss function. Must be a loss that follows the base class pattern
        :class:`enlaight.core.loss.TemplateInputComparisonBasedLoss`.
    :param loss_activation: Activation function of the loss.
    :param encoder: Encoder used to encode the data before applying the CBC computation.
    :param learning_rate: Learning rate of the Adam optimizer.

    :Shape:

        - ``components``: (*number_of_components*, *number_of_features*).
        - ``reasoning_labels``: (*number_of_reasoning_concepts*, *number_of_classes*).
        - ``init_reasoning_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``init_component_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``init_reasoning_probabilities``:
          (2 \* *number_of_components*, *number_of_reasoning_concepts*).
    """

    def __init__(
        self,
        *,
        components: Module,
        reasoning_labels: torch.Tensor,
        detection_probability: Module,
        init_requiredness_probabilities: Optional[Tensor],
        init_component_probabilities: Optional[Tensor],
        init_reasoning_probabilities: Optional[Tensor] = None,
        loss: Module = MarginLoss(margin=0.3, similarity=True),
        loss_activation: Optional[Module] = None,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-2,
    ) -> None:
        r"""Initialize an object of the class."""
        if (
            init_requiredness_probabilities is not None
            and init_component_probabilities is not None
            and init_reasoning_probabilities is None
        ):
            requiredness_given = True
        elif (
            init_requiredness_probabilities is None
            and init_component_probabilities is None
            and init_reasoning_probabilities is not None
        ):
            requiredness_given = False
        else:
            raise ValueError(
                "Either init_requiredness_probabilities and "
                "init_component_probabilities must be ``None`` and "
                "init_reasoning_probabilities must not be ``None`` or "
                "init_requiredness_probabilities and init_component_probabilities must "
                "not be ``None`` and init_reasoning_probabilities must be ``None``."
            )

        super().__init__(
            components=components,
            reasoning_labels=reasoning_labels,
            detection_probability=detection_probability,
            init_requiredness_probabilities=None,  # noqa: None is handled internally
            init_component_probabilities=None,
            loss=loss,
            loss_activation=loss_activation,
            encoder=encoder,
            learning_rate=learning_rate,
            eps=0,
        )

        # we only store the reasoning probabilities (joint probs aka slim version).
        # This tensor should be column-wise equal 1.
        if requiredness_given:
            init_reasoning_probabilities = torch.cat(
                [
                    init_component_probabilities * init_requiredness_probabilities,
                    init_component_probabilities
                    * (1 - init_requiredness_probabilities),
                ]
            )

        if (
            not torch.allclose(
                init_reasoning_probabilities.sum(0),
                torch.ones(init_reasoning_probabilities.shape[1]),
            )
            or not 0 <= init_reasoning_probabilities.min()
            or not init_reasoning_probabilities.max() <= 1
        ):
            raise ValueError(
                "The `init_reasoning_probabilities` must be column-wise a probability "
                "vector (`init_reasoning_probabilities.sum(0) == 1` and "
                "`0 <= init_reasoning_probabilities <= 1`). As this is "
                "violated, please check your inputs; "
                "`init_component_probabilities.sum(0) == 1` and "
                "`0 <= init_requiredness_probabilities <= 1`."
            )

        # we apply the log here because during the decoding softmax is applied
        self.reasoning_probabilities = Parameter(
            torch.log(init_reasoning_probabilities)
        )

    def _reasoning(
        self,
        *,
        detection_probabilities: Tensor,
        full_report: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        r"""Wrap the reasoning to realize that the class can be used as a base class.

        :param detection_probabilities: Tensor of probabilities of detected components
            in the inputs.
        :param full_report: If ``False``, only the agreement probability is computed
            (the class probability). If ``True``, all internal probabilities of the
            reasoning process are returned as *detached* and *cloned* tensors. See
            "return" for the returned probabilities.
        :return:
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i]`` is the probability for agreement ``i`` if
              ``detection_probabilities.dim()==1``.
            - ``probabilities``: If ``full_report==False``, the agreement probabilities
              tensor, where ``output[i,j]`` is the probability for agreement ``j``
              given the input ``i`` from the batch of ``detection_probabilities`` if
              ``detection_probabilities.dim()==2``.
            - ``report``: If ``full_report==True``, for each probability tensor
              returned in the dictionary, the specification for ``full_report==False``
              is correct. The dictionary holds the following probability tensors:
              *agreement probability* (key 'agreement'),
              *disagreement probability* (key 'disagreement'),
              *detection probability* (key 'detection'; returned for completeness),
              *positive agreement probability* (key 'positive agreement'),
              *negative agreement probability* (key 'negative agreement'),
              *positive disagreement probability* (key 'positive disagreement'),
              *negative disagreement probability* (key 'negative disagreement'),
              *positive reasoning probability*  (key 'positive reasoning'),
              *negative reasoning probability* (key 'negative reasoning'),
              *positive requiredness probability* (key 'positive requiredness'),
              *negative requiredness probability* (key 'negative requiredness'),
              *component prior probability* (key 'component prior').

        :Shapes-In:
            - ``detection_probabilities``: Either (*number_of_components*,) or
              (*batch*, *number_of_components*).

        :Shapes-Out:
            - ``probabilities``: (*number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==1``.
            - ``probabilities``: (*batch*, *number_of_reasoning_concepts*,) If
              ``full_report==False`` and ``detection_probabilities.dim()==2``.
            - ``report``: {\*} If ``full_report==True``, dictionary of tensors with the
              format specified for ``full_report==False``.
        """
        probabilities = stable_cbc_reasoning(
            detection_probabilities=detection_probabilities,
            requiredness_probabilities=None,
            component_probabilities=None,
            reasoning_probabilities=self._decode_reasoning_probabilities,
            full_report=full_report,
        )

        return probabilities

    @property
    def _decode_requiredness_probabilities(self) -> Tensor:
        r"""Decode the requiredness probabilities.

        The output is a tensor with the positive requiredness probabilities.

        :return: Tensor with the positive requiredness probabilities.

        :Shape-Out:
            - ``requiredness_probabilities``:
              (*number_of_components*, *number_of_reasoning_concepts*).
        """
        reasoning_probabilities = self._decode_reasoning_probabilities.reshape(
            2, -1, self.reasoning_probabilities.shape[-1]
        )

        component_probabilities = torch.sum(reasoning_probabilities, dim=0)

        zero_probability_positions = component_probabilities == 0
        if torch.any(zero_probability_positions):
            warnings.warn(
                message=(
                    "Some of the computed component probabilities are zero. "
                    "Consequently, the corresponding requiredness probabilities "
                    "cannot be restored. To output a requiredness probability "
                    "anyways, a default value 0.5 is returned for those "
                    "probabilities. During probability computations according to "
                    "the probability model, this default value is consistent with "
                    "the overall model."
                ),
                category=UserWarning,
            )
        component_probabilities[zero_probability_positions] = 1

        requiredness_probabilities = (
            reasoning_probabilities[0] / component_probabilities
        )
        if torch.any(zero_probability_positions):
            requiredness_probabilities[zero_probability_positions] = 0.5
            component_probabilities[zero_probability_positions] = 0

        return requiredness_probabilities

    @property
    def _decode_component_probabilities(self) -> Tensor:
        r"""Decode the component probabilities.

        The output is a tensor with the component probabilities.

        :return: Tensor with the component probabilities.

        :Shape-Out:
            - ``component_probabilities``:
              (*number_of_components*, *number_of_reasoning_concepts*).
        """
        reasoning_probabilities = self._decode_reasoning_probabilities.reshape(
            2, -1, self.reasoning_probabilities.shape[-1]
        )

        component_probabilities = torch.sum(reasoning_probabilities, dim=0)

        return component_probabilities

    @property
    def _decode_reasoning_probabilities(self) -> Tensor:
        r"""Decode the reasoning probabilities.

        The output is a tensor with the reasoning probabilities. The first half
        corresponds to positive and the second half to negative reasoning.

        :return: Tensor with the reasoning_probabilities.

        :Shape-Out:
            - ``reasoning_probabilities``:
              (2 \* *number_of_components*, *number_of_reasoning_concepts*).
        """
        reasoning_probabilities = torch.softmax(self.reasoning_probabilities, dim=0)

        return reasoning_probabilities

    @property
    def decoded_requiredness_probabilities(self) -> Tensor:
        r"""Get requiredness probabilities.

        The output is a tensor ``p`` with the requiredness probabilities,
        where ``p[0]`` is the probability of positive requiredness and ``p[1]`` is the
        probability of negative requiredness. Note that the sum over the first
        dimension of the tensor is 1. The returned tensor is a detached clone.

        :return: Requiredness probabilities.

        :Shape-Out:
            - ``requiredness_probabilities``:
              (2, *number_of_components*, *number_of_reasoning_concepts*).
        """
        p = self._decode_requiredness_probabilities
        probs = [torch.unsqueeze(p, 0), 1 - torch.unsqueeze(p, 0)]

        requiredness_probabilities = torch.cat(probs).detach().clone()

        return requiredness_probabilities

    @property
    def decoded_component_probabilities(self) -> Tensor:
        r"""Get component probabilities.

        The output is a tensor with the component probabilities.
        The returned tensor is a detached clone.

        :return: Component probabilities.

        :Shape-Out:
            - ``component_probabilities``:
              (*number_of_components*, *number_of_reasoning_concepts*).
        """
        p = self._decode_component_probabilities
        component_probabilities = p.data.detach().clone()

        return component_probabilities

    @property
    def decoded_reasoning_probabilities(self) -> Tensor:
        r"""Get reasoning probabilities.

        The output is a tensor with the reasoning probabilities.
        The returned tensor is a detached clone.

        :return: Reasoning probabilities.

        :Shape-Out:
            - ``reasoning_probabilities``:
              (2 \* *number_of_components*, *number_of_reasoning_concepts*).
        """
        p = self._decode_reasoning_probabilities
        reasoning_probabilities = p.data.detach().clone()

        return reasoning_probabilities

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


class RobustStableCBC(StableCBC):
    r"""Stable CBC algorithm as Pytorch Lightning module with the robust loss.

    **RobustStableCBC** ...  Robust Stable Classification-by-Components

    The class is similar to :class:`enlaight.models.StableCBC` and enhances this class
    with the robust loss optimization criterion.

    The class tries to recognize automatically whether the given distance is squared
    and a tangent distance. If this is not possible, the functions will always return
    the basic robust loss (Theorem 1 in the paper).

    Please note that the distance function must be a proper metric. Otherwise, the
    theory of the robust loss doesn't hold.

    The loss function is negative for incorrect predictions. In case of a squared
    distance, the positive and negative loss part are differently computed. Thus, they
    can be imbalanced. To control for this issue, ``negative_loss_weight`` can be used
    to scale the negative (incorrect prediction) loss part.

    Since this class is a StableCBC with a different loss, the class comes without a
    dedicated tutorial. See the **StableCBC example notebook** for an introduction.

    :param components: Module that returns the components.
    :param reasoning_labels: Tensor of reasoning labels that define the class mapping
        of the reasoning concepts (one-hot encoded).
    :param distance: A module that measures the distances between the components and
        the inputs. Note that this must be a proper metric.
    :param init_requiredness_probabilities: The initial requiredness probabilities
        used to initialize the reasoning probabilities or None.
    :param init_component_probabilities: The initial component probabilities used
        to initialize the reasoning probabilities or None.
    :param init_reasoning_probabilities: The initial reasoning probabilities
        used to initialize the reasoning probabilities or None.
    :param encoder: Encoder used to encode the data before applying the CBC computation.
    :param learning_rate: Learning rate of the Adam optimizer.
    :param sigma: The temperature of the kernel. Usually a positive number.
    :param trainable_sigma: Whether to have trainable ``sigma`` parameters.
    :param margin: The margin value if the loss should optimize for a minimal
        robustness value.
    :param negative_loss_weight: Scaling of the negative loss part.
    :param eps: Epsilon value to stabilize the division.

    :Shape-In:

        - ``components``: (*number_of_components*, *number_of_features*).
        - ``reasoning_labels``: (*number_of_reasoning_concepts*, *number_of_classes*).
        - ``init_reasoning_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``init_component_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``init_reasoning_probabilities``:
          (2 \* *number_of_components*, *number_of_reasoning_concepts*).
        - ``sigma``: Any suitable shape.
    """

    def __init__(
        self,
        *,
        components: Module,
        reasoning_labels: torch.Tensor,
        distance: Module,
        init_requiredness_probabilities: Tensor,
        init_component_probabilities: Tensor,
        init_reasoning_probabilities: Optional[Tensor] = None,
        encoder: Optional[Module] = None,
        learning_rate: float = 1.0e-2,
        trainable_sigma: bool = False,
        sigma: Union[float, Tensor] = 1.0,
        margin: Optional[float] = None,
        negative_loss_weight: float = 1.0,
        eps: float = 1.0e-8,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__(
            components=components,
            reasoning_labels=reasoning_labels,
            detection_probability=ExponentiallySquashedNegativeDistance(
                distance=distance,
                sigma=sigma,
                trainable_sigma=trainable_sigma,
                eps=eps,
            ),
            init_requiredness_probabilities=init_requiredness_probabilities,
            init_component_probabilities=init_component_probabilities,
            init_reasoning_probabilities=init_reasoning_probabilities,
            loss=None,  # noqa: We use this to surpass the loss definition attribute
            loss_activation=None,
            encoder=encoder,
            learning_rate=learning_rate,
        )

        self.margin = margin
        self.eps = eps
        self.negative_loss_weight = negative_loss_weight

        if not hasattr(self.detection_probability.distance, "squared"):
            distance_name = self.detection_probability.distance.__class__.__name__
            warnings.warn(
                f"Distance function {distance_name} has no attribute 'squared'. "
                f"Therefore, it cannot be inferred if the distance is or is not "
                f"squared. Consequently, the default 'non-squared' is assumed. "
                f"Make sure that this is appropriate."
            )

    def loss(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
        max_distances: Tensor,
    ) -> Tensor:
        r"""Compute the robust stable CBC loss function.

        :param data_template_comparisons: Tensor of detection probabilities.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :param max_distances: The maximum distance in the prediction.
        :return: Tensor of loss values for each sample.

        :Shape-In:
            - ``data_template_comparisons``: (*number_of_samples*,
              *number_of_components*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).

        :Shape-Out:
            - ``loss_per_sample``: (*number_of_samples*,).
        """
        loss = -robust_stable_cbc_loss(
            data_template_comparisons=data_template_comparisons,
            template_labels=template_labels,
            data_labels=data_labels,
            requiredness_probabilities=None,
            component_probabilities=None,
            reasoning_probabilities=self._decode_reasoning_probabilities,
            sigma=self.detection_probability.sigma,
            margin=None,
            eps=self.eps,
        )

        distance_func = self.detection_probability.distance
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

        Besides the predicted class labels, the function returns the loss values as
        they are a measure for stability of the classification. Moreover, the indices
        of the highest probable reasoning concepts are returned.

        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :param dataloader_idx: Index of the dataloader.
        :return: Tuple of (``predicted_class_labels``, ``loss_values``,
            ``highest_probable_reasoning``). The ``predicted_class_labels`` are one-hot
            coded. The ``loss_values`` are not activated by ``loss_activation``.

        :Shape:

            - ``batch``: (*batch*, \*).
            - ``predicted_class_labels``: (*batch*, *number_of_classes*).
            - ``loss_values``: (*batch*,).
            - ``closest_prototypes``: (*batch*,).
        """
        class_probabilities, detection_probabilities = self._shared_forward(
            batch, full_report=False
        )

        # reverse detection probabilities to distances and compute the max
        max_distances = torch.max(
            -self.detection_probability.sigma * torch.log(detection_probabilities),
            dim=1,
        )[0]

        _, highest_probable_reasoning = torch.max(class_probabilities, 1)
        predicted_labels = self.reasoning_labels[highest_probable_reasoning]

        loss = self.loss(
            data_template_comparisons=detection_probabilities,
            template_labels=self.reasoning_labels,
            data_labels=predicted_labels,
            max_distances=max_distances,
        )

        return predicted_labels, loss, highest_probable_reasoning

    def _shared_training_validation_test_step(
        self, name: str, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Shared training/validation/test step.

        :param name: Name of the step. Used for logging.
        :param batch: Tensor of data batch.
        :param batch_idx: Index of the batch.
        :return: Tuple of (``mean_of_loss_values``, ``accuracy``).
        """
        data, data_labels = batch

        class_probabilities, detection_probabilities = self._shared_forward(
            data, full_report=False
        )

        # reverse detection probabilities to distances and compute the max
        # We only compute it if squared is True; otherwise we use a dummy value.
        distance_func = self.detection_probability.distance

        # We check if the distance is squared. If the attribute doesn't exist, we
        # always assume the non-squared case.
        if not hasattr(distance_func, "squared"):
            squared_distance = False
        else:
            squared_distance = distance_func.squared

        if squared_distance:
            max_distances = torch.max(
                -self.detection_probability.sigma * torch.log(detection_probabilities),
                dim=1,
            )[0]
        else:
            # dummy value
            max_distances = torch.tensor(1.0)

        loss = self.loss(
            detection_probabilities, self.reasoning_labels, data_labels, max_distances
        )

        (
            mean_activated_loss,
            accuracy,
        ) = self._shared_training_validation_test_step_post_processing(
            name=name,
            data_labels=data_labels,
            loss=loss,
            class_probabilities=class_probabilities,
        )

        return mean_activated_loss, accuracy
