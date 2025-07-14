"""Module with loss functions."""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules import Module

warnings.simplefilter("once", UserWarning)


def _tensor_dimension_and_size_check(
    data_template_comparisons: Tensor,
    template_labels: Tensor,
    data_labels: Tensor,
) -> None:
    r"""Check the sizes and dimensions of the tensors for loss functions.

    The loss is computed based on data-template-comparisons (e.g., prototype to input
    distances or probabilities based on the CBC reasoning), labels of the templates
    (e.g, prototypes), and the data labels. This function checks that the tensors have
    the correct sizes.

    Note that the function does not check the one-hot coding of the labels.

    :param data_template_comparisons: Tensor of distances, probabilities, etc.
    :param template_labels: The labels of the templates (e.g., reasoning concepts or
        prototypes) one-hot coded.
    :param data_labels: The labels of the data one-hot coded.

    :raises ValueError: If tensors are not of class:`Tensor`.
    :raises ValueError: If tensors are not of dimension 2.
    :raises ValueError: If the tensor sizes do not match as specified.

    :Shapes-In:
        - ``data_template_comparisons``: (*number_of_samples*, *number_of_prototypes*).
        - ``template_labels``: (*number_of_prototypes*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).
    """
    if (
        not isinstance(data_template_comparisons, Tensor)
        or not isinstance(template_labels, Tensor)
        or not isinstance(data_labels, Tensor)
    ):
        raise ValueError(
            f"The inputs must be of class Tensor. "
            f"Provided type(data_template_comparisons)="
            f"{type(data_template_comparisons)}, "
            f"type(template_labels)={type(template_labels)}, and "
            f"type(data_labels)={type(data_labels)}."
        )

    if (
        data_template_comparisons.dim() != 2
        or template_labels.dim() != 2
        or data_labels.dim() != 2
    ):
        raise ValueError(
            f"The dimensions of the tensors must be 2. "
            f"Provided data_template_comparisons.dim()="
            f"{data_template_comparisons.dim()}, "
            f"template_labels.dim()={template_labels.dim()} (one-hot coding), "
            f"and data_labels.dim()={data_labels.dim()} (one-hot coding)."
        )

    if data_template_comparisons.shape[1] != template_labels.shape[0]:
        raise ValueError(
            f"The number of templates does not match: "
            f"data_template_comparisons.shape[1]="
            f"{data_template_comparisons.shape[1]} != "
            f"template_labels.shape[0]={template_labels.shape[0]}."
        )

    if data_template_comparisons.shape[0] != data_labels.shape[0]:
        raise ValueError(
            f"The number of samples does not match: "
            f"data_template_comparisons.shape[0]="
            f"{data_template_comparisons.shape[0]} != "
            f"data_labels.shape[0]={data_labels.shape[0]}."
        )

    if data_labels.shape[1] != template_labels.shape[1]:
        raise ValueError(
            f"The number of classes does not match (one-hot coding): "
            f"data_labels.shape[1]={data_labels.shape[1]} != "
            f"template_labels.shape[1]={template_labels.shape[1]}."
        )


def _closest_correct_and_incorrect_distance(
    *,
    distances: Tensor,
    prototype_labels: Tensor,
    data_labels: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Determine the closest correct and incorrect prototype.

    :param distances: Tensor of distances.
    :param prototype_labels: The labels of the prototypes one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :return:
        - ``distance_closest_correct``: Distances of closest correct prototypes.
        - ``idx_closest_correct``: Indices of closest correct prototypes.
        - ``distance_closest_incorrect``: Distances of closest incorrect prototypes.
        - ``idx_closest_incorrect``: Indices of closest incorrect prototypes.

    :Shapes-In:
        - ``distances``: (*number_of_samples*, *number_of_prototypes*).
        - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).

    :Shapes-Out:
        - ``distance_closest_correct``: (*number_of_samples*,).
        - ``idx_closest_correct``: (*number_of_samples*,).
        - ``distance_closest_incorrect``: (*number_of_samples*,).
        - ``idx_closest_incorrect``: (*number_of_samples*,).

    """
    # compute a value that is larger than any distance
    max_distances = torch.max(distances) + 1

    # compute a matrix where one means that this data sample has the same label as the
    # prototype (number_samples, number_prototypes)
    labels_agree = data_labels @ prototype_labels.T

    # increase distance by max_distances for all prototypes that are not correct
    distance_closest_correct, idx_closest_correct = torch.min(
        distances + (1 - labels_agree) * max_distances, 1
    )

    distance_closest_incorrect, idx_closest_incorrect = torch.min(
        distances + labels_agree * max_distances, 1
    )

    return (
        distance_closest_correct,
        idx_closest_correct,
        distance_closest_incorrect,
        idx_closest_incorrect,
    )


def glvq_loss(
    *,
    distances: Tensor,
    prototype_labels: Tensor,
    data_labels: Tensor,
    eps: float = 1.0e-8,
) -> Tensor:
    r"""GLVQ loss function.

    Functional implementation of the :class:`.GLVQLoss`. See this class for
    further information.

    :param distances: Tensor of distances.
    :param prototype_labels: The labels of the prototypes one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :param eps: Small epsilon added to division by zero.
    :return: ``loss``: Tensor of loss values for each sample.

    :Shapes-In:
        - ``distances``: (*number_of_samples*, *number_of_prototypes*).
        - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).

    :Shapes-Out:
        - ``loss``: (*number_of_samples*,).

    :Example:

    >>> distances = torch.rand(64, 4)
    >>> prototype_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> output = loss.glvq_loss(
    ...     distances=distances,
    ...     prototype_labels=prototype_labels,
    ...     data_labels=data_labels,
    ... )
    """
    _tensor_dimension_and_size_check(
        data_template_comparisons=distances,
        template_labels=prototype_labels,
        data_labels=data_labels,
    )

    (
        distance_closest_correct,
        _,
        distance_closest_incorrect,
        _,
    ) = _closest_correct_and_incorrect_distance(
        distances=distances, prototype_labels=prototype_labels, data_labels=data_labels
    )

    loss_per_sample = (distance_closest_correct - distance_closest_incorrect) / (
        distance_closest_correct + distance_closest_incorrect + eps
    )

    return loss_per_sample


def margin_loss(
    *,
    data_template_comparisons: Tensor,
    template_labels: Tensor,
    data_labels: Tensor,
    margin: float,
    similarity: bool,
) -> Tensor:
    r"""Margin loss function.

    Functional implementation of the :class:`.MarginLoss`. See this class for
    further information.

    :param data_template_comparisons: Tensor of comparison values. This could be a
        vector of distances or probabilities that is later used to determine the
        output class.
    :param template_labels: The labels of the templates one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :param margin: Positive value that specifies the minimal margin to be achieved
        between the best correct and incorrect template.
    :param similarity: A binary parameter that determines whether similarities
        (maximum determines the winner) or dissimilarities (minimum determines the
        winner) are expected for ``data_template_comparisons``.
    :return: ``loss``: Tensor of loss values for each sample.

    :Shapes-In:
        - ``data_template_comparisons``: (*number_of_samples*, *number_of_templates*).
        - ``template_labels``: (*number_of_templates*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).

    :Shapes-Out:
        - ``loss``: (*number_of_samples*,).

    :Example:

    >>> data_template_comparisons = torch.rand(64, 4)
    >>> template_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> output = loss.margin_loss(
    ...     data_template_comparisons=data_template_comparisons,
    ...     template_labels=template_labels,
    ...     data_labels=data_labels,
    ...     margin=0.3,
    ...     similarity=True
    ... )
    """
    _tensor_dimension_and_size_check(
        data_template_comparisons=data_template_comparisons,
        template_labels=template_labels,
        data_labels=data_labels,
    )

    if similarity:
        if template_labels.shape[0] == template_labels.shape[1]:
            max_probability_per_class = data_template_comparisons @ template_labels
        else:
            max_probability_per_class, _ = torch.max(
                template_labels.unsqueeze(0) * data_template_comparisons.unsqueeze(-1),
                dim=1,
            )

        # In the following, the minus is required to have a minimization problem:
        # best_correct -> highest_correct_probability
        best_correct = -torch.sum(data_labels * max_probability_per_class, dim=1)

        # best_incorrect -> highest_incorrect_probability
        best_incorrect, _ = torch.max(max_probability_per_class - data_labels, dim=1)
        best_incorrect = -best_incorrect

    else:
        # best_correct -> distance_closest_correct
        # best_incorrect -> distance_closest_incorrect
        (best_correct, _, best_incorrect, _) = _closest_correct_and_incorrect_distance(
            distances=data_template_comparisons,
            prototype_labels=template_labels,
            data_labels=data_labels,
        )

    loss_per_sample = torch.relu(best_correct - best_incorrect + margin)

    return loss_per_sample


def robust_stable_cbc_loss(
    *,
    data_template_comparisons: Tensor,
    template_labels: Tensor,
    data_labels: Tensor,
    requiredness_probabilities: Optional[Tensor],
    component_probabilities: Optional[Tensor],
    sigma: Tensor,
    margin: Optional[float],  # should be positive but this is not checked
    reasoning_probabilities: Optional[Tensor] = None,
    eps: float = 1.0e-8,
) -> Tensor:
    r"""Robust stable CBC loss implementation.

    This loss is the robust version of the loss for ordinary distance functions
    (Theorem 1 in the StableCBC paper); not squared or tangent distance. The respective
    changes for these special cases can be easily computed by using this implementation.

    The implementation is optimized to increase the efficiency of the implementation.
    The following strategies are used:

    - We use broadcasting to increase the efficiency of the implementation.
    - We only compute ``a`` and ``b`` and use the identity ``a = -c.T`` in the
      computation.
    - We compute the robustness for each reasoning concept i, j and extract the correct
      values in the end by masking and computing the min-max robustness (min over all
      incorrect indices and max over all correct indices). This means that we compute
      the loss for combinations where i and j are from the same class, which doesn't
      make sense, and filter these values later.
    - According to the theorem, we compress sigma by min (no matter how the shape is).
    - We compute the min-max before the log to avoid that we apply this expensive
      operation to each value.
    - We check that ``a`` is not close to zero (eps). If so, we replace it with ``eps``.

    Note that the loss is positve for incorrect classifications. Moreover, we don't
    support a class implementation for this loss as it violates our generic class
    interface since the reasoning probabilities have to be provided all the time.

    The function supports the computation by a given requiredness and component
    probability tensor ot by a given reasoning probability tensors. If all three
    tensors are given, the function raises a ValueError.

    :param data_template_comparisons: Tensor of component similarity values.
    :param template_labels: The labels of the templates one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :param requiredness_probabilities: Tensor of requiredness probabilities for each
        class or ``None``.
    :param component_probabilities: Class-wise component prior probabilities tensor
         or ``None``.
    :param reasoning_probabilities: Tensor of reasoning probabilities for each class
         or ``None``.
    :param sigma: The tensor with the temperature(s).
    :param margin: The margin value if the loss should optimize for a minimal
        robustness value.
    :param eps: Epsilon value to stabilize the division.
    :raises ValueError: If not ``requiredness_probabilities`` and
        ``component_probabilities`` are Tensor and ``reasoning_probabilities`` is None
        OR not ``requiredness_probabilities`` and
        ``component_probabilities`` are None and ``reasoning_probabilities`` is Tensor.
    :raises ValueError: If the number of components is different in
        ``data_template_comparisons`` and the probabilities.
    :return: Loss values per sample.

    :Shapes-In:
        - ``data_template_comparisons``:
          (*number_of_samples*, *number_of_reasoning_concepts*).
        - ``template_labels``: (*number_of_reasoning_concepts*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).
        - ``requiredness_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*) if not None.
        - ``component_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*) if not None.
        - ``reasoning_probabilities``:
          (2 \* *number_of_components*, *number_of_reasoning_concepts*) if not None.
        - ``sigma``: Any suitable shape.

    :Shapes-Out:
        - ``loss_per_sample``: (*number_of_samples*, *number_of_reasoning_concepts*,).

    :Example:

    >>> data_template_comparisons = torch.rand(64, 4)
    >>> template_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> reasoning_probabilities = torch.softmax(torch.rand(8, 4), dim=0)
    >>> sigma = torch.rand(4)
    >>> output = loss.robust_stable_cbc_loss(
    ...     data_template_comparisons=data_template_comparisons,
    ...     template_labels=template_labels,
    ...     data_labels=data_labels,
    ...     reasoning_probabilities=reasoning_probabilities,
    ...     requiredness_probabilities=None,
    ...     component_probabilities=None,
    ...     sigma=sigma,
    ...     margin=None,
    ... )
    >>>
    >>> component_probabilities = torch.softmax(torch.rand(4, 4), dim=0)
    >>> requiredness_probabilities = torch.rand(4, 4)
    >>> output = loss.robust_stable_cbc_loss(
    ...     data_template_comparisons=data_template_comparisons,
    ...     template_labels=template_labels,
    ...     data_labels=data_labels,
    ...     reasoning_probabilities=None,
    ...     requiredness_probabilities=requiredness_probabilities,
    ...     component_probabilities=component_probabilities,
    ...     sigma=sigma,
    ...     margin=None,
    ... )
    """
    if not (
        requiredness_probabilities is not None
        and component_probabilities is not None
        and reasoning_probabilities is None
    ) and not (
        requiredness_probabilities is None
        and component_probabilities is None
        and reasoning_probabilities is not None
    ):
        raise ValueError(
            "Either requiredness_probabilities and component_probabilities must be "
            "``None`` and reasoning_probabilities must not be ``None`` or "
            "requiredness_probabilities and component_probabilities must not be "
            "``None`` and reasoning_probabilities must be ``None``."
        )

    # We mock data_template_comparisons as the size is (batches, number_components)
    # but should be (batches, number_reasoning_concepts).
    _tensor_dimension_and_size_check(
        data_template_comparisons=torch.empty(
            (data_template_comparisons.shape[0], template_labels.shape[0]),
            dtype=torch.bool,
        ),
        template_labels=template_labels,
        data_labels=data_labels,
    )

    if requiredness_probabilities is not None:
        number_of_components = requiredness_probabilities.shape[0]
    else:
        number_of_components = (
            reasoning_probabilities.shape[0] / 2  # type: ignore[union-attr]
        )

    if data_template_comparisons.shape[1] != number_of_components:
        raise ValueError(
            f"The number of components does not match: "
            f"data_template_comparisons.shape[1]="
            f"{data_template_comparisons.shape[1]} != "
            f"number_of_components={number_of_components} (inferred from the reasoning "
            f"or requiredness probabilities)."
        )

    eps = torch.tensor(eps)

    if reasoning_probabilities is None:
        reasoning_probabilities = (
            requiredness_probabilities * component_probabilities
        )  # noqa: both probs are not None in this case
    else:
        component_probabilities = reasoning_probabilities.reshape(
            2, data_template_comparisons.shape[-1], -1
        ).sum(dim=0)
        reasoning_probabilities = reasoning_probabilities[
            0 : data_template_comparisons.shape[-1]
        ]

    # Compute the coefficients of the quadratic equation by broadcasting.
    a = (
        data_template_comparisons @ (reasoning_probabilities - component_probabilities)
    ).unsqueeze(-1) - (data_template_comparisons @ reasoning_probabilities).unsqueeze(1)
    b = torch.sum((component_probabilities - reasoning_probabilities), 0, keepdim=True)
    b = (b.T - b).unsqueeze(0)

    # These are the pre-robustness values (without the log and the sigma). From these
    # values, we have to filter the valid broadcasting combinations.
    pre_robustness = -(
        b + torch.sqrt(torch.maximum(b**2 + 4 * a * a.transpose(1, 2), eps))
    ) / (2 * torch.minimum(a, -eps))

    # We use these values for the masking of invalid broadcasting combinations
    max_value = torch.max(pre_robustness)
    min_value = torch.min(pre_robustness)

    # Determine which of the reasoning concepts are from the correct class. We use this
    # for the masking of the invalid broadcasting combinations.
    correct_templates = data_labels @ template_labels.T

    # Filter the valid broadcasting operations.
    max_pre_robustness = torch.max(
        pre_robustness
        - ((1 - correct_templates) * (max_value - min_value + 1)).unsqueeze(2),
        dim=1,
    )[0]
    min_max_pre_robustness = torch.min(
        max_pre_robustness + correct_templates * (max_value - min_value + 1),
        dim=1,
    )[0]

    # Compute robustness values.
    robustness = torch.min(sigma) * torch.log(
        torch.maximum(min_max_pre_robustness, eps)
    )

    if margin is not None:
        loss_per_sample = torch.relu(margin - robustness)
    else:
        # negate to have a minimization problem
        loss_per_sample = -robustness

    return loss_per_sample


class TemplateInputComparisonBasedLoss(Module, ABC):
    r"""Base class for a template-input-comparison-based loss.

    Can be used to implement custom loss functions. See the classes :class:`.GLVQLoss`
    or :class:`.MarginLoss` for examples. The naming is generic so that 'comparison'
    could mean distances or similarities and 'templates' could be prototypes or
    components.

    :param dimension_and_size_check: Determines whether the dimensions and sizes of the
        inputs are checked. If this check is already done in the loss function call, set
        this to False.
    """

    def __init__(self, dimension_and_size_check: bool = True):
        r"""Initialize an object of the class."""
        super().__init__()
        self.dimension_and_size_check = dimension_and_size_check

    def forward(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Calls the :meth:`.loss_function` and depending on ``dimension_and_size_check``
        the dimension and size check.

        :param data_template_comparisons: Tensor of distances or similarities.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: ``loss``: Tensor of loss values for each sample.

        :Shapes-In:
            - ``data_template_comparisons``:
              (*number_of_samples*, *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).

        :Shapes-Out:
            - ``loss``: (*number_of_samples*,).
        """
        if self.dimension_and_size_check:
            _tensor_dimension_and_size_check(
                data_template_comparisons=data_template_comparisons,
                template_labels=template_labels,
                data_labels=data_labels,
            )

        loss = self.loss_function(
            data_template_comparisons,
            template_labels,
            data_labels,
        )

        return loss

    @abstractmethod
    def loss_function(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Abstract method for the loss function implementation.

        The loss function should be implemented here. To support different namings of
        the arguments this function is called by :meth:`.forward` without keywords.
        Therefore, **do not** change the order of the arguments when implementing this
        method.

        :param data_template_comparisons: Tensor of distances or similarities.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: ``loss``: Tensor of loss values for each sample.

        :Shapes-In:
            - ``data_template_comparisons``:
                (*number_of_samples*, *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).
        """


class GLVQLoss(TemplateInputComparisonBasedLoss):
    r"""Computes the GLVQ loss.

    The GLVQ loss is defined as: Let :math:`d^+` be the distance to the closest
    prototype of the correct class and :math:`d^-` be the distance to the closest
    prototype of an incorrect class (correct means same class as the data point).
    Then, the GLVQ loss is

    .. math::
        \mathrm{loss}(\mathrm{all\_prototypes}, x) = \frac{d^+ - d^-}{d^+ + d^- + eps}.

    The epsilon is added to avoid division by zero.

    Note that the loss values are always returned **not accumulated**. This means the
    loss value for each sample is returned.

    :param eps: Small epsilon added to division by zero.

    :Example:

    >>> distances = torch.rand(64, 4)
    >>> prototype_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> loss_func = loss.GLVQLoss(eps=1.e-7)
    >>> output = loss_func(distances, prototype_labels, data_labels)
    """

    def __init__(self, eps: float = 1.0e-8) -> None:
        r"""Initialize an object of the class."""
        super().__init__(dimension_and_size_check=False)
        self.eps = eps

    def forward(
        self,
        distances: Tensor,
        prototype_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        :param distances: Tensor of distances.
        :param prototype_labels: The labels of the prototypes one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: ``loss``: Tensor of loss values for each sample.

        :Shapes-In:
            - ``distances``: (*number_of_samples*, *number_of_prototypes*).
            - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).

        :Shapes-Out:
            - ``loss``: (*number_of_samples*,).
        """
        loss = super().forward(
            data_template_comparisons=distances,
            template_labels=prototype_labels,
            data_labels=data_labels,
        )

        return loss

    def loss_function(
        self,
        distances: Tensor,
        prototype_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""GLVQ loss computation.

        :param distances: Tensor of distances.
        :param prototype_labels: The labels of the prototypes one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: ``loss``: Tensor of loss values for each sample.

        :Shapes-In:
            - ``distances``: (*number_of_samples*, *number_of_prototypes*).
            - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).

        :Shapes-Out:
            - ``loss``: (*number_of_samples*,).
        """
        loss = glvq_loss(
            distances=distances,
            prototype_labels=prototype_labels,
            data_labels=data_labels,
            eps=self.eps,
        )

        return loss


class MarginLoss(TemplateInputComparisonBasedLoss):
    r"""Computes the Margin loss for similarities or dissimilarities.

    This loss is the margin loss implementation with a specifiable ``margin``
    value. If ``similarity`` is ``True``, then the loss assumes that the comparison
    values are based on similarities (e.g., probabilities) so that large values mean
    high similarity. In this case, the loss is given by

    .. math::
        \mathrm{loss}(\mathrm{all\_templates},x) =
        \max\left\{ s^- - s^+ + \mathrm{margin}, 0 \right\},

    where :math:`s^+` is the highest similarity of a template of the same class as the
    input and :math:`s^-` is the highest similarity of a template of a different
    class than the input :math:`x`. In case of, dissimilarities (e.g., distance
    functions) the loss becomes

    .. math::
        \mathrm{loss}(\mathrm{all\_templates},x) =
        \max\left\{ d^+ - d^- + \mathrm{margin}, 0 \right\},

    where :math:`d^+` is the smallest dissimilarity of a template of the same class as
    the input and :math:`d^-` is the smallest dissimilarity of a template of a
    different class than the input :math:`x`.

    The function uses the word "template" instead of using the names "prototypes" or
    "components". However, depending on the use case, a "template" could be a
    "prototype". Moreover, instead of "similarity" or "dissimilarity", the function
    uses the word 'comparison'.

    Note that the loss values are always returned **not accumulated**. This means the
    loss value for each sample is returned.

    :param margin: The margin value of the loss function. Usually, a positive value.
    :param similarity: A boolean value to specify if the input is a similarity or
        dissimilarity.

    :Example:

    >>> distances = torch.rand(64, 4)
    >>> prototype_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> loss_func = loss.MarginLoss(margin=0.3, similarity=False)
    >>> output = loss_func(distances, prototype_labels, data_labels)
    """

    def __init__(self, margin: float, similarity: bool) -> None:
        r"""Initialize an object of the class."""
        super().__init__(dimension_and_size_check=False)
        self.margin = margin
        self.similarity = similarity

    def loss_function(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Margin loss computation.

        :param data_template_comparisons: Tensor of comparison values. This could be a
            vector of distances or probabilities that is later used to determine the
            output class.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: ``loss``: Tensor of loss values for each sample.

        :Shapes-In:
            - ``data_template_comparisons``: (*number_of_samples*,
              *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).

        :Shapes-Out:
            - ``loss``: (*number_of_samples*,).
        """
        loss = margin_loss(
            data_template_comparisons=data_template_comparisons,
            template_labels=template_labels,
            data_labels=data_labels,
            margin=self.margin,
            similarity=self.similarity,
        )

        return loss
