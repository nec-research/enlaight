"""Module with reasoning functions."""

import warnings
from typing import Dict, Optional, Union

import torch
from torch import Tensor

# Right now, I don't think that we need a class support for these functions since
# they are directly called within CBCs - Sascha.


def cbc_reasoning(
    *,
    detection_probabilities: Tensor,
    requiredness_probabilities: Tensor,
    component_probabilities: Optional[Tensor] = None,
    eps: float = 1.0e-7,
    full_report: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Probability computation according to original CBC.

    The Classification-by-Components networks are based on a specific computation of
    the output probabilities derived from a probability tree diagram. See this paper
    as a reference:

        `"Classification-by-Components: Probabilistic Modeling of Reasoning over a Set
        of Components" by Saralajew et al., 2019.
        <https://proceedings.neurips.cc/paper_files/paper/2019/file/
        dca5672ff3444c7e997aa9a2c4eb2094-Paper.pdf>`_

    Please note that there is a difference in the wording between the paper and
    the implementation. In the paper, we call the requiredness probabilities reasoning
    probabilities. However, here, reasoning probabilities are the probabilities that
    are multiplied with the detection and model the classification output.

    The idea is that the detection of a component (template) can contribute with its
    presence or absence to the recognition of a class. Therefore, negative (absence
    of the class contributes to the detection), positive (presence of the class
    contributes to the detection), and indefinite requiredness (the component is
    ignored) can happen. Note that the probabilities of negative, positive,
    and indefinite requiredness must sum up to one for each component and class. Here,
    it is assumed that ``requiredness_probabilities`` provides only the positive and the
    negative requiredness probabilities (as this implicitly contains the indefinite
    requiredness probabilities). In particular, ``requiredness_probabilities[0]``
    corresponds to the positive and ``requiredness_probabilities[1]`` corresponds to the
    negative requiredness probabilities. Consequently, the sum over the first dimension
    of the tensor (i.e., ``requiredness_probabilities.sum(0)``) must be less than or
    equal to 1. Moreover, because all values in all tensors are probability values,
    each value must be an element from the unit interval. If a
    ``component_probabilities`` tensor is provided, it must fulfill the condition
    ``component_probabilities.sum(0) == 1`` since it represents the class-independent
    prior probability of detecting a component over the data space. It is **important
    to note** that these "probability conditions" are not checked in the function (
    for efficiency reasons). *Thus, the user has to ensure that appropriate values
    are provided.*

    The function supports batches with respect to the ``detection_probabilities``.

    Note that the *number_of_reasonings* does not necessarily equal the number of
    classes. Each reasoning is associated with a class label by the externally stored
    reasoning labels.

    :param detection_probabilities: Tensor of probabilities of detected components in
        the inputs.
    :param requiredness_probabilities: Tensor of requiredness probabilities for each
        class.
    :param component_probabilities: Component-wise prior probabilities. If ``None``,
        it is assumed that each component is equally important.
    :param eps: Small epsilon that is added to stabilize the division.
    :param full_report: If ``False``, only the agreement probability is computed (the
        class probability). If ``True``, all internal probabilities of the reasoning
        process are returned as *detached* and *cloned* tensors. See "return" for the
        returned probabilities.
    :raises ValueError: If ``detection_probabilities`` or ``requiredness_probabilities``
        are not of :class:`torch.Tensor`.
    :raises ValueError: If ``component_probabilities`` is not of :class:`torch.Tensor`
        or ``None``.
    :raises ValueError: If ``detection_probabilities.shape[-1]`` does not match
        ``requiredness_probabilities.shape[1]``.
    :raises ValueError: If ``component_probabilities`` is not ``None`` and
        ``component_probabilities.shape[0]`` is unequal
        ``detection_probabilities.shape[-1]``.
    :raises ValueError: If ``requiredness_probabilities.shape[0]`` is not 2.
    :raises ValueError: If ``detection_probabilities.dim()`` is not 1 or 2.
    :raises ValueError: If ``requiredness_probabilities.dim()`` is not 3.
    :raises ValueError: If ``component_probabilities`` is not ``None`` and
        ``component_probabilities.dim()`` is not 1.
    :return:
        - ``probabilities``: If ``full_report==False``, the agreement probabilities
          tensor, where ``output[i]`` is the probability for agreement ``i`` if
          ``detection_probabilities.dim()==1``.
        - ``probabilities``: If ``full_report==False``, the agreement probabilities
          tensor, where ``output[i,j]`` is the probability for agreement ``j`` given the
          input ``i`` from the batch of ``detection_probabilities`` if
          ``detection_probabilities.dim()==2``.
        - ``report``: If ``full_report==True``, for each probability tensor returned in
          the dictionary, the specification for ``full_report==False`` is correct. The
          dictionary holds the following probability tensors: *agreement probability*
          (key 'agreement'), *disagreement probability* (key 'disagreement'),
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
        - ``requiredness_probabilities``:
          (2, *number_of_components*, *number_of_reasoning_concepts*).
        - ``component_probabilities``: (*number_of_components*,).

    :Shapes-Out:
        - ``probabilities``: (*number_of_reasoning_concepts*,) If ``full_report==False``
          and ``detection_probabilities.dim()==1``.
        - ``probabilities``: (*batch*, *number_of_reasoning_concepts*,) If
          ``full_report==False`` and ``detection_probabilities.dim()==2``.
        - ``report``: {*} If ``full_report==True``, dictionary of tensors with the
          format specified for ``full_report==False``.

    :Example:

    >>> detection_probabilities = torch.tensor([[1, 0, 0], [0.25, 0.5, 1]])
    >>> requiredness_probabilities = torch.tensor(
    ...     [[[0.25, 0], [0, 1], [0, 0]], [[0.25, 1], [1, 0], [0, 1]]]
    ... )
    >>> probabilities = reasoning.cbc_reasoning(
    ...     detection_probabilities=detection_probabilities,
    ...     requiredness_probabilities=requiredness_probabilities,
    ... )
    """
    if not isinstance(detection_probabilities, Tensor) or not isinstance(
        requiredness_probabilities, Tensor
    ):
        raise ValueError(
            f"The detection_probabilities and requiredness_probabilities must be of "
            f"class Tensor. Provided type(detection_probabilities)="
            f"{type(detection_probabilities)} and type(requiredness_probabilities)="
            f"{type(requiredness_probabilities)}."
        )

    if (
        not isinstance(component_probabilities, Tensor)
        and component_probabilities is not None
    ):
        raise ValueError(
            f"The component_probabilities and requiredness_probabilities must be of "
            f"class Tensor or None. Provided type(component_probabilities)="
            f"{type(component_probabilities)}."
        )

    number_of_components = detection_probabilities.shape[-1]

    if number_of_components != requiredness_probabilities.shape[1]:
        raise ValueError(
            f"The number of components must be equal in detection_probabilities "
            f"(last dim) and requiredness_probabilities (dim 1). "
            f"Provided detection_probabilities.shape={detection_probabilities.shape} "
            f"and requiredness_probabilities.shape={requiredness_probabilities.shape}."
        )

    if (
        component_probabilities is not None
        and number_of_components != component_probabilities.shape[0]
    ):
        raise ValueError(
            f"The number of components must be equal in detection_probabilities "
            f"(last dim) and component_probabilities (dim 0). "
            f"Provided detection_probabilities.shape={detection_probabilities.shape} "
            f"and component_probabilities.shape={component_probabilities.shape}."
        )

    if (
        requiredness_probabilities.dim() != 3
        or requiredness_probabilities.shape[0] != 2
    ):
        raise ValueError(
            f"requiredness_probabilities is expected to have 3 dimensions and the size "
            f"of the first one must be 2. Provided requiredness_probabilities.shape="
            f"{requiredness_probabilities.shape}."
        )

    if detection_probabilities.dim() not in (1, 2):
        raise ValueError(
            f"detection_probabilities must be a tensor of dim equals 1 or 2. "
            f"Provided detection_probabilities.dim()="
            f"{detection_probabilities.dim()}."
        )

    if component_probabilities is not None and component_probabilities.dim() != 1:
        raise ValueError(
            f"component_probabilities must be a vector (tensor of dim=1). "
            f"Provided component_probabilities.dim()="
            f"{component_probabilities.dim()}."
        )

    # make a copy before we may override it
    if full_report:
        requiredness_probabilities_ = requiredness_probabilities

    if component_probabilities is not None:
        requiredness_probabilities = (
            requiredness_probabilities * component_probabilities.reshape(1, -1, 1)
        )

    positive_requiredness_probabilities = requiredness_probabilities[0]
    negative_requiredness_probabilities = requiredness_probabilities[1]

    # stabilize the division with a small epsilon
    probabilities = (
        detection_probabilities
        @ (positive_requiredness_probabilities - negative_requiredness_probabilities)
        + torch.sum(negative_requiredness_probabilities, 0)
    ) / (
        torch.sum(
            positive_requiredness_probabilities + negative_requiredness_probabilities,
            dim=0,
        )
        + eps
    )

    if not full_report:
        return probabilities
    else:
        positive_reasoning_probabilities = positive_requiredness_probabilities / (
            torch.sum(
                (
                    positive_requiredness_probabilities
                    + negative_requiredness_probabilities
                ),
                dim=0,
            ).reshape(1, -1)
            + eps
        )

        negative_reasoning_probabilities = negative_requiredness_probabilities / (
            torch.sum(
                positive_requiredness_probabilities
                + negative_requiredness_probabilities,
                dim=0,
            ).reshape(1, -1)
            + eps
        )

        if component_probabilities is None:
            component_probabilities = (
                torch.ones(requiredness_probabilities.shape[1]).float()
                / requiredness_probabilities.shape[1]
            )

        report: Dict[str, Tensor] = {
            "agreement": probabilities,
            "disagreement": 1 - probabilities,
            "detection": detection_probabilities,
            "positive agreement": (
                detection_probabilities @ positive_reasoning_probabilities
            ),
            "negative agreement": (
                (1 - detection_probabilities) @ negative_reasoning_probabilities
            ),
            "positive disagreement": (
                (1 - detection_probabilities) @ positive_reasoning_probabilities
            ),
            "negative disagreement": (
                detection_probabilities @ negative_reasoning_probabilities
            ),
            "positive reasoning": positive_reasoning_probabilities,
            "negative reasoning": negative_reasoning_probabilities,
            "positive requiredness": requiredness_probabilities_[0],
            "indefinite requiredness": (
                1 - torch.sum(requiredness_probabilities_, dim=0)
            ),
            "negative requiredness": requiredness_probabilities_[1],
            "component prior": component_probabilities,
        }

        for key in report.keys():
            report[key] = report[key].detach().clone()

        return report


def stable_cbc_reasoning(
    *,
    detection_probabilities: Tensor,
    requiredness_probabilities: Optional[Tensor],
    component_probabilities: Optional[Tensor],
    reasoning_probabilities: Optional[Tensor] = None,
    full_report: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    r"""Probability computation according to stable CBC.

    Conceptually, the idea of this reasoning approach is inspired by the standard CBC
    reasoning. The difference is that the indefinite reasoning state is removed and
    instead a class-wise component prior probability is learned. Consequently,
    we have only positive and negative requiredness (reasoning in the standard CBC).

    If you input ``requiredness_probabilities``, it is assumed that you also input
    ``component_probabilities`` and that ``reasoning_probabilities`` is ``None``.
    Moreover , it is assumed that ``requiredness_probabilities`` provides only the
    positive requiredness probabilities (as this implicitly contains the negative
    requiredness probabilities). Consequently, all elements must be from the unit
    interval. The ``component_probabilities`` tensor must fulfill the condition
    ``component_probabilities.sum(0) == 1`` since it represents the class-wise
    component prior probability of detecting a component over the data space.

    If you input ``reasoning_probabilities``, it is assumed that
    ``requiredness_probabilities`` and ``component_probabilities`` are ``None``.
    Morover, it is assumed that ``reasoning_probabilities.sum(0) == 1`` and the first
    half of the tensore corresponds to positive reasoning and the second half to
    negative reasoning. The ``component_probabilities`` can be computed via
    marginalization by summing the positive and negative reasoning probabilities.
    Similarily, the ``reasoning_probabilities`` can be derived.

    It is **important to note** that the mentioned "probability conditions" are not
    checked in the function (for efficiency reasons). *Thus, the user has to ensure
    that appropriate values are provided.*

    The function supports batches with respect to the ``detection_probabilities``.

    Note that the *number_of_reasonings* does not necessarily equal the number of
    classes. Each reasoning is associated with a class label by the externally stored
    reasoning labels.

    :param detection_probabilities: Tensor of probabilities of detected components in
        the inputs.
    :param requiredness_probabilities: Tensor of requiredness probabilities for each
        class or ``None``.
    :param component_probabilities: Class-wise component prior probabilities tensor
         or ``None``.
    :param reasoning_probabilities: Tensor of reasoning probabilities for each class
         or ``None``.
    :param full_report: If ``False``, only the agreement probability is computed (the
        class probability). If ``True``, all internal probabilities of the reasoning
        process are returned as *detached* and *cloned* tensors. See "return" for the
        returned probabilities.
    :raises ValueError: If ``detection_probabilities``, ``requiredness_probabilities``,
        ``reasoning_probabilities``, or ``component_probabilities`` are not of
        :class:`torch.Tensor` or ``None``.
    :raises ValueError: If not ``requiredness_probabilities`` and
        ``component_probabilities`` are :class:`torch.Tensor` and
        ``reasoning_probabilities`` is ``None``. If not ``reasoning_probabilities`` is
        a :class:`torch.Tensor` and ``requiredness_probabilities`` and
        ``component_probabilities` are ``None``.
    :raises ValueError: If ``detection_probabilities.shape[-1]`` does not match
        ``requiredness_probabilities.shape[0]`` or ``component_probabilities.shape[0]``
        assuming that they are not None.
    :raises ValueError: If ``component_probabilities.shape[1]`` is unequal
        ``requiredness_probabilities.shape[1]`` assuming that they are not None.
    :raises ValueError: If ``detection_probabilities.dim()`` is not 1 or 2.
    :raises ValueError: If ``reasoning_probabilities.dim()`` is not 2 assuming that
        it is not None.
    :raises ValueError: If ``reasoning_probabilities.shape[0]`` does not match
        ``2 * detection_probabilities.shape[-1]`` assuming that they are not None.
    :return:
        - ``probabilities``: If ``full_report==False``, the agreement probabilities
          tensor, where ``output[i]`` is the probability for agreement ``i`` if
          ``detection_probabilities.dim()==1``.
        - ``probabilities``: If ``full_report==False``, the agreement probabilities
          tensor, where ``output[i,j]`` is the probability for agreement ``j`` given the
          input ``i`` from the batch of ``detection_probabilities`` if
          ``detection_probabilities.dim()==2``.
        - ``report``: If ``full_report==True``, for each probability tensor returned in
          the dictionary, the specification for ``full_report==False`` is correct. The
          dictionary holds the following probability tensors: *agreement probability*
          (key 'agreement'), *disagreement probability* (key 'disagreement'),
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
        - ``detection_probabilities``: (*number_of_components*,) or
          (*batch*, *number_of_components*).
        - ``requiredness_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``component_probabilities``:
          (*number_of_components*, *number_of_reasoning_concepts*).
        - ``reasoning_probabilities``:
          (2 \* *number_of_components*, *number_of_reasoning_concepts*).

    :Shapes-Out:
        - ``probabilities``: (*number_of_reasoning_concepts*,) If ``full_report==False``
          and ``detection_probabilities.dim()==1``.
        - ``probabilities``: (*batch*, *number_of_reasoning_concepts*,) If
          ``full_report==False`` and ``detection_probabilities.dim()==2``.
        - ``report``: {*} If ``full_report==True``, dictionary of tensors with the
          format specified for ``full_report==False``.

    :Example:

    >>> detection_probabilities = torch.tensor([[0.25, 0.5, 1], [1, 0, 0]])
    >>> requiredness_probabilities = torch.tensor([[0.5, 0], [0, 0.5], [0, 0]])
    >>> component_probabilities = torch.tensor([[0.5, 0.5], [0.25, 0], [0.25, 0.5]])
    >>> probabilities = reasoning.stable_cbc_reasoning(
    ...     detection_probabilities=detection_probabilities,
    ...     requiredness_probabilities=requiredness_probabilities,
    ...     component_probabilities=component_probabilities,
    ... )
    >>> detection_probabilities = torch.tensor([[0.25, 1], [1, 0]])
    >>> reasoning_probabilities = torch.tensor([[0.5, 0], [0, 0.5], [0, 0.5], [0.5, 0]])
    >>> probabilities = reasoning.stable_cbc_reasoning(
    ...     detection_probabilities=detection_probabilities,
    ...     reasoning_probabilities=reasoning_probabilities,
    ... )
    """
    if (
        requiredness_probabilities is not None
        and component_probabilities is not None
        and reasoning_probabilities is None
    ):
        requiredness_given = True
    elif (
        requiredness_probabilities is None
        and component_probabilities is None
        and reasoning_probabilities is not None
    ):
        requiredness_given = False
    else:
        raise ValueError(
            "Either requiredness_probabilities and component_probabilities must be "
            "``None`` and reasoning_probabilities must not be ``None`` or "
            "requiredness_probabilities and component_probabilities must not be "
            "``None`` and reasoning_probabilities must be ``None``."
        )

    if requiredness_given is False:
        if not isinstance(detection_probabilities, Tensor) or not isinstance(
            reasoning_probabilities, Tensor
        ):
            raise ValueError(
                f"The detection_probabilities and reasoning_probabilities "
                f"must be of class Tensor. Provided type(detection_probabilities)="
                f"{type(detection_probabilities)} and type(reasoning_probabilities)="
                f"{type(reasoning_probabilities)}."
            )

        if 2 * detection_probabilities.shape[-1] != reasoning_probabilities.shape[0]:
            raise ValueError(
                f"The number of reasoning_probabilities (dim 0) must be twice the "
                f"number of detection_probabilities (last dim). "
                f"Provided detection_probabilities.shape="
                f"{detection_probabilities.shape} and reasoning_probabilities.shape="
                f"{reasoning_probabilities.shape}."
            )

        if reasoning_probabilities.dim() != 2:
            raise ValueError(
                f"reasoning_probabilities are expected "
                f"to have dimensions 2. Provided reasoning_probabilities.dim()="
                f"{reasoning_probabilities.dim()}."
            )

    else:
        if (
            not isinstance(detection_probabilities, Tensor)
            or not isinstance(requiredness_probabilities, Tensor)
            or not isinstance(component_probabilities, Tensor)
        ):
            raise ValueError(
                f"The detection_probabilities, requiredness_probabilities, and "
                f"component probabilities must be of class"
                f"Tensor. Provided type(detection_probabilities)="
                f"{type(detection_probabilities)}, type(requiredness_probabilities)="
                f"{type(requiredness_probabilities)}, and "
                f"type(component_probabilities)={type(component_probabilities)}."
            )

        if (
            detection_probabilities.shape[-1] != requiredness_probabilities.shape[0]
            or detection_probabilities.shape[-1] != component_probabilities.shape[0]
        ):
            raise ValueError(
                f"The number of components must be equal in detection_probabilities "
                f"(last dim), requiredness_probabilities (dim 0), and "
                f"component_probabilities (dim 0). "
                f"Provided detection_probabilities.shape="
                f"{detection_probabilities.shape}, requiredness_probabilities.shape="
                f"{requiredness_probabilities.shape}, and "
                f"component_probabilities.shape={component_probabilities.shape}."
            )

        if requiredness_probabilities.dim() != 2 or component_probabilities.dim() != 2:
            raise ValueError(
                f"requiredness_probabilities and component_probabilities are expected "
                f"to have dimensions 2. Provided requiredness_probabilities.dim()="
                f"{requiredness_probabilities.dim()}, and "
                f"component_probabilities.dim()={component_probabilities.dim()}."
            )

        if requiredness_probabilities.shape[1] != component_probabilities.shape[1]:
            raise ValueError(
                f"The number of reasoning concepts (dim 1) is expected to be the same "
                f"in requiredness_probabilities and component_probabilities. "
                f"Provided requiredness_probabilities.shape="
                f"{requiredness_probabilities.shape} and component_probabilities.shape="
                f"{component_probabilities.shape}."
            )

    if detection_probabilities.dim() not in (1, 2):
        raise ValueError(
            f"detection_probabilities must be a tensor of dim equals 1 or 2. "
            f"Provided detection_probabilities.dim()="
            f"{detection_probabilities.dim()}."
        )

    if requiredness_given is False:
        # reasoning_probabilities[0] -> positive reasoning multiplied with prior
        # reasoning_probabilities[1] -> negative reasoning multiplied with prior
        reasoning_probabilities = reasoning_probabilities.reshape(
            2, detection_probabilities.shape[-1], -1
        )

        probabilities = detection_probabilities @ (
            reasoning_probabilities[0] - reasoning_probabilities[1]
        ) + torch.sum(reasoning_probabilities[1], dim=0)

    else:
        probabilities = torch.sum(
            (
                detection_probabilities.unsqueeze(-1)
                * (2 * requiredness_probabilities - 1)
                + 1
                - requiredness_probabilities
            )
            * component_probabilities,
            dim=detection_probabilities.dim() - 1,
        )

    if not full_report:
        return probabilities
    else:
        if requiredness_given is False:
            # reasoning probs are already in the correct view
            # compute component probabilities by marginalization
            component_probabilities = torch.sum(reasoning_probabilities, dim=0)

            # if a component prob is zero, we cannot restore the requiredness prob.
            # In this case, we assign the 0.5 to the requiredness and throw a warning.
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

        positive_reasoning_probabilities = (
            requiredness_probabilities * component_probabilities
        )

        negative_reasoning_probabilities = (
            1 - requiredness_probabilities
        ) * component_probabilities

        report: Dict[str, Tensor] = {
            "agreement": probabilities,
            "disagreement": 1 - probabilities,
            "detection": detection_probabilities,
            "positive agreement": (
                detection_probabilities @ positive_reasoning_probabilities
            ),
            "negative agreement": (
                (1 - detection_probabilities) @ negative_reasoning_probabilities
            ),
            "positive disagreement": (
                (1 - detection_probabilities) @ positive_reasoning_probabilities
            ),
            "negative disagreement": (
                detection_probabilities @ negative_reasoning_probabilities
            ),
            "positive reasoning": positive_reasoning_probabilities,
            "negative reasoning": negative_reasoning_probabilities,
            "positive requiredness": requiredness_probabilities,
            "negative requiredness": 1 - requiredness_probabilities,
            "component prior": component_probabilities,
        }

        for key in report.keys():
            report[key] = report[key].detach().clone()

        return report
