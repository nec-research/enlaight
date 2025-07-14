import argparse
import json
import os
from copy import deepcopy
from typing import List, Mapping, Tuple

import lightning as pl
import numpy as np
import torch
from autoattack import AutoAttack
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Lambda, ToTensor

from enlaight.core.detection_probability import ExponentiallySquashedNegativeDistance
from enlaight.core.distance import EuclideanDistance, TangentDistance
from enlaight.core.loss import MarginLoss, robust_stable_cbc_loss
from enlaight.core.utils import AffineSubspaces, Components
from enlaight.models import CBC, GLVQ, GTLVQ, RBF, RobustRBF, RobustStableCBC, StableCBC

if __name__ == "__main__":

    def list_of_float(arg):
        """Used to convert argparse to list of floats."""
        return list(map(float, arg.split(",")))

    methods = (
        "glvq",
        "gtlvq",
        "cbc",
        "cbc_td",
        "stable_cbc",
        "stable_cbc_td",
        "robust_rbf",
        "robust_rbf_td",
        "robust_stable_cbc",
        "robust_stable_cbc_td",
        "rbf",
        "rbf_td",
        "rbf_norm",
        "rbf_td_norm",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument("--squared", action="store_true", help="A boolean switch")
    parser.add_argument("--method", type=str, help=str(methods))
    parser.add_argument("--device", type=int, default=0, help="GPU device")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument(
        "--base_path", type=str, default=".", help="Root path of experiments"
    )
    parser.add_argument(
        "--robust_margin",
        type=float,
        default=1.58,
        help="Robustness margin for robust CBC training.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Margin for margin loss CBC training.",
    )
    parser.add_argument(
        "--eps_list",
        type=list_of_float,
        default=[0.5, 1, 1.58],
        help="List of epsilons used for the attacks (comma seperated without blanks)",
    )

    args = parser.parse_args()

    method = args.method
    if method not in methods:
        raise ValueError(f"{method} unknown. Should be some of these: {methods}")

    batch_size = 128
    epochs = args.epochs  # NOTE: increase to 40 to reproduce results
    steps = -1  # NOTE: set to -1 to reproduce results
    margin = args.margin  # Value of original CBC paper
    robust_margin = args.robust_margin
    runs = args.runs
    seeds = list(range(runs))
    number_classes = 10
    number_components = 20
    number_reasoning_per_class = 2
    subspace_dimension = 12
    eps_list = args.eps_list
    device = args.device
    squared = args.squared
    root_path = f"{args.base_path}/robustness_squared_{squared}"
    sigma_trainable = True
    initial_sigma = torch.ones(1, number_components) * 58  # change to torch.tensor(58) for one sigma shared
    negative_loss_weight = 1  # change to 0.09 for negative loss scaling

    def image_transform(x: np.ndarray) -> torch.tensor:
        x = ToTensor()(x)
        x = x.view(28 * 28)
        return x

    def label_transform(x: np.ndarray) -> torch.tensor:
        if method.find('rbf') == -1:  # rbf requires flat labels; find is -1 if rbf is not included in the method name
            x = torch.tensor(x, dtype=torch.int64)
            x = torch.nn.functional.one_hot(x, num_classes=number_classes).float()
        return x

    def clip(x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, min=0, max=1)

    train_dataset = MNIST(
        args.base_path,
        train=True,
        download=True,
        transform=Lambda(image_transform),
        target_transform=Lambda(label_transform),
    )
    test_dataset = MNIST(
        args.base_path,
        train=False,
        download=True,
        transform=Lambda(image_transform),
        target_transform=Lambda(label_transform),
    )

    if torch.cuda.is_available():
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=8,
            shuffle=True,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=8, shuffle=False
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def plot_components(
        model: pl.LightningModule, fig_size: Tuple[int, int], path: str
    ) -> None:
        try:
            os.makedirs(f"{path}/components")
        except OSError:
            return

        if hasattr(model, "prototypes"):
            _components = model.prototypes
        else:
            _components = model.components

        if hasattr(_components, "bias"):
            components = _components.bias.detach().cpu().numpy()
        else:
            components = _components.weight.detach().cpu().numpy()

        components = np.reshape(components, (components.shape[0], 28, 28))

        plt.figure(figsize=fig_size)
        for i, component in enumerate(components[:40]):
            plt.subplot(8, 5, i + 1)
            plt.imshow(component, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.title(f"Component {i}", fontsize=8)
            plt.imsave(
                f"{path}/components/component_{i}.png",
                (component * 255).astype("uint8"),
                cmap="gray",
            )

        plt.savefig(f"{path}/components/overview.pdf")

        plt.close("all")

    def plot_adversaries(
        model: pl.LightningModule, x_adv: np.ndarray, path: str, samples: int = 30
    ) -> None:
        try:
            os.makedirs(f"{path}/adversaries")
        except OSError:
            return

        for i in range(samples):
            plt.figure()

            output = model.predict_step(
                torch.tensor(x_adv[i : i + 1]).view(1, -1).to(model.device)
            )
            if len(output) > 1:
                predicted = output[0]
            else:
                predicted = output

            plt.imshow(x_adv[i, 0], cmap="gray")
            plt.title(f"Label {labels[i]}, Predicted {torch.argmax(predicted)}")
            plt.imsave(f"{path}/adversaries/sample_{i}.png", x_adv[i, 0], cmap="gray")
            plt.savefig(f"{path}/adversaries/sample_{i}.pdf")  # plt.show()

        plt.close("all")

    def plot_matrix(fig_size, matrix, xticks, yticks, title):
        fig, ax = plt.subplots(figsize=fig_size)
        ax.imshow(matrix, vmin=0, vmax=1)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                ax.text(
                    j,
                    i,
                    f"{int(matrix[i, j] * 100)}",
                    ha="center",
                    va="center",
                    color="w",
                )

        ax.set_title(title)
        fig.tight_layout()

    def plot_reasoning_matrices(
        model: pl.LightningModule,
        path: str,
        fig_size: Tuple[int, int],
        class_labels: List[int],
    ) -> None:
        try:
            os.makedirs(f"{path}/reasonings")
        except OSError:
            return

        reasoning_probabilities = (
            model.decoded_requiredness_probabilities.detach().cpu().numpy()
        )
        component_probabilities = (
            model.decoded_component_probabilities.detach().cpu().numpy()
        )

        if reasoning_probabilities.shape[0] == 2:
            probabilities = reasoning_probabilities * component_probabilities
        else:
            probabilities = reasoning_probabilities

        for i in range(probabilities.shape[2]):
            plot_matrix(
                fig_size,
                probabilities[:, :, i],
                list(range(probabilities.shape[1])),
                (
                    ["positive", "negative"]
                    if probabilities.shape[0] == 2
                    else ["positive", "indefinite", "negative"]
                ),
                f"Reasoning; class {class_labels[i]}",
                # ; sum {np.sum(probabilities[:, :, i], axis=1)}"
            )
            plt.savefig(f"{path}/reasonings/reasoning_{i}.pdf")  # plt.show()

        plt.close("all")

    def plot_random_samples(
        model: pl.LightningModule, path: str, sample_steps: int = 10
    ) -> None:
        try:
            os.makedirs(f"{path}/random_samples")
        except OSError:
            return

        if hasattr(model, "prototypes"):
            tangents = model.prototypes.weight.detach().cpu().numpy()
            translations = model.prototypes.bias.detach().cpu().numpy()
        else:
            tangents = model.components.weight.detach().cpu().numpy()
            translations = model.components.bias.detach().cpu().numpy()

        plt.figure()

        for i in range(tangents.shape[0]):
            translation = translations[i]
            tangent = tangents[i]
            for j in range(sample_steps):
                sample = translation + tangent @ np.random.randn(tangent.shape[1])
                sample = np.reshape(sample, (28, 28))
                plt.subplot(tangents.shape[0], sample_steps, i * sample_steps + j + 1)
                plt.imshow(sample, cmap="gray", vmin=0, vmax=1)
                plt.axis("off")
                plt.imsave(
                    f"{path}/random_samples/prototype_{i}_sample{j}.png",
                    sample,
                    vmin=0,
                    vmax=1,
                    cmap="gray",
                )

        plt.savefig(f"{path}/random_samples/overview.pdf")
        plt.close("all")

    def plot_tangents(model: pl.LightningModule, path: str) -> None:
        try:
            os.makedirs(f"{path}/tangents")
        except OSError:
            return

        if hasattr(model, "prototypes"):
            tangents = model.prototypes.weight.detach().cpu().numpy()
        else:
            tangents = model.components.weight.detach().cpu().numpy()

        tangents = np.reshape(tangents, (tangents.shape[0], 28, 28, tangents.shape[2]))

        plt.figure()

        for i, digit_tangents in enumerate(tangents):
            for j in range(digit_tangents.shape[2]):
                tangent = digit_tangents[:, :, j]
                plt.subplot(
                    tangents.shape[0], tangents.shape[3], i * subspace_dimension + j + 1
                )
                plt.imshow(tangent, cmap="gray")
                plt.axis("off")
                plt.imsave(
                    f"{path}/tangents/prototype_{i}_tangent{j}.png",
                    tangent,
                    cmap="gray",
                )

        plt.savefig(f"{path}/tangents/overview.pdf")
        plt.close("all")

    def train_test_routine(
        model: pl.LightningModule,
        epochs: int,
        steps: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        path: str,
    ) -> Tuple[pl.LightningModule, List[Mapping[str, float]]]:
        if torch.cuda.is_available():
            trainer = pl.Trainer(
                max_epochs=epochs,
                max_steps=steps,
                accelerator="gpu",
                devices=[device],
                logger=pl.pytorch.loggers.CSVLogger(path),
                callbacks=[
                    pl.pytorch.callbacks.ModelCheckpoint(
                        monitor="val_loss",
                        dirpath=path,
                        filename="{epoch}-{val_loss:.2f}",
                    ),
                    pl.pytorch.callbacks.ModelCheckpoint(monitor=None, dirpath=path),
                ],
                default_root_dir=path,
            )
        else:
            trainer = pl.Trainer(
                max_epochs=epochs,
                max_steps=steps,
                accelerator="cpu",
            )

        checkpoint = "epoch=39-step=18760.ckpt"
        if os.path.isfile(f"{path}/{checkpoint}"):
            print("Skip training and load model")
            model = model.__class__.load_from_checkpoint(
                f"{path}/{checkpoint}"
            )
        else:
            trainer.fit(
                model=model,
                train_dataloaders=train_loader,
                val_dataloaders=train_loader,
            )

        test_metric = trainer.test(model=model, dataloaders=test_loader)

        return model, test_metric

    # push images and labels to device only once
    images = (test_dataset.data.view(-1, 1, 28, 28) / 255).to(f"cuda:{device}")
    labels = test_dataset.targets.to(f"cuda:{device}")

    def auto_attack(
        model: pl.LightningModule,
        eps: float,
        path: str,
        squeeze_output: bool,
        similarity_output: bool,
    ) -> Tuple[np.ndarray, float]:
        os.mknod(f"{path}/autoattack.txt")

        model_ = model.to(f"cuda:{device}")

        def output_probs(x):
            y = model_(x.view(x.shape[0], -1))
            if squeeze_output:
                if similarity_output:
                    output_per_class, _ = torch.max(
                        model.reasoning_labels.unsqueeze(0) * y.unsqueeze(-1),
                        dim=1,
                    )
                else:
                    # minus because they assume maximization
                    output_per_class, _ = torch.min(
                        y.unsqueeze(-1)
                        + (1 - model.prototype_labels.unsqueeze(0))
                        * (torch.max(y) + 1),
                        dim=1,
                    )
                    output_per_class = -output_per_class
            else:
                output_per_class = y

            return output_per_class

        adversary = AutoAttack(
            output_probs,
            norm="L2",
            eps=eps,
            version="standard",
            log_path=f"{path}/autoattack.txt",
        )
        # DEBUG
        # adversary.attacks_to_run = ["apgd-ce", "apgd-t"]  # for debug speed up

        x_adv = (
            adversary.run_standard_evaluation(images, labels, bs=2048)
            .cpu()
            .detach()
            .numpy()
        )

        with open(f"{path}/autoattack.txt") as f:
            last_line: str = f.readlines()[-1]
        robust_acc = float(last_line.split(" ")[-1][:-2]) / 100

        return x_adv, robust_acc

    def certificate(model: pl.LightningModule, eps: float, method: str) -> float:
        count = 0
        batch = 500

        model_ = model.to(f"cuda:{device}")

        if method in ("robust_stable_cbc", "robust_stable_cbc_td"):
            margin = model_.margin
            model_.margin = None

            for i in range(images.shape[0] // batch):
                x = images[i * batch : (i + 1) * batch].view(batch, -1)
                y = labels[i * batch : (i + 1) * batch]

                predicted, loss, prob = model_.predict_step(x)
                for j in range(batch):
                    if torch.argmax(predicted[j]) == y[j] and loss[j] <= -eps:
                        count += 1

            model_.margin = margin

        elif method in ("gtlvq", "glvq"):
            for i in range(images.shape[0] // batch):
                x = images[i * batch : (i + 1) * batch].view(batch, -1)
                y = labels[i * batch : (i + 1) * batch]

                distances = model_(x)

                if squared:
                    distances = torch.sqrt(distances)

                predicted, _, _ = model_.predict_step(x)

                y_one_hot = torch.nn.functional.one_hot(
                    torch.tensor(y, dtype=torch.int64),
                    num_classes=number_classes,
                ).float()

                loss_func = MarginLoss(similarity=False, margin=2 * eps)
                loss = loss_func(distances, model_.prototype_labels, y_one_hot)

                for j in range(batch):
                    if torch.argmax(predicted[j]) == y[j] and loss[j] <= 0:
                        count += 1

        elif method in ("rbf_norm", "rbf_td_norm", "robust_rbf", "robust_rbf_td"):

            if method in ("robust_rbf", "robust_rbf_td"):
                margin = model_.margin
                model_.margin = None
            else:
                margin = None
                model_ = RobustRBF(
                    components=model_.components,
                    number_outputs=model_.number_outputs,
                    number_components=model_.number_components,
                    learning_rate=model_.learning_rate,
                    distance=model_.kernel.distance,
                    trainable_sigma=sigma_trainable,
                    sigma=initial_sigma,
                    margin=margin,
                    negative_loss_weight=negative_loss_weight,
                )
                model_.weights.data = model.weights.data.detach().clone()
                model_.kernel._sigma.data = model.kernel._sigma.data.detach().clone()
                model_ = model_.to(f"cuda:{device}")

            for i in range(images.shape[0] // batch):
                x = images[i * batch: (i + 1) * batch].view(batch, -1)
                y = labels[i * batch: (i + 1) * batch]

                predicted, loss, prob = model_.predict_step(x)
                for j in range(batch):
                    if predicted[j] == y[j] and loss[j] <= -eps:
                        count += 1

            model_.margin = margin

        elif method in ("stable_cbc", "stable_cbc_td"):
            for i in range(images.shape[0] // batch):
                x = images[i * batch : (i + 1) * batch].view(batch, -1)
                y = labels[i * batch : (i + 1) * batch]

                _, detection_probabilities = model_._shared_forward(
                    x, full_report=False
                )

                predicted, _, _ = model_.predict_step(x)
                distances = model_.detection_probability.distance(
                    x, model_.components(x)
                )

                y_one_hot = torch.nn.functional.one_hot(
                    torch.tensor(y, dtype=torch.int64),
                    num_classes=number_classes,
                ).float()

                loss = -robust_stable_cbc_loss(
                    data_template_comparisons=detection_probabilities,
                    template_labels=model_.reasoning_labels,
                    data_labels=y_one_hot,
                    reasoning_probabilities=model_.decoded_reasoning_probabilities,
                    component_probabilities=None,
                    requiredness_probabilities=None,
                    sigma=model_.detection_probability.sigma,
                    margin=None,
                )

                if not squared:
                    if method == "stable_cbc":
                        loss = -loss
                    else:
                        loss = -loss / 2
                else:
                    max_distances, _ = torch.max(distances, dim=1)
                    max_distances_ = torch.sqrt(max_distances)
                    loss_tmp = torch.where(
                        loss > 0,
                        -max_distances_ / 3
                        + torch.sqrt(max_distances / 9 + torch.abs(loss) / 3),
                        loss,
                    )
                    loss = -loss_tmp

                    if method == "stable_cbc":
                        loss = loss
                    else:
                        loss = loss / 2

                for j in range(batch):
                    if torch.argmax(predicted[j]) == y[j] and loss[j] <= -eps:
                        count += 1

        else:
            pass

        certified_acc = count / test_dataset.targets.shape[0]
        print(f"Certified accuracy: {certified_acc}")

        return certified_acc

    def append_test_metric(test_metrics: dict, test_metric: dict) -> None:
        test_metrics["test_acc"].append(test_metric[0]["test_acc"])
        test_metrics["test_loss"].append(test_metric[0]["test_loss"])

    def write_test_metrics(test_metrics: dict, path: str) -> None:
        test_metrics["test_acc_mean"] = np.mean(test_metrics["test_acc"])
        test_metrics["test_acc_std"] = np.std(test_metrics["test_acc"])
        test_metrics["test_loss_mean"] = np.mean(test_metrics["test_loss"])
        test_metrics["test_loss_std"] = np.std(test_metrics["test_loss"])

        for eps in eps_list:
            test_metrics["test_acc_mean"] = np.mean(test_metrics["test_acc"])
            test_metrics["test_acc_std"] = np.std(test_metrics["test_acc"])
            test_metrics["test_loss_mean"] = np.mean(test_metrics["test_loss"])
            test_metrics["test_loss_std"] = np.std(test_metrics["test_loss"])
            test_metrics[f"test_empirical_robust_acc_eps_{eps}_mean"] = np.mean(
                test_metrics[f"test_empirical_robust_acc_eps_{eps}"]
            )
            test_metrics[f"test_empirical_robust_acc_eps_{eps}_std"] = np.std(
                test_metrics[f"test_empirical_robust_acc_eps_{eps}"]
            )
            test_metrics[f"test_certified_robust_acc_eps_{eps}_mean"] = np.mean(
                test_metrics[f"test_certified_robust_acc_eps_{eps}"]
            )
            test_metrics[f"test_certified_robust_acc_eps_{eps}_std"] = np.std(
                test_metrics[f"test_certified_robust_acc_eps_{eps}"]
            )

        with open(f"{path}/results.json", "w") as f:
            json.dump(test_metrics, f, indent=4)

    base_dict = {
        "test_acc": [],
        "test_loss": [],
        "test_acc_mean": None,
        "test_acc_std": None,
        "test_loss_mean": None,
        "test_loss_std": None,
    }
    for eps in eps_list:
        base_dict.update(
            {
                f"test_empirical_robust_acc_eps_{eps}": [],
                f"test_certified_robust_acc_eps_{eps}": [],
                f"test_empirical_robust_acc_eps_{eps}_mean": None,
                f"test_empirical_robust_acc_eps_{eps}_std": None,
                f"test_certified_robust_acc_eps_{eps}_mean": None,
                f"test_certified_robust_acc_eps_{eps}_std": None,
            }
        )

    test_metrics = deepcopy(base_dict)

    int_reasoning_labels = [
        i for i in range(number_classes) for _ in range(number_reasoning_per_class)
    ]

    reasoning_labels = torch.nn.functional.one_hot(
        torch.tensor(int_reasoning_labels, dtype=torch.int64),
        num_classes=number_classes,
    ).float()

    for i in range(runs):
        torch.manual_seed(seeds[i])
        torch.cuda.manual_seed_all(seeds[i])
        np.random.seed(seeds[i])

        path = f"{root_path}/{method}_{robust_margin}/run_{i}"

        if method in (
            "cbc",
            "stable_cbc",
            "glvq",
            "rbf",
            "rbf_norm",
            "robust_stable_cbc",
            "robust_rbf",
        ):
            components = Components(
                init_weight=torch.rand((number_components, 28 * 28)) / 10 + 0.45,
                weight_constraint=clip,
            )
        else:
            components = AffineSubspaces(
                init_weight=torch.rand(
                    (number_components, 28 * 28, subspace_dimension)
                ),
                init_bias=torch.rand((number_components, 28 * 28)) / 10 + 0.45,
            )

        if method in (
            "stable_cbc",
            "stable_cbc_td",
            "cbc",
            "cbc_td",
            "robust_stable_cbc",
            "robust_stable_cbc_td",
        ):
            squeeze_output = True
            similarity_output = True

            if method in ("cbc", "cbc_td"):
                init_requiredness_probabilities = (
                    0.6
                    - torch.rand(
                        2,
                        number_components,
                        number_reasoning_per_class * number_classes,
                    )
                    * 0.2
                )
            else:
                init_requiredness_probabilities = (
                    0.6
                    - torch.rand(
                        number_components, number_reasoning_per_class * number_classes
                    )
                    * 0.2
                )

                init_component_probabilities = torch.softmax(torch.rand(
                    number_components, number_reasoning_per_class * number_classes
                ), dim=0)

            if method == "cbc":
                model = CBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    detection_probability=ExponentiallySquashedNegativeDistance(
                        distance=EuclideanDistance(squared=squared),
                        sigma=initial_sigma,
                        trainable_sigma=sigma_trainable,
                    ),
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    loss=MarginLoss(similarity=True, margin=margin),
                    learning_rate=5.0e-3,
                )

            elif method == "cbc_td":
                model = CBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    detection_probability=ExponentiallySquashedNegativeDistance(
                        distance=TangentDistance(squared=squared),
                        sigma=initial_sigma,
                        trainable_sigma=sigma_trainable,
                    ),
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    loss=MarginLoss(similarity=True, margin=margin),
                    learning_rate=5.0e-3,
                )

            elif method == "stable_cbc":
                model = StableCBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    detection_probability=ExponentiallySquashedNegativeDistance(
                        distance=EuclideanDistance(squared=squared),
                        sigma=initial_sigma,
                        trainable_sigma=sigma_trainable,
                    ),
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    init_component_probabilities=init_component_probabilities,
                    loss=MarginLoss(similarity=True, margin=margin),
                    learning_rate=5.0e-3,
                )

            elif method == "stable_cbc_td":
                model = StableCBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    detection_probability=ExponentiallySquashedNegativeDistance(
                        distance=TangentDistance(squared=squared),
                        sigma=initial_sigma,
                        trainable_sigma=sigma_trainable,
                    ),
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    init_component_probabilities=init_component_probabilities,
                    loss=MarginLoss(similarity=True, margin=margin),
                    learning_rate=5.0e-3,
                )

            elif method == "robust_stable_cbc":
                model = RobustStableCBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    distance=EuclideanDistance(squared=squared),
                    sigma=initial_sigma,
                    trainable_sigma=sigma_trainable,
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    init_component_probabilities=init_component_probabilities,
                    learning_rate=5.0e-3,
                    margin=robust_margin,
                    negative_loss_weight=negative_loss_weight,
                )

            elif method == "robust_stable_cbc_td":
                model = RobustStableCBC(
                    components=components,
                    reasoning_labels=reasoning_labels,
                    distance=TangentDistance(squared=squared),
                    sigma=initial_sigma,
                    trainable_sigma=sigma_trainable,
                    init_requiredness_probabilities=init_requiredness_probabilities,
                    init_component_probabilities=init_component_probabilities,
                    learning_rate=5.0e-3,
                    margin=robust_margin,
                    negative_loss_weight=negative_loss_weight,
                )

            else:
                raise ValueError(f"Don't know how to handle {method}.")

        elif method in ("glvq", "gtlvq"):
            squeeze_output = True
            similarity_output = False

            if method == "glvq":
                model = GLVQ(
                    prototypes=components,
                    prototype_labels=reasoning_labels,
                    learning_rate=5.0e-3,
                    distance=EuclideanDistance(squared=squared),
                )

            elif method == "gtlvq":
                model = GTLVQ(
                    prototypes=components,
                    prototype_labels=reasoning_labels,
                    squared=squared,
                )

            else:
                raise ValueError(f"Don't know how to handle {method}.")

        elif method in (
            "rbf",
            "rbf_norm",
            "rbf_td",
            "rbf_td_norm",
            "robust_rbf",
            "robust_rbf_td",
        ):
            squeeze_output = False
            similarity_output = True

            if method in ("rbf", "rbf_td"):
                layer_normalization = False
            else:
                layer_normalization = True

            if method in ("rbf", "rbf_norm"):
                model = RBF(
                    components=components,
                    number_outputs=10,
                    number_components=number_components,
                    learning_rate=5.0e-3,
                    kernel=ExponentiallySquashedNegativeDistance(
                        distance=EuclideanDistance(squared=squared),
                        trainable_sigma=sigma_trainable,
                        sigma=initial_sigma,
                    ),
                    layer_normalization=layer_normalization,
                )
            elif method in ("rbf_td", "rbf_td_norm"):
                model = RBF(
                    components=components,
                    number_outputs=10,
                    number_components=number_components,
                    learning_rate=5.0e-3,
                    kernel=ExponentiallySquashedNegativeDistance(
                        distance=TangentDistance(squared=squared),
                        sigma=initial_sigma,
                        trainable_sigma=sigma_trainable,
                    ),
                    layer_normalization=layer_normalization,
                )

            elif method == "robust_rbf":
                model = RobustRBF(
                    components=components,
                    number_outputs=10,
                    number_components=number_components,
                    learning_rate=5.0e-3,
                    distance=EuclideanDistance(squared=squared),
                    trainable_sigma=sigma_trainable,
                    sigma=initial_sigma,
                    margin=robust_margin,
                    negative_loss_weight=negative_loss_weight,
                )

            elif method == "robust_rbf_td":
                model = RobustRBF(
                    components=components,
                    number_outputs=10,
                    number_components=number_components,
                    learning_rate=5.0e-3,
                    distance=TangentDistance(squared=squared),
                    trainable_sigma=sigma_trainable,
                    sigma=initial_sigma,
                    margin=robust_margin,
                    negative_loss_weight=negative_loss_weight,
                )

            else:
                raise ValueError(f"Don't know how to handle {method}.")

        model, test_metric = train_test_routine(
            model=model,
            epochs=epochs,
            steps=steps,
            train_loader=train_loader,
            test_loader=test_loader,
            path=path,
        )

        append_test_metric(test_metrics, test_metric)

        plot_components(model=model, fig_size=(6, 11), path=path)

        if method in (
            "stable_cbc",
            "stable_cbc_td",
            "cbc",
            "cbc_td",
            "robust_stable_cbc",
            "robust_stable_cbc_td",
        ):
            plot_reasoning_matrices(
                model=model,
                path=path,
                fig_size=(6, 5),
                class_labels=int_reasoning_labels,
            )

        if method in (
            "stable_cbc_td",
            "cbc_td",
            "gtlvq",
            "rbf_td",
            "rbf_td_norm",
            "robust_stable_cbc_td",
            "robust_rbf_td",
        ):
            plot_random_samples(model=model, path=path)
            plot_tangents(model=model, path=path)

        exists = False
        for eps in eps_list:
            path_robustness = f"{path}/eps_{eps}"
            try:
                os.makedirs(path_robustness)
            except OSError:
                exists = True

            if not exists:
                x_adv, robust_acc = auto_attack(
                    model=model,
                    eps=eps,
                    path=path_robustness,
                    squeeze_output=squeeze_output,
                    similarity_output=similarity_output,
                )
            else:
                with open(f"{path_robustness}/autoattack.txt") as f:
                    last_line: str = f.readlines()[-1]
                robust_acc = float(last_line.split(" ")[-1][:-2]) / 100

            test_metrics[f"test_empirical_robust_acc_eps_{eps}"].append(robust_acc)

            if not exists:
                plot_adversaries(model=model, x_adv=x_adv, path=path_robustness)

            certified_acc = certificate(model=model, eps=eps, method=method)

            test_metrics[f"test_certified_robust_acc_eps_{eps}"].append(certified_acc)

    write_test_metrics(test_metrics, path=f"{root_path}/{method}_{robust_margin}")
