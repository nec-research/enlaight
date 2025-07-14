"""Plotting accuracy plots against increasing attack strengths for multiple margins to
carry out robustness analysis."""
import os

from matplotlib import pyplot as plt
import json
import numpy as np

# Specify the result log path for the given experiment run.
root_path = "robustness_analysis/robustness_squared_False"
eps_list = [
    "0",
    "0.25",
    "0.5",
    "1.0",
    "1.25",
    "1.5",
    "1.75",
    "2.0",
    "2.5",
    "3.0",
    "3.5",
    "4.0",
    "4.5",
    "5.0",
]
if not os.path.exists(f"{root_path}/results.json"):
    methods = os.listdir(root_path)

    all_metrics = {}
    for method in methods:
        if method in (".ipynb_checkpoints",):
            continue

        with open(f"{root_path}/{method}/results.json") as f:
            test_metrics = json.load(f)

        all_metrics.update({method: test_metrics})

    with open(f"{root_path}/results.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

with open(f"{root_path}/results.json") as f:
    results = json.load(f)

for method in ["stable_cbc", "robust_stable_cbc"]:
    plt.figure(figsize=(10, 8))
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(results) // 2)))
    margins = []

    for key in sorted(results.keys()):
        if method != "_".join(key.split("_")[:-1]):
            continue

        margins.append(key.split("_")[-1])
        empirical_acc = []
        certified_acc = []
        for eps in eps_list:
            if eps == "0":
                empirical_acc.append(results[key][f"test_acc_mean"])
                certified_acc.append(results[key][f"test_acc_mean"])
            else:
                empirical_acc.append(
                    results[key][f"test_empirical_robust_acc_eps_{eps}_mean"]
                )
                certified_acc.append(
                    results[key][f"test_certified_robust_acc_eps_{eps}_mean"]
                )
        color = next(colors)
        plt.plot(eps_list, empirical_acc, color=color)
        plt.plot(eps_list, certified_acc, "--", color=color)
        plt.legend(
            [f"margin {margin}, {t}" for margin in margins for t in ["emp.", "cert."]]
        )
        plt.ylim([0, 1])

    plt.xlabel("Attack strength")
    plt.ylabel("Accuracy")
    if method == "stable_cbc":
        name = "CBC"
    else:
        name = "Robust CBC"
    plt.title(name)
    plt.savefig(f"{name}.pdf")
