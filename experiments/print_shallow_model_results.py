"""Log fixing script with better formatting."""
import os
import json


# Specify the result log path for the given experiment run.
root_path = "model_comparison/robustness_squared_False"
# root_path = "model_comparison/robustness_squared_True"

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

# load data and plot results nicely for easy copy-paste to the table
with open(f"{root_path}/results.json", "r") as f:
    results = json.load(f)

for result in results.keys():
    print(
        f"\n **** {result} **** \n"
        f"acc: \t   {results[result]['test_acc_mean'] * 100:.1f} \pm {results[result]['test_acc_std'] * 100:.1f}\n"
        f"eps 0.5:   {results[result]['test_empirical_robust_acc_eps_0.5_mean'] * 100:.1f} \pm {results[result]['test_empirical_robust_acc_eps_0.5_std'] * 100:.1f}\n"
        f"cert 0.5:  {results[result]['test_certified_robust_acc_eps_0.5_mean'] * 100:.1f} \pm {results[result]['test_certified_robust_acc_eps_0.5_std'] * 100:.1f}\n"
        f"eps 1: \t   {results[result]['test_empirical_robust_acc_eps_1_mean'] * 100:.1f} \pm {results[result]['test_empirical_robust_acc_eps_1_std'] * 100:.1f}\n"
        f"cert 1:    {results[result]['test_certified_robust_acc_eps_1_mean'] * 100:.1f} \pm {results[result]['test_certified_robust_acc_eps_1_std'] * 100:.1f}\n"
        f"eps 1.58:  {results[result]['test_empirical_robust_acc_eps_1.58_mean'] * 100:.1f} \pm {results[result]['test_empirical_robust_acc_eps_1.58_std'] * 100:.1f}\n"
        f"cert 1.58: {results[result]['test_certified_robust_acc_eps_1.58_mean'] * 100:.1f} \pm {results[result]['test_certified_robust_acc_eps_1.58_std'] * 100:.1f}\n"
    )
