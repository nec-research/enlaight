import os

from ray import tune


def objective(config):
    if config["method"] == "stable_cbc" and config["robust_margin"] != 0.1:
        return 1

    elif config["method"] == "stable_cbc" and config["robust_margin"] == 0.1:
        # lazy solution to have the correct file names
        config["robust_margin"] = config["margin"]

    elif config["method"] == "robust_stable_cbc" and config["margin"] != 0.1:
        return 1

    os.system(
        f"python {config['path']}/base_script.py --method {config['method']} "
        f"--base_path {config['path'] + '/robustness_analysis'} "
        f"--robust_margin {config['robust_margin']} "
        f"--margin {config['margin']} --eps_list {config['eps_list']} > /dev/null"
    )
    return 0


def tune_models(path: str):
    # lazy implementation of the search space. robust_margin should only be active for
    # robust_stable_cbc and margin for stable_cbc. We filter in the objective and
    # return a dummy value
    search_space = {
        "method": tune.grid_search(["stable_cbc", "robust_stable_cbc"]),
        "eps_list": "0.25,0.5,1.0,1.25,1.5,1.75,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
        "robust_margin": tune.grid_search(
            [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
        ),
        "margin": tune.grid_search(
            [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
        ),
        "path": path,
    }

    tuner = tune.Tuner(
        tune.with_resources(objective, {"gpu": 1}),
        param_space=search_space,
    )

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min").config)


if __name__ == "__main__":
    path = os.getcwd()
    tune_models(path=path)
