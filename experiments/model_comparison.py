import os

from ray import tune


def objective(config):
    os.system(
        f"python {config['path']}/base_script.py --method {config['method']} "
        f"{'--squared' if config['squared'] else ''} "
        f"--base_path {config['path'] + '/model_comparison'} "
        f"> /dev/null"
    )
    return 100


def tune_models(path: str):
    search_space = {
        "method": tune.grid_search(
            [
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
            ]
        ),
        "squared": tune.grid_search([False, True]),
        "path": path,
    }

    tuner = tune.Tuner(
        tune.with_resources(objective, {"gpu": 1}),
        param_space=search_space,
    )

    results = tuner.fit()
    print(results.get_best_result(metric="score", mode="min", filter_nan_and_inf=False).config)


if __name__ == "__main__":
    path = os.getcwd()
    tune_models(path=path)
