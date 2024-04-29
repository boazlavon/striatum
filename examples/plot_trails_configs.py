import json
import os
import sys
import matplotlib.pyplot as plt
import itertools
from datetime import datetime


def get_bandit_trials_kwargs(bandit, trials_config, hyper_params):
    bandit_trials = {}
    for hyper_param in hyper_params:
        hyper_param_config = [config for config in trials_config if config["name"] == hyper_param][0]
        if bandit in hyper_param_config["bandits"]:
            bandit_trials[hyper_param] = hyper_param_config["values"]
    print(bandit, bandit_trials)
    keys = bandit_trials.keys()
    values = bandit_trials.values()
    combinations = itertools.product(*values)
    # Create a list of dictionaries for each combination
    trials_kwargs = [dict(zip(keys, combination)) for combination in combinations]
    # trials_kwargs = [{'bandit': bandit, **trial_kwargs} for trial_kwargs in trials_kwargs]
    return trials_kwargs


def get_label(bandit, bandit_kwargs):
    label = bandit
    for key, value in sorted(bandit_kwargs.items()):
        label += f"_{key}={value}"
    return label


def get_result_path(bandit, bandit_kwargs, regrets_dir_path):
    label = get_label(bandit, bandit_kwargs)
    return os.path.join(regrets_dir_path, f"{label}.json")


def plot_trails(plot_trials_config_path, hyper_params_path):
    plot_trials_config = {}
    hyper_params = {}
    with open(plot_trials_config_path, "r") as f:
        plot_trials_config = json.load(f)
    with open(hyper_params_path, "r") as f1:
        hyper_params = json.load(f1)

    col = ["b", "g", "r", "c", "m", "y", "k", "w"]
    results_dir_path = os.path.join("results")
    regrets_dir_path = os.path.join(results_dir_path, "results_jsons")

    bandits = [config for config in plot_trials_config if config["name"] == "bandit"][0]["values"]
    results = []
    for bandit in bandits:
        bandit_trials_kwargs = get_bandit_trials_kwargs(bandit, plot_trials_config, hyper_params)
        for idx, bandit_kwargs in enumerate(bandit_trials_kwargs):
            result_json_path = get_result_path(bandit, bandit_kwargs, regrets_dir_path)
            if not os.path.exists(result_json_path):
                print(f"Skipping bandit {bandit} with kwargs {bandit_kwargs}")
                continue

            result = None
            with open(result_json_path, "r") as f:
                result = json.load(f)

            if result is not None:
                results.append(result)

    i = len(results)
    results = sorted(results, key=lambda x: x["regret"][-1])[::-1]
    for result in results:
        bandit, bandit_kwargs, bandit_regret = result["bandit"], result["kwargs"], result["regret"]
        label = get_label(bandit, bandit_kwargs)
        label = f"{bandit}_{i}"
        print(f"[{i}] Plotting bandit {bandit} (r={bandit_regret[-1]}) with kwargs {bandit_kwargs} ")
        plt.plot(range(len(bandit_regret)), bandit_regret, c=col[i % len(col)], ls="-", label=label)
        i -= 1

    # set plot
    plt.xlabel("time")
    plt.ylabel("regret")
    # plt.legend()
    plt.legend(loc="lower left", fontsize="small", ncol=2)
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.title("Regret Bound with respect to T")

    results_dir_path = os.path.join("results")
    plots_dir_path = os.path.join(results_dir_path, "plots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = str(timestamp)

    plot_filename = f"{timestamp}.png"
    plot_path = os.path.join(plots_dir_path, plot_filename)
    plt.savefig(plot_path)
    plt.show()
    plt.close()  # Close the figure context
    print(f"Saved plot as {plot_path}")


def main():
    plot_trials_config_path = sys.argv[1]
    hyper_params_path = sys.argv[2]
    plot_trails(plot_trials_config_path, hyper_params_path)


if __name__ == "__main__":
    main()
