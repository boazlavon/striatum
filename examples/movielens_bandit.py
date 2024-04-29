# -*- coding: utf-8 -*-
"""
==============================
Contextual bandit on MovieLens
==============================

The script uses real-world data to conduct contextual bandit experiments. Here we use
MovieLens 10M Dataset, which is released by GroupLens at 1/2009. Please fist pre-process
datasets (use "movielens_preprocess.py"), and then you can run this example.
"""

import random
import torch
import json
import sys
from tqdm import tqdm
import time


import pandas as pd
import numpy as np
import os
from os.path import join

from striatum.storage import history
from striatum.storage import model
from striatum.bandit import ucb1
from striatum.bandit import linucb
from striatum.bandit import linthompsamp
from striatum.bandit import exp4p
from striatum.bandit import exp4pnn
from striatum.bandit import exp3
from striatum.bandit import exp3nn
from striatum.storage.action import Action, MemoryActionStorage
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from movielens_preprocess import PROCESSED_DATA_DIR
from plot_trails_configs import get_bandit_trials_kwargs, get_result_path


def set_seed(seed):
    # Set seed for Python's built-in random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure that PyTorch uses deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data():
    streaming_batch = pd.read_csv(
        join(PROCESSED_DATA_DIR, "streaming_batch.csv"), sep="\t", names=["user_id"], engine="c"
    )
    user_feature = pd.read_csv(
        join(PROCESSED_DATA_DIR, "user_feature.csv"), sep="\t", header=0, index_col=0, engine="c"
    )
    actions_id = list(
        pd.read_csv(join(PROCESSED_DATA_DIR, "actions.csv"), sep="\t", header=0, engine="c")["movie_id"]
    )
    reward_list = pd.read_csv(join(PROCESSED_DATA_DIR, "reward_list.csv"), sep="\t", header=0, engine="c")
    action_context = pd.read_csv(
        join(PROCESSED_DATA_DIR, "action_context.csv"), sep="\t", header=0, engine="c"
    )

    actions = []
    for key in actions_id:
        action = Action(key)
        actions.append(action)
    return streaming_batch, user_feature, actions, reward_list, action_context


def train_expert(action_context):
    # Define the models
    rf = OneVsRestClassifier(RandomForestClassifier())
    gbc = OneVsRestClassifier(GradientBoostingClassifier())
    adb = OneVsRestClassifier(AdaBoostClassifier(algorithm="SAMME"))
    logreg = OneVsRestClassifier(LogisticRegression())
    mlp = OneVsRestClassifier(MLPClassifier(max_iter=1000))
    mnb = OneVsRestClassifier(MultinomialNB())
    svc = OneVsRestClassifier(SVC(probability=True))
    knn = OneVsRestClassifier(KNeighborsClassifier())
    et = OneVsRestClassifier(ExtraTreesClassifier())
    models = [rf, gbc, adb, logreg, mlp, mnb, svc, knn, et]

    # Fit the models
    for model in models:
        model.fit(action_context.iloc[:, 2:], action_context.iloc[:, 0])

    # Return a list of trained models
    return models


def get_advice(context, actions_id, experts):
    advice = {}
    for time in context.keys():
        advice[time] = {}
        for i in range(len(experts)):
            prob = experts[i].predict_proba(context[time])[0]
            advice[time][i] = {}
            for j in range(len(prob)):
                advice[time][i][actions_id[j]] = prob[j]
    return advice


def policy_generation(bandit, actions, max_rounds, kwargs={}):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    if bandit == "Exp4P":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4p.Exp4P(
            actions_storage, historystorage, modelstorage, p_min=None, max_rounds=max_rounds, **kwargs
        )

    if bandit == "Exp4PNN":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4pnn.Exp4PNN(
            actions_storage, historystorage, modelstorage, p_min=None, max_rounds=max_rounds, **kwargs
        )

    if bandit == "Exp4PInnerNNUpdate":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4pnn.Exp4PInnerNNUpdate(
            actions_storage, historystorage, modelstorage, p_min=None, max_rounds=max_rounds, **kwargs
        )

    if bandit == "Exp4POuterNNUpdate":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4pnn.Exp4POuterNNUpdate(
            actions_storage, historystorage, modelstorage, p_min=None, max_rounds=max_rounds, **kwargs
        )

    if bandit == "OuterNNUpdateExperts":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4pnn.OuterNNUpdateExperts(
            actions_storage, historystorage, modelstorage, p_min=None, max_rounds=max_rounds, **kwargs
        )

    elif bandit == "LinUCB":
        policy = linucb.LinUCB(actions, historystorage, modelstorage, 0.3, 20)

    elif bandit == "LinThompSamp":
        policy = linthompsamp.LinThompSamp(
            actions, historystorage, modelstorage, d=20, delta=0.61, r=0.01, epsilon=0.71
        )

    elif bandit == "UCB1":
        policy = ucb1.UCB1(actions, historystorage, modelstorage)

    elif bandit == "Exp3":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3.Exp3(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == "Exp3NN":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3nn.Exp3NN(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == "Exp3OuterNNUpdate":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3nn.Exp3OuterNNUpdate(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == "OuterNNUpdate":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3nn.OuterNNUpdate(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == "NeuralAgent":
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3nn.NeuralAgent(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == "random":
        policy = 0

    return policy


def policy_evaluation(
    policy,
    bandit,
    streaming_batch,
    user_feature,
    reward_list,
    actions,
    action_context=None,
    num_advisors=None,
):
    times = len(streaming_batch)
    seq_error = np.zeros(shape=(times, 1))
    model_storage_over_time = []
    actions_id = [actions[i].action_id for i in range(len(actions))]
    print()
    pbar = tqdm(total=times)
    if bandit in [
        "LinUCB",
        "LinThompSamp",
        "UCB1",
        "Exp3",
        "Exp3NN",
        "Exp3OuterNNUpdate",
        "OuterNNUpdate",
        "NeuralAgent",
    ]:
        for t in range(1, times):
            feature = user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])]
            full_context = {}
            for action_id in actions_id:
                full_context[action_id] = feature
            history_id, action = policy.get_action(full_context, 1)
            watched_list = reward_list[reward_list["user_id"] == int(streaming_batch.iloc[t, 0])]

            if action[0].action.action_id not in list(watched_list["movie_id"]):
                policy.reward(history_id, {action[0].action.action_id: 0.0})
                if t == 1:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.action_id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

            result_dict = []
            model_storage_dict = policy._model_storage.get_model()
            if model_storage_dict is not None:
                result_dict = model_storage_dict.copy()
                for key, value in model_storage_dict.items():
                    try:
                        result_dict[key] = value.tolist()
                    except:
                        pass
            else:
                print("Warning: model storage is empty")
            model_storage_over_time.append(result_dict)
            # print(f"regret={seq_error[t] / (1.0 * t)} | t={t}/{times}")
            pbar.update(1)  # Manually update the progress bar by 1

    elif bandit in ["Exp4P", "Exp4PNN", "Exp4PInnerNNUpdate", "Exp4POuterNNUpdate", "OuterNNUpdateExperts"]:
        experts = train_expert(action_context)
        assert (
            num_advisors is not None and len(experts) > -num_advisors
        ), "Number of advisors must be specified and less than the number of experts"
        experts = experts[:num_advisors]
        for t in range(1, times):
            feature = user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])]
            if not len(feature):
                print(f"No feature for user {int(streaming_batch.iloc[t, 0])}")

            advice = {}
            for i in range(len(experts)):
                prob = experts[i].predict_proba(feature)[0]
                advice[i] = {}
                for j in range(len(prob)):
                    advice[i][actions_id[j]] = prob[j]
            history_id, action = policy.get_action(advice)
            watched_list = reward_list[reward_list["user_id"] == int(streaming_batch.iloc[t, 0])]
            if len(watched_list) == 0:
                print(f"No watched list for user {int(streaming_batch.iloc[t, 0])}")

            if action[0].action.action_id not in list(watched_list["movie_id"]):
                policy.reward(history_id, {action[0].action.action_id: 0.0})
                if t == 1:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.action_id: 1.0})
                if t > 1:
                    seq_error[t] = seq_error[t - 1]
            # print(f"regret={seq_error[t] / (1.0 * t)} | t={t}/{times}")
            result_dict = []
            model_storage_dict = policy._model_storage.get_model()
            if model_storage_dict is not None:
                result_dict = model_storage_dict.copy()
                for key, value in model_storage_dict.items():
                    try:
                        result_dict[key] = value.tolist()
                    except:
                        pass
            else:
                print("Warning: model storage is empty")
            model_storage_over_time.append(result_dict)
            pbar.update(1)  # Manually update the progress bar by 1

    elif bandit == "random":
        for t in range(times):
            action = actions_id[np.random.randint(0, len(actions) - 1)]
            watched_list = reward_list[reward_list["user_id"] == streaming_batch.iloc[t, 0]]

            if action not in list(watched_list["movie_id"]):
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
    pbar.close()  # Close the progress bar after the loop is done
    return seq_error, model_storage_over_time


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def run_single_trial(
    bandit,
    bandit_kwargs,
    actions,
    streaming_batch_small,
    user_feature,
    reward_list,
    action_context,
    regrets_dir_path,
    max_rounds,
):
    set_seed(bandit_kwargs["seed"])
    policy_bandit_kwargs = bandit_kwargs.copy()
    del policy_bandit_kwargs["seed"]

    num_advisors = None
    policy = policy_generation(bandit, actions, max_rounds, policy_bandit_kwargs)
    if (
        bandit in ["Exp4P", "Exp4PNN", "Exp4PInnerNNUpdate", "Exp4POuterNNUpdate", "OuterNNUpdateExperts"]
        and "num_advisors" in policy_bandit_kwargs
    ):
        num_advisors = policy_bandit_kwargs["num_advisors"]

    seq_error, model_storage_over_time = policy_evaluation(
        policy,
        bandit,
        streaming_batch_small,
        user_feature,
        reward_list,
        actions,
        action_context,
        num_advisors,
    )
    bandit_regret = regret_calculation(seq_error)
    bandit_regret = [val[0] for val in bandit_regret]

    result = {
        "bandit": bandit,
        "kwargs": bandit_kwargs,
        "regret": bandit_regret,
        "model_storage": model_storage_over_time,
    }
    result_json_path = get_result_path(bandit, bandit_kwargs, regrets_dir_path)
    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved results to {result_json_path} (r={bandit_regret[-1]})")


def main():
    streaming_batch, user_feature, actions, reward_list, action_context = get_data()

    trials_config_path = sys.argv[1]
    hyper_params_path = sys.argv[2]

    trials_config = {}
    hyper_params = {}
    with open(trials_config_path, "r") as f:
        trials_config = json.load(f)
    with open(hyper_params_path, "r") as f1:
        hyper_params = json.load(f1)

    results_dir_path = os.path.join("results")
    regrets_dir_path = os.path.join(results_dir_path, "results_jsons")
    os.makedirs(regrets_dir_path, exist_ok=True)

    bandits = [config for config in trials_config if config["name"] == "bandit"][0]["values"]
    max_rounds = [config for config in trials_config if config["name"] == "max_rounds"][0]["values"]
    assert len(max_rounds) == 1, "Only one max_rounds value is supported"
    max_rounds = max_rounds[0]
    streaming_batch_small = streaming_batch.iloc[0:max_rounds]

    print(f"Running bandits: {bandits} for {max_rounds} rounds")
    for bandit in bandits:
        bandit_trials_kwargs = get_bandit_trials_kwargs(bandit, trials_config, hyper_params)
        for idx, bandit_kwargs in enumerate(bandit_trials_kwargs):
            result_json_path = get_result_path(bandit, bandit_kwargs, regrets_dir_path)
            if os.path.exists(result_json_path):
                print(
                    f"[{idx + 1}/{len(bandit_trials_kwargs)}] Skipping bandit {bandit} with kwargs {bandit_kwargs}"
                )
                continue

            print(
                f"[{idx + 1}/{len(bandit_trials_kwargs)}] Running bandit {bandit} with kwargs {bandit_kwargs}"
            )
            run_single_trial(
                bandit,
                bandit_kwargs,
                actions,
                streaming_batch_small,
                user_feature,
                reward_list,
                action_context,
                regrets_dir_path,
                max_rounds,
            )


if __name__ == "__main__":
    set_seed(42)
    main()
