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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from datetime import datetime

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
from movielens_preprocess import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from collections import OrderedDict

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
    streaming_batch = pd.read_csv(join(PROCESSED_DATA_DIR, 'streaming_batch.csv'), sep='\t', names=['user_id'], engine='c')
    user_feature = pd.read_csv(join(PROCESSED_DATA_DIR, 'user_feature.csv'), sep='\t', header=0, index_col=0, engine='c')
    actions_id = list(pd.read_csv(join(PROCESSED_DATA_DIR, 'actions.csv'), sep='\t', header=0, engine='c')['movie_id'])
    reward_list = pd.read_csv(join(PROCESSED_DATA_DIR, 'reward_list.csv'), sep='\t', header=0, engine='c')
    action_context = pd.read_csv(join(PROCESSED_DATA_DIR, 'action_context.csv'), sep='\t', header=0, engine='c')

    actions = []
    for key in actions_id:
        action = Action(key)
        actions.append(action)
    return streaming_batch, user_feature, actions, reward_list, action_context


def train_expert(action_context):
    logreg = OneVsRestClassifier(LogisticRegression())
    mnb = OneVsRestClassifier(MultinomialNB(), )
    logreg.fit(action_context.iloc[:, 2:], action_context.iloc[:, 0])
    mnb.fit(action_context.iloc[:, 2:], action_context.iloc[:, 0])
    return [logreg, mnb]


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


def policy_generation(bandit, actions, kwargs={}):
    historystorage = history.MemoryHistoryStorage()
    modelstorage = model.MemoryModelStorage()
    max_rounds = 10000
    if bandit == 'Exp4P':
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4p.Exp4P(actions_storage, historystorage, modelstorage, delta=0.5, p_min=None, max_rounds=max_rounds)

    if bandit == 'Exp4PNN':
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp4pnn.Exp4PNN(actions_storage, historystorage, modelstorage, delta=0.5, p_min=None, max_rounds=max_rounds)

    elif bandit == 'LinUCB':
        policy = linucb.LinUCB(actions, historystorage, modelstorage, 0.3, 20)

    elif bandit == 'LinThompSamp':
        policy = linthompsamp.LinThompSamp(actions, historystorage, modelstorage,
                                           d=20, delta=0.61, r=0.01, epsilon=0.71)

    elif bandit == 'UCB1':
        policy = ucb1.UCB1(actions, historystorage, modelstorage)

    elif bandit == 'Exp3':
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3.Exp3(historystorage, modelstorage, actions_storage, gamma=0.3)

    elif bandit == 'Exp3NN':
        actions_storage = MemoryActionStorage()
        actions_storage.add(actions)
        policy = exp3nn.Exp3NN(historystorage, modelstorage, actions_storage, **kwargs)

    elif bandit == 'random':
        policy = 0

    return policy


def policy_evaluation(policy, bandit, streaming_batch, user_feature, reward_list, actions, action_context=None):
    times = len(streaming_batch)
    seq_error = np.zeros(shape=(times, 1))
    actions_id = [actions[i].action_id for i in range(len(actions))]
    print()
    if bandit in ['LinUCB', 'LinThompSamp', 'UCB1', 'Exp3', 'Exp3NN', 'NN', 'NNP']:
        for t in range(1, times):
            feature = user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])]
            full_context = {}
            for action_id in actions_id:
                full_context[action_id] = feature
            history_id, action = policy.get_action(full_context, 1)
            watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t, 0])]

            if action[0].action.action_id not in list(watched_list['movie_id']):
                policy.reward(history_id, {action[0].action.action_id: 0.0})
                if t == 1:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.action_id: 1.0})
                if t > 0:
                    seq_error[t] = seq_error[t - 1]
            print(f'regret={seq_error[t] / (1.0 * t)} | t={t}/{times}')

    elif bandit == 'Exp4P' or bandit == 'Exp4PNN':
        for t in range(1, times):
            feature = user_feature[user_feature.index == int(streaming_batch.iloc[t, 0])]
            if not len(feature):
                print(f'No feature for user {int(streaming_batch.iloc[t, 0])}')
            experts = train_expert(action_context)
            advice = {}
            for i in range(len(experts)):
                prob = experts[i].predict_proba(feature)[0]
                advice[i] = {}
                for j in range(len(prob)):
                    advice[i][actions_id[j]] = prob[j]
            history_id, action = policy.get_action(advice)
            watched_list = reward_list[reward_list['user_id'] == int(streaming_batch.iloc[t, 0])]
            if len(watched_list) == 0:
                print(f'No watched list for user {int(streaming_batch.iloc[t, 0])}')

            if action[0].action.action_id not in list(watched_list['movie_id']):
                policy.reward(history_id, {action[0].action.action_id: 0.0})
                if t == 1:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                policy.reward(history_id, {action[0].action.action_id: 1.0})
                if t > 1:
                    seq_error[t] = seq_error[t - 1]
            print(f'regret={seq_error[t] / (1.0 * t)} | t={t}/{times}')

    elif bandit == 'random':
        for t in range(times):
            action = actions_id[np.random.randint(0, len(actions)-1)]
            watched_list = reward_list[reward_list['user_id'] == streaming_batch.iloc[t, 0]]

            if action not in list(watched_list['movie_id']):
                if t == 0:
                    seq_error[t] = 1.0
                else:
                    seq_error[t] = seq_error[t - 1] + 1.0

            else:
                if t > 0:
                    seq_error[t] = seq_error[t - 1]

    return seq_error


def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret


def main():
    streaming_batch, user_feature, actions, reward_list, action_context = get_data()
    streaming_batch_small = streaming_batch.iloc[0:10000]

    #conduct regret analyses
    #experiment_bandit = ['LinUCB', 'LinThompSamp', 'Exp4P', 'UCB1', 'Exp3', 'random']
    experiment_bandit = []
    experiment_bandit_kwargs = []

    #exp3nn_kwargs = [{'use_exp3': True}, {'use_exp3': True, 'use_nn_update': True}, {'use_exp3': False, 'use_nn_update': True}, {'use_exp3': False, 'use_nn_probs': True}]
    exp3nn_kwargs = [{'use_exp3': True, 'use_nn_update': True}, {'use_exp3': False, 'use_nn_update': True}]
    for kwargs in exp3nn_kwargs:
        experiment_bandit.append('Exp3NN')
        experiment_bandit_kwargs.append(kwargs)

    no_kwargs_bandits = ['Exp3', 'Exp4P', 'Exp4PNN']
    for bandit in no_kwargs_bandits:
        experiment_bandit.append(bandit)
        experiment_bandit_kwargs.append({})

    regret = {}
    col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for idx, (bandit, bandit_kwargs) in enumerate(zip(experiment_bandit, experiment_bandit_kwargs)):
        set_seed(42)
        print(f"Running bandit {bandit} with kwargs {bandit_kwargs}")
        try:
            policy = policy_generation(bandit, actions, bandit_kwargs)
            seq_error = policy_evaluation(policy, bandit, streaming_batch_small, user_feature, reward_list,
                                        actions, action_context)
            label=bandit
            for key, value in bandit_kwargs.items():
                label += f"_{key}={value}"
            regret[label] = regret_calculation(seq_error)
            plt.plot(range(len(regret[label])), regret[label], c=col[idx], ls='-', label=label)
            plt.xlabel('time')
            plt.ylabel('regret')
            plt.legend()
            axes = plt.gca()
            axes.set_ylim([0, 1])
            plt.title("Regret Bound with respect to T")
        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()
            print(e)
    #plt.show()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamp = str(timestamp)
    figure_name = 'movielens'
    filename = f"plots/{figure_name}_{timestamp}.png"
    plt.savefig(filename)
    plt.close()  # Close the figure context
    print(f"Saved plot as {filename}")

    json_regret = { bandit : [val[0] for val in values] for bandit, values in regret.items() }
    print(json_regret)
    regret_json_filename = f"plots/{figure_name}_{timestamp}_regret.json"
    with open(regret_json_filename, 'w') as f:
        json.dump(json_regret, f, indent=4)
    print(f"Saved regret data as {regret_json_filename}")


if __name__ == '__main__':
    set_seed(42)
    main()
