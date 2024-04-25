""" EXP4.P: An extention to exponential-weight algorithm for exploration and
exploitation. This module contains a class that implements EXP4.P, a contextual
bandit algorithm with expert advice.
"""

import logging

import six
from six.moves import zip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..storage import Recommendation


from striatum.bandit.bandit import BaseBandit

LOGGER = logging.getLogger(__name__)


class NeuralNetwork(nn.Module):
    def __init__(self, layers_dims, dropout_prob=0.5):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to properly register the layers
        for idx, (in_layer_size, out_layer_size) in enumerate(zip(layers_dims[:-1], layers_dims[1:])):
            self.layers.append(nn.Linear(in_layer_size, out_layer_size))
            if idx < len(layers_dims) - 2:
                self.layers.append(nn.Dropout(dropout_prob))
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=0)
        return x


class Exp4PNN(BaseBandit):
    r"""Exp4.P with pre-trained supervised learning algorithm.

    Parameters
    ----------
    actions : list of Action objects
        List of actions to be chosen from.

    historystorage: a HistoryStorage object
        The place where we store the histories of contexts and rewards.

    modelstorage: a ModelStorage object
        The place where we store the model parameters.

    delta: float, 0 < delta <= 1
        With probability 1 - delta, LinThompSamp satisfies the theoretical
        regret bound.

    p_min: float, 0 < p_min < 1/k
        The minimum probability to choose each action.

    References
    ----------
    .. [1]  Beygelzimer, Alina, et al. "Contextual bandit algorithms with
            supervised learning guarantees." International Conference on
            Artificial Intelligence and Statistics (AISTATS). 2011u.
    """

    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_size1=64,
        hidden_size2=128,
        dropout_prob=0.5,
        lr=1e-3
    ):
        super(Exp4PNN, self).__init__(historystorage, modelstorage, actions)
        self.n_total = 0
        # number of actions (i.e. K in the paper)
        self.n_actions = self._action_storage.count()
        self.action_ids = list(self._action_storage.iterids())
        self.action_ids.sort()
        self.max_rounds = max_rounds
        self.num_advisors = num_advisors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = NeuralNetwork(
            [self.num_advisors, hidden_size1, hidden_size2, self.num_advisors], dropout_prob
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        # delta > 0
        if not isinstance(delta, float):
            raise ValueError("delta should be float, the one" "given is: %f" % p_min)
        self.delta = delta

        # p_min in [0, 1/k]
        if p_min is None:
            self.p_min = np.sqrt(np.log(10) / self.n_actions / self.max_rounds)
        elif not isinstance(p_min, float):
            raise ValueError("p_min should be float, the one" "given is: %f" % p_min)
        elif (p_min < 0) or (p_min > (1.0 / self.n_actions)):
            raise ValueError("p_min should be in [0, 1/k], the one" "given is: %f" % p_min)
        else:
            self.p_min = p_min

        # Initialize the model storage

        model = {
            # probability distribution for action recommendation
            "action_probs": None,
            # weight vector for each expert
            "w": None,
        }
        self._model_storage.save_model(model)

    def _calculate_action_probs(self, context, w):
        # Convert context to a structured numpy array for vectorized operations
        advisor_ids = torch.tensor(list(context.keys()), dtype=torch.long)
        n_advisors = len(advisor_ids)
        n_actions = len(self.action_ids)

        # Fill the context matrix and reward vector
        context_matrix = torch.zeros((n_advisors, n_actions), dtype=torch.float)
        action_index = {aid: idx for idx, aid in enumerate(self.action_ids)}
        for advisor_idx, actions in context.items():
            for action, value in actions.items():
                context_matrix[advisor_idx, action_index[action]] = value

        # Calculate weighted probabilities for actions
        w_sum = torch.sum(w)
        weighted_sums = torch.matmul(w, context_matrix)
        action_probs = (1 - self.n_actions * self.p_min) * (weighted_sums / w_sum) + self.p_min
        action_probs /= torch.sum(action_probs)

        return action_probs

    def _Exp4PNN_score(self, context):
        # Convert context to a structured numpy array for vectorized operations
        advisor_ids = torch.tensor(list(context.keys()), dtype=torch.long)
        n_advisors = len(advisor_ids)

        # Get or initialize weights
        w = self._model_storage.get_model().get("w", torch.ones(n_advisors))
        if w is None:
            w = torch.ones(n_advisors)

        action_probs = self._calculate_action_probs(context, w)

        # Prepare to save and return the results
        estimated_reward = {}
        uncertainty = {}
        score = {}
        for action_id, action_prob in zip(self.action_ids, action_probs.tolist()):
            estimated_reward[action_id] = action_prob
            uncertainty[action_id] = 0
            score[action_id] = action_prob
        self._model_storage.save_model({"action_probs": action_probs, "w": w})

        return estimated_reward, uncertainty, score

    def get_action(self, context=None, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : dictionary
            Contexts {expert_id: {action_id: expert_prediction}} of
            different actions.

        n_actions: int
            Number of actions wanted to recommend users.

        Returns
        -------
        history_id : int
            The history id of the action.

        action_recommendation : list of dictionaries
            In each dictionary, it will contains {Action object,
            estimated_reward, uncertainty}.
        """
        estimated_reward, uncertainty, score = self._Exp4PNN_score(context)

        action_recommendation = []
        action_recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]

        for action_id in action_recommendation_ids:
            action = self.get_action_with_id(action_id)
            action_recommendation.append(
                Recommendation(
                action=action,
                estimated_reward=estimated_reward[action_id],
                uncertainty=uncertainty[action_id],
                score=score[action_id]
            ))

        self.n_total += 1
        history_id = self._history_storage.add_history(context, action_recommendation, rewards=None)
        return history_id, action_recommendation

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        # print(rewards)
        # Retrieve the context for the given history_id
        context = self._history_storage.get_unrewarded_history(history_id).context
        model = self._model_storage.get_model()
        w = model["w"]
        action_probs = model["action_probs"]

        # Prepare arrays from the context and rewards
        n_actions = len(self.action_ids)
        n_advisors = len(context)
        context_matrix = torch.zeros((n_advisors, n_actions), dtype=torch.float)
        action_index = {aid: idx for idx, aid in enumerate(self.action_ids)}

        # Fill the context matrix and reward vector
        for advisor_idx, actions in context.items():
            for action, value in actions.items():
                context_matrix[advisor_idx, action_index[action]] = value

        # Calculate y_hat and v_hat using vectorized operations
        n_advisors = torch.tensor(n_advisors, dtype=torch.float)
        for action_id, reward in six.viewitems(rewards):
            y_hat = (context_matrix[:, action_index[action_id]] * reward) / action_probs[
                action_index[action_id]
            ]
            v_hat = torch.sum(context_matrix / action_probs, dim=1)

            # Update weights
            updates = (
                self.p_min
                / 2
                * (
                    y_hat
                    + v_hat * torch.sqrt(torch.log(n_advisors / self.delta) / (n_actions * self.max_rounds))
                )
            )
            self.optimizer.zero_grad()
            updates = self.model(updates)
            w = w * updates
            w = w / torch.sum(w)
            grad_action_probs = self._calculate_action_probs(context, w)
            p = grad_action_probs[action_index[action_id]]
            loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
            print(f"{w=}")
            print(f"{loss=}")
            loss.backward()
            self.optimizer.step()
            w = w.detach()

        # self._history_storage.add_reward(history_id, rewards)
        self._model_storage.save_model({"action_probs": action_probs, "w": w})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)
