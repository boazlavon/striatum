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
    def __init__(self, layers_dims, dropout_prob=0.5, init_p_factor_raw_weight=0):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to properly register the layers
        for idx, (in_layer_size, out_layer_size) in enumerate(zip(layers_dims[:-1], layers_dims[1:])):
            self.layers.append(nn.Linear(in_layer_size, out_layer_size))
            if idx < len(layers_dims) - 2:
                self.layers.append(nn.Dropout(dropout_prob))
                self.layers.append(nn.ReLU())

        # Add a learnable residual weight for the final layer
        self.p_factor_raw_weight = nn.Parameter(
            torch.tensor(init_p_factor_raw_weight, dtype=torch.float64)
        )  # Use only the residual first
        self.p_factor = 0.5 * (torch.tanh(self.p_factor_raw_weight) + 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=0)

        self.p_factor = 0.5 * (torch.tanh(self.p_factor_raw_weight) + 1)
        x = x * self.p_factor
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
        use_nn_inner_update,
        use_nn_outer_update,
        use_exp4p_update,
        use_expert=None,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
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
        # print(self.device)

        assert hidden_sizes is not None, "hidden_sizes should not be None"
        layers = [self.num_advisors] + hidden_sizes + [self.num_advisors]
        self.use_nn_inner_update = use_nn_inner_update
        self.inner_nn_update_model = NeuralNetwork(layers, dropout_prob).to(self.device)
        self.inner_nn_update_optimizer = optim.AdamW(self.inner_nn_update_model.parameters(), lr=lr)

        self.use_nn_outer_update = use_nn_outer_update
        self.outer_nn_update_model = NeuralNetwork(layers, dropout_prob).to(self.device)
        self.outer_nn_update_optimizer = optim.AdamW(self.outer_nn_update_model.parameters(), lr=lr)

        self.use_exp4p_update = use_exp4p_update
        self.use_expert = use_expert

        #print("use_exp4p_update", self.use_exp4p_update)
        #print("use_nn_inner_update", self.use_nn_inner_update)
        #print("use_nn_outer_update", self.use_nn_outer_update)

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
        advisor_ids = torch.tensor(list(context.keys()), dtype=torch.long).to(self.device)
        n_advisors = len(advisor_ids)
        n_actions = len(self.action_ids)

        # Fill the context matrix and reward vector
        context_matrix = torch.zeros((n_advisors, n_actions), dtype=torch.float).to(self.device)
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

    def _exp4pnn_score(self, context):
        # Convert context to a structured numpy array for vectorized operations
        advisor_ids = torch.tensor(list(context.keys()), dtype=torch.long).to(self.device)
        n_advisors = len(advisor_ids)

        # Get or initialize weights
        w = self._model_storage.get_model().get("w", torch.ones(n_advisors).to(self.device))
        if w is None:
            w = torch.ones(n_advisors).to(self.device)

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
        estimated_reward, uncertainty, score = self._exp4pnn_score(context)

        action_recommendation = []
        action_recommendation_ids = sorted(score, key=score.get, reverse=True)[:n_actions]

        for action_id in action_recommendation_ids:
            action = self.get_action_with_id(action_id)
            action_recommendation.append(
                Recommendation(
                    action=action,
                    estimated_reward=estimated_reward[action_id],
                    uncertainty=uncertainty[action_id],
                    score=score[action_id],
                )
            )

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
        context_matrix = torch.zeros((n_advisors, n_actions), dtype=torch.float).to(self.device)
        action_index = {aid: idx for idx, aid in enumerate(self.action_ids)}

        # Fill the context matrix and reward vector
        for advisor_idx, actions in context.items():
            for action, value in actions.items():
                context_matrix[advisor_idx, action_index[action]] = value

        # Calculate y_hat and v_hat using vectorized operations
        n_advisors = torch.tensor(n_advisors, dtype=torch.float).to(self.device)
        for action_id, reward in six.viewitems(rewards):
            if self.use_exp4p_update:
                #print("use_exp4p_update")
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
                        + v_hat
                        * torch.sqrt(torch.log(n_advisors / self.delta) / (n_actions * self.max_rounds))
                    )
                )
                #print(f"exp4: {updates=}")
                if self.use_nn_inner_update:
                    #print("use_nn_inner_update")
                    self.inner_nn_update_optimizer.zero_grad()
                    train_nn_updates = updates.clone().detach().to(self.device)
                    w_out = w.clone().detach().to(self.device)

                    # the input is exp4_update
                    train_nn_updates = self.inner_nn_update_model(train_nn_updates)
                    w_out = w_out * torch.exp(train_nn_updates)
                    w_out = w_out / torch.sum(w_out)
                    train_action_probs = self._calculate_action_probs(context, w_out)
                    p = train_action_probs[action_index[action_id]]
                    loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
                    loss.backward()
                    self.inner_nn_update_optimizer.step()

                    updates = self.inner_nn_update_model(updates)
                    #print(f"inner: {updates=}")

                # exp4 update
                w = w * torch.exp(updates)
                w = w / torch.sum(w)
                w = w.detach()
                #print(f"{w=}")

            if self.use_nn_outer_update:
                #print("use_nn_outer_update")
                self.outer_nn_update_optimizer.zero_grad()
                w_input = w.clone().detach().to(self.device)
                train_nn_updates = self.outer_nn_update_model(w_input)
                w_out = w_input * torch.exp(train_nn_updates)
                w_out = w_out / torch.sum(w_out)
                train_action_probs = self._calculate_action_probs(context, w_out)
                p = train_action_probs[action_index[action_id]]
                loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
                loss.backward()
                self.outer_nn_update_optimizer.step()

                # the input is w after the exp4 update
                neural_updates = self.outer_nn_update_model(w_input)
                neural_updates = torch.exp(neural_updates)

                w = w * torch.exp(neural_updates)
                w = w / torch.sum(w)
                w = w.detach()

                #print(f"outer: {neural_updates=}")
                #print(f"{w=}")
        if self.use_expert is not None and self.use_expert < self.num_advisors:
            w = torch.zeros(int(n_advisors)).to(self.device)
            w[self.use_expert] = 1.0
            w.detach()

        # self._history_storage.add_reward(history_id, rewards)
        self._model_storage.save_model({"action_probs": action_probs, "w": w})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)


class Exp4PInnerNNUpdate(Exp4PNN):
    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
    ):
        use_nn_inner_update = True
        use_nn_outer_update = False
        use_exp4p_update = True
        super(Exp4PInnerNNUpdate, self).__init__(
            actions,
            historystorage,
            modelstorage,
            use_nn_inner_update,
            use_nn_outer_update,
            use_exp4p_update,
            num_advisors,
            delta,
            p_min,
            max_rounds,
            hidden_sizes,
            dropout_prob,
            lr,
        )


class Exp4PInnerOuterNNUpdate(Exp4PNN):
    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
    ):
        use_nn_inner_update = True
        use_nn_outer_update = True
        use_exp4p_update = True
        super(Exp4PInnerOuterNNUpdate, self).__init__(
            actions,
            historystorage,
            modelstorage,
            use_nn_inner_update,
            use_nn_outer_update,
            use_exp4p_update,
            num_advisors,
            delta,
            p_min,
            max_rounds,
            hidden_sizes,
            dropout_prob,
            lr,
        )


class Exp4POuterNNUpdate(Exp4PNN):
    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
    ):
        use_nn_inner_update = False
        use_nn_outer_update = True
        use_exp4p_update = True
        super(Exp4POuterNNUpdate, self).__init__(
            actions,
            historystorage,
            modelstorage,
            use_nn_inner_update,
            use_nn_outer_update,
            use_exp4p_update,
            num_advisors,
            delta,
            p_min,
            max_rounds,
            hidden_sizes,
            dropout_prob,
            lr,
        )


class OuterNNUpdateExperts(Exp4PNN):
    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
    ):
        use_nn_inner_update = False
        use_nn_outer_update = True
        use_exp4p_update = False
        super(OuterNNUpdateExperts, self).__init__(
            actions,
            historystorage,
            modelstorage,
            use_nn_inner_update,
            use_nn_outer_update,
            use_exp4p_update,
            num_advisors,
            delta,
            p_min,
            max_rounds,
            hidden_sizes,
            dropout_prob,
            lr,
        )

class ExpertAgent(Exp4PNN):
    def __init__(
        self,
        actions,
        historystorage,
        modelstorage,
        use_expert,
        num_advisors=2,
        delta=0.1,
        p_min=None,
        max_rounds=10000,
        hidden_sizes=None,
        dropout_prob=0.5,
        lr=1e-3,
    ):
        use_nn_inner_update = False
        use_nn_outer_update = False
        use_exp4p_update = False
        hidden_sizes=[num_advisors, num_advisors]
        super(ExpertAgent, self).__init__(
            actions,
            historystorage,
            modelstorage,
            use_nn_inner_update,
            use_nn_outer_update,
            use_exp4p_update,
            use_expert,
            num_advisors,
            delta,
            p_min,
            max_rounds,
            hidden_sizes,
            dropout_prob,
            lr,
        )
