""" Exp3: Exponential-weight algorithm for Exploration and Exploitation
This module contains a class that implements EXP3, a bandit algorithm that
randomly choose an action according to a learned probability distribution.
"""

import logging

import numpy as np
import six
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .bandit import BaseBandit
from ..utils import get_random_state

LOGGER = logging.getLogger(__name__)


class NeuralNetwork(nn.Module):
    def __init__(
        self, layers_dims, dropout_prob=0.5, init_p_factor_raw_weight=2.8, init_exp3_factor_raw_weight=0
    ):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()  # Use ModuleList to properly register the layers
        # Initialize layers
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

        self.exp3_factor_raw_weight = nn.Parameter(
            torch.tensor(init_exp3_factor_raw_weight, dtype=torch.float64)
        )  # Use only the residual first
        self.exp3_factor = 0.5 * (torch.tanh(self.exp3_factor_raw_weight) + 1)

    def forward(self, x, exp3_update=None):
        if exp3_update is not None:
            x = torch.cat((x, torch.tensor([exp3_update])), dim=0)
        for layer in self.layers:
            x = layer(x)

        # Apply the single learnable residual connection before the final output
        # Using the original input, scaled by a learnable parameter
        x = F.softmax(x, dim=0)
        self.p_factor = 0.5 * (torch.tanh(self.p_factor_raw_weight) + 1)
        x = x * self.p_factor

        if exp3_update is not None:
            x = x + self.exp3_factor * exp3_update
        return x


class Exp3NN(BaseBandit):
    r"""Exp3 algorithm.

    Parameters
    ----------
    history_storage : HistoryStorage object
        The HistoryStorage object to store history context, actions and rewards.

    model_storage : ModelStorage object
        The ModelStorage object to store model parameters.

    action_storage : ActionStorage object
        The ActionStorage object to store actions.

    recommendation_cls : class (default: None)
        The class used to initiate the recommendations. If None, then use
        default Recommendation class.

    gamma: float, 0 < gamma <= 1
        The parameter used to control the minimum chosen probability for each
        action.

    random_state: {int, np.random.RandomState} (default: None)
        If int, np.random.RandomState will used it as seed. If None, a random
        seed will be used.

    References
    ----------
    .. [1]  Peter Auer, Nicolo Cesa-Bianchi, et al. "The non-stochastic
            multi-armed bandit problem ." SIAM Journal of Computing. 2002.
    """

    def __init__(
        self,
        history_storage,
        model_storage,
        action_storage,
        use_exp3=True,
        use_nn_update=False,
        use_nn_probs=False,
        use_exp3_nn_updates=False,
        gamma=0.3,
        hidden_sizes=[64, 128],
        dropout_prob=0.5,
        lr=1e-2,
        recommendation_cls=None,
        random_state=None,
    ):
        super(Exp3NN, self).__init__(history_storage, model_storage, action_storage, recommendation_cls)
        self.random_state = get_random_state(random_state)
        self.action_ids = list(self._action_storage.iterids())
        self.action_ids.sort()
        self.action_index = {action_id: idx for idx, action_id in enumerate(self.action_ids)}
        self.n_actions = len(self.action_ids)
        self.use_exp3 = use_exp3
        self.use_nn_update = use_nn_update
        self.use_nn_probs = use_nn_probs
        self.use_exp3_nn_updates = use_exp3_nn_updates

        # gamma in (0,1]
        if not isinstance(gamma, float):
            raise ValueError("gamma should be float, the one" "given is: %f" % gamma)
        elif (gamma <= 0) or (gamma > 1):
            raise ValueError("gamma should be in (0, 1], the one" "given is: %f" % gamma)
        else:
            self.gamma = gamma

        # Initialize the model storage
        w = torch.ones(self.n_actions)
        self._model_storage.save_model({"w": w})

        layers = [self.n_actions] + hidden_sizes + [self.n_actions]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.neural_update_model = NeuralNetwork(layers, dropout_prob).to(self.device)
        self.neural_update_optimizer = optim.AdamW(self.neural_update_model.parameters(), lr=lr)

        self.neural_probs_model = NeuralNetwork(layers, dropout_prob).to(self.device)
        self.neural_probs_optimizer = optim.AdamW(self.neural_probs_model.parameters(), lr=lr)

        self.exp3_update_model = NeuralNetwork(layers, dropout_prob).to(self.device)
        self.exp3_update_optimizer = optim.AdamW(self.exp3_update_model.parameters(), lr=lr)

    def _exp3_probs(self, w, gamma):
        """Exp3 algorithm."""
        w_sum = torch.sum(w)
        probs = (1 - gamma) * (w / w_sum) + (gamma / self.n_actions)
        return probs

    def get_action(self, context=None, n_actions=1):
        """Return the action to perform

        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context available.

        n_actions: int (default: None)
            Number of actions wanted to recommend users. If None, only return
            one action. If -1, get all actions.

        Returns
        -------
        history_id : int
            The history id of the action.

        recommendations : list of dict
            Each dict contains
            {Action object, estimated_reward, uncertainty}.
        """
        if len(self.action_ids) == 0:
            return self._get_action_with_empty_action_storage(context, n_actions)
        w = self._model_storage.get_model()["w"]
        prob = self._exp3_probs(w, self.gamma)
        if n_actions == -1:
            n_actions = self._action_storage.count()

        np_prob = prob.cpu().numpy()
        recommendation_ids = self.random_state.choice(
            self.action_ids, size=n_actions, p=np_prob, replace=False
        )

        recommendations = []  # pylint: disable=redefined-variable-type
        for action_id in recommendation_ids:
            recommendations.append(
                self._recommendation_cls(
                    action=self._action_storage.get(action_id),
                    estimated_reward=prob[self.action_index[action_id]],
                    uncertainty=prob[self.action_index[action_id]],
                    score=prob[self.action_index[action_id]],
                )
            )

        history_id = self._history_storage.add_history(context, recommendations)
        return history_id, recommendations

    def reward(self, history_id, rewards):
        """Reward the previous action with reward.

        Parameters
        ----------
        history_id : int
            The history id of the action to reward.

        rewards : dictionary
            The dictionary {action_id, reward}, where reward is a float.
        """
        w = self._model_storage.get_model()["w"]
        history = self._history_storage.get_unrewarded_history(history_id)
        n_actions = self._action_storage.count()
        if isinstance(history.recommendations, list):
            recommendations = history.recommendations
        else:
            recommendations = [history.recommendations]

        probs = self._exp3_probs(w, self.gamma)

        # Update the model
        for action_id, reward in rewards.items():
            if self.use_exp3:
                updates = self.gamma * (reward / probs[self.action_index[action_id]]) / n_actions
                updates = torch.exp(updates)
                # print(f"exp3 updates: {updates}")
                w[self.action_index[action_id]] = w[self.action_index[action_id]] * updates

            if self.use_nn_update:
                self.neural_update_optimizer.zero_grad()
                w_input = w.clone().detach().to(self.device)
                train_neural_updates = self.neural_update_model(w_input)
                w_out = w_input * torch.exp(train_neural_updates)
                w_out = w_out / torch.sum(w_out)
                train_probs = self._exp3_probs(w_out, self.gamma)
                p = train_probs[self.action_index[action_id]]
                loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
                loss.backward()
                self.neural_update_optimizer.step()

                neural_updates = self.neural_update_model(w_input)
                neural_updates = torch.exp(neural_updates)
                neural_updates = neural_updates[self.action_index[action_id]]
                # print(f"neural updates: {neural_updates}")
                w[self.action_index[action_id]] = w[self.action_index[action_id]] * neural_updates

            if self.use_nn_probs:
                self.neural_probs_optimizer.zero_grad()
                w_input = w.clone().detach().to(self.device)
                w_out = self.neural_update_model(w_input)
                train_probs = self._exp3_probs(w_out, self.gamma)
                p = train_probs[self.action_index[action_id]]
                loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
                loss.backward()
                self.neural_probs_optimizer.step()
                w_out = self.neural_update_model(w_input).detach()
                # print(f"full neural update: {w - w_out}")
                w = w_out

            if self.use_exp3_nn_updates:
                exp3_updates = self.gamma * (reward / probs[self.action_index[action_id]]) / n_actions
                # exp3_updates = torch.exp(exp3_updates)
                # print(f"exp3 updates: {torch.exp(exp3_updates)}")
                # w[self.action_index[action_id]] = w[self.action_index[action_id]] * exp3_updates

                self.exp3_update_optimizer.zero_grad()
                w_input = w.clone().detach().to(self.device)
                train_neural_updates = self.exp3_update_model(w_input, exp3_updates.clone().detach())
                w_out = w_input * torch.exp(train_neural_updates)
                w_out = w_out / torch.sum(w_out)
                train_probs = self._exp3_probs(w_out, self.gamma)
                p = train_probs[self.action_index[action_id]]
                loss = -1 * (torch.log(p) * reward + torch.log(1 - p) * (1 - reward))
                loss.backward(retain_graph=True)
                self.exp3_update_optimizer.step()

                with torch.no_grad():
                    neural_updates = self.exp3_update_model(w_input, exp3_updates.clone().detach())
                    neural_updates = torch.exp(neural_updates)
                    neural_updates = neural_updates[self.action_index[action_id]]
                    neural_updates = neural_updates.detach()
                    # print(f"neural+exp3 updates: {neural_updates}")
                    w[self.action_index[action_id]] = w[self.action_index[action_id]] * neural_updates

            w = w / torch.sum(w)
            w = w.detach()

        self._model_storage.save_model({"w": w})

        # Update the history
        self._history_storage.add_reward(history_id, rewards)

    def add_action(self, actions):
        """Add new actions (if needed).

        Parameters
        ----------
        actions : iterable
            A list of Action objects for recommendation
        """
        self._action_storage.add(actions)

        w = self._model_storage.get_model()["w"]

        for action in actions:
            w[action.id] = 1.0  # weight vector

        self._model_storage.save_model({"w": w})

    def remove_action(self, action_id):
        """Remove action by id.

        Parameters
        ----------
        action_id : int
            The id of the action to remove.
        """
        w = self._model_storage.get_model()["w"]
        del w[action_id]
        self._model_storage.save_model({"w": w})
        self._action_storage.remove(action_id)


class NeuralUpdateExp3(Exp3NN):
    def __init__(
        self,
        history_storage,
        model_storage,
        action_storage,
        use_exp3=True,
        use_nn_update=True,
        use_nn_probs=False,
        use_exp3_nn_updates=False,
        gamma=0.3,
        hidden_size1=64,
        hidden_size2=128,
        dropout_prob=0.5,
        lr=1e-2,
        recommendation_cls=None,
        random_state=None,
    ):
        super(Exp3NN, self).__init__(
            history_storage,
            model_storage,
            action_storage,
            use_exp3,
            use_nn_update,
            use_nn_probs,
            use_exp3_nn_updates,
            gamma,
            hidden_size1,
            hidden_size2,
            dropout_prob,
            lr,
            recommendation_cls,
            random_state,
        )


class Exp3NNUpdate(Exp3NN):
    def __init__(
        self,
        history_storage,
        model_storage,
        action_storage,
        use_exp3,
        gamma=0.3,
        hidden_sizes=[64, 128],
        dropout_prob=0.5,
        lr=1e-2,
        recommendation_cls=None,
        random_state=None,
    ):
        use_nn_update = True
        use_nn_probs = False
        use_exp3_nn_updates = False
        super(Exp3NNUpdate, self).__init__(
            history_storage,
            model_storage,
            action_storage,
            use_exp3,
            use_nn_update,
            use_nn_probs,
            use_exp3_nn_updates,
            gamma,
            hidden_sizes,
            dropout_prob,
            lr,
            recommendation_cls,
            random_state,
        )


class Exp3NNDist(Exp3NN):
    def __init__(
        self,
        history_storage,
        model_storage,
        action_storage,
        hidden_sizes=[64, 128],
        dropout_prob=0.5,
        lr=1e-2,
        recommendation_cls=None,
        random_state=None,
    ):
        use_exp3 = False
        use_nn_update = False
        use_nn_probs = True
        use_exp3_nn_updates = False
        gamma = 1e-5
        super(Exp3NNDist, self).__init__(
            history_storage,
            model_storage,
            action_storage,
            use_exp3,
            use_nn_update,
            use_nn_probs,
            use_exp3_nn_updates,
            gamma,
            hidden_sizes,
            dropout_prob,
            lr,
            recommendation_cls,
            random_state,
        )
