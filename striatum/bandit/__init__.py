"""Bandit algorithm classes
"""

from .exp3 import Exp3
from .exp3nn import Exp3NN, Exp3NNUpdate, Exp3NNDist
from .exp4p import Exp4P
from .exp4pnn import Exp4PNN
from .linthompsamp import LinThompSamp
from .linucb import LinUCB
from .ucb1 import UCB1


__all__ = ['Exp3', 'Exp3NN', 'Exp3NNUpdate', 'Exp3NNDist', 'Exp4P', 'Exp4PNN', 'LinThompSamp', 'LinUCB', 'UCB1']
