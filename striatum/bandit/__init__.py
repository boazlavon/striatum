"""Bandit algorithm classes
"""

from .exp3 import Exp3
from .exp3nn import Exp3NN, Exp3OuterNNUpdate, OuterNNUpdate, NeuralAgent
from .exp4p import Exp4P
from .exp4pnn import Exp4PNN, Exp4POuterNNUpdate, Exp4PInnerNNUpdate, OuterNNUpdateExperts, Exp4PInnerOuterNNUpdate
from .linthompsamp import LinThompSamp
from .linucb import LinUCB
from .ucb1 import UCB1


__all__ = [
    "Exp3",
    "Exp3NN",
    "Exp3OuterNNUpdate",
    "OuterNNUpdate",
    "NeuralAgent",
    "Exp4P",
    "Exp4PNN",
    "Exp4POuterNNUpdate",
    "Exp4PInnerNNUpdate",
    "Exp4PInnerOuterNNUpdate",
    "OuterNNUpdateExperts",
    "LinThompSamp",
    "LinUCB",
    "UCB1",
]
