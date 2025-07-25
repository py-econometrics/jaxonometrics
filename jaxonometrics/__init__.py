"""
jaxonometrics: a Python package for econometric analysis in JAX.
"""

__version__ = "0.0.1"

from .base import BaseEstimator
from .causal import EntropyBalancing, IPW, AIPW # Added IPW, AIPW
from .gmm import GMM, LinearIVGMM, TwoStepGMM
from .linear import LinearRegression
from .mle import LogisticRegression, PoissonRegression, MaximumLikelihoodEstimator # Added MLE models
from .demean import demean_jax, prepare_fixed_effects

__all__ = [
    "BaseEstimator",
    "EntropyBalancing",
    "IPW", # Added
    "AIPW", # Added
    "GMM",
    "LinearIVGMM",
    "TwoStepGMM",
    "LinearRegression",
    "MaximumLikelihoodEstimator",
    "LogisticRegression",
    "PoissonRegression",
    "demean_jax",
    "prepare_fixed_effects",
]
