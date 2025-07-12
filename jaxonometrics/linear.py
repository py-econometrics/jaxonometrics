from typing import Dict, Optional, Union, List
from functools import partial

import numpy as np
import jax # Ensure jax is imported
import jax.numpy as jnp
import lineax as lx

from .base import BaseEstimator
from .demean import demean_jax, prepare_fixed_effects


# Helper function for JIT compilation of vcov calculations
@partial(jax.jit, static_argnames=['se_type', 'n', 'k']) # Mark se_type, n, and k as static
def _calculate_vcov_details(
    coef: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, se_type: str, n: int, k: int
):
    """Helper function to compute SEs, designed to be JIT compiled."""
    # n and k are also marked static because they are used in calculations
    # that might affect array shapes or intermediate computations in ways
    # JAX prefers to know at compile time (e.g., n / (n - k)).
    # While JAX can often trace through these, being explicit can be safer.
    ε = y - X @ coef
    if se_type == "HC1":
        M = jnp.einsum("ij,i,ik->jk", X, ε**2, X)
        XtX_inv = jnp.linalg.inv(X.T @ X)
        Σ = XtX_inv @ M @ XtX_inv
        return jnp.sqrt((n / (n - k)) * jnp.diag(Σ))
    elif se_type == "classical":
        XtX_inv = jnp.linalg.inv(X.T @ X)
        return jnp.sqrt(jnp.diag(XtX_inv) * jnp.var(ε, ddof=k))
    return None # Should not be reached if se_type is valid


class LinearRegression(BaseEstimator):
    """
    Linear regression model using lineax for efficient solving.

    This class provides a simple interface for fitting a linear regression
    model, especially useful for high-dimensional problems where p > n.
    """

    def __init__(self, solver="lineax"):
        """Initialize the LinearRegression model.

        Args:
            solver (str, optional): Solver. Defaults to "lineax", can also be "jax" or "numpy".
        """
        super().__init__()
        self.solver: str = solver

    def fit(
        self,
        X: jnp.ndarray,
        y: jnp.ndarray,
        se: str = None,
        fe: Optional[Union[List, jnp.ndarray]] = None,
        weights: Optional[jnp.ndarray] = None,
    ) -> "LinearRegression":
        """
        Fit the linear model.

        Args:
            X: The design matrix of shape (n_samples, n_features).
            y: The target vector of shape (n_samples,).
            se: Whether to compute standard errors. "HC1" for robust standard errors, "classical" for classical SEs.
            fe: Fixed effects variables. Can be a list of arrays or a 2D array.
            weights: Sample weights of shape (n_samples,).

        Returns:
            The fitted estimator.
        """

        # Store original data for potential SE calculation
        X_orig, y_orig = X, y
        
        # Handle fixed effects demeaning
        if fe is not None:
            # Prepare fixed effects
            if isinstance(fe, list):
                flist = prepare_fixed_effects(fe)
            else:
                flist = jnp.asarray(fe, dtype=jnp.int32)
                if flist.ndim == 1:
                    flist = flist[:, None]
            
            # Demean both X and y
            X_demeaned, X_converged = demean_jax(X, flist, weights)
            y_demeaned, y_converged = demean_jax(y[:, None], flist, weights)
            y_demeaned = y_demeaned.flatten()
            
            if not (X_converged and y_converged):
                print("Warning: Demeaning did not converge")
            
            # Use demeaned data for regression
            X, y = X_demeaned, y_demeaned

        if self.solver == "lineax":
            # Use least-squares solver when we have fixed effects (demeaned data may not be well-posed)
            well_posed_flag = False if fe is not None else None
            sol = lx.linear_solve(
                operator=lx.MatrixLinearOperator(X),
                vector=y,
                solver=lx.AutoLinearSolver(well_posed=well_posed_flag),
            )
            self.params = {"coef": sol.value}

        elif self.solver == "jax":
            sol = jnp.linalg.lstsq(X, y)
            self.params = {"coef": sol[0]}
        elif self.solver == "numpy":  # for completeness
            X_np, y_np = np.array(X), np.array(y) # Convert to numpy arrays for numpy solver
            sol = np.linalg.lstsq(X_np, y_np, rcond=None)
            self.params = {"coef": jnp.array(sol[0])} # Convert back to jax array

        if se:
            self._vcov(
                y=y_orig if fe is not None else y,
                X=X_orig if fe is not None else X,
                se_type=se, # Renamed to avoid conflict with self.se if it existed
            )
        return self

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if not isinstance(X, jnp.ndarray):
            X = jnp.array(X)
        return jnp.dot(X, self.params["coef"])

    def _vcov(
        self,
        y: jnp.ndarray,
        X: jnp.ndarray,
        se_type: str = "HC1", # Renamed from 'se'
    ) -> None:
        n, k = X.shape
        if self.params and "coef" in self.params:
            coef = self.params["coef"]
            se_values = _calculate_vcov_details(coef, X, y, se_type, n, k)
            if se_values is not None:
                self.params["se"] = se_values
        else:
            # This case should ideally not be reached if fit() is called first.
            print("Coefficients not available for SE calculation.")
