"""
JAX-based demeaning for fixed effects regression.
Adapted from pyfixest's JAX backend implementation.
"""
from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import config


@partial(jax.jit, static_argnames=("n_groups", "tol", "maxiter"))
def _demean_jax_impl(
    x: jnp.ndarray,
    flist: jnp.ndarray,
    weights: jnp.ndarray,
    n_groups: int,
    tol: float,
    maxiter: int,
) -> tuple[jnp.ndarray, bool]:
    """JIT-compiled implementation of demeaning via alternating projections."""
    n_factors = flist.shape[1]

    @jax.jit
    def _apply_factor(carry, j):
        """Process a single factor."""
        x = carry
        factor_ids = flist[:, j]
        wx = x * weights[:, None]

        # Compute group weights and weighted sums
        group_weights = jnp.bincount(factor_ids, weights=weights, length=n_groups)
        group_sums = jax.vmap(
            lambda col: jnp.bincount(factor_ids, weights=col, length=n_groups)
        )(wx.T).T

        # Compute and subtract means
        means = group_sums / jnp.maximum(group_weights[:, None], 1e-12)  # Avoid division by zero
        return x - means[factor_ids], None

    @jax.jit
    def _demean_step(x_curr):
        """Single demeaning step for all factors."""
        # Process all factors using scan
        result, _ = jax.lax.scan(_apply_factor, x_curr, jnp.arange(n_factors))
        return result

    @jax.jit
    def _body_fun(state):
        """Body function for while_loop."""
        i, x_curr, x_prev, converged = state
        x_new = _demean_step(x_curr)
        max_diff = jnp.max(jnp.abs(x_new - x_curr))
        has_converged = max_diff < tol
        return i + 1, x_new, x_curr, has_converged

    @jax.jit
    def _cond_fun(state):
        """Condition function for while_loop."""
        i, _, _, converged = state
        return jnp.logical_and(i < maxiter, jnp.logical_not(converged))

    # Run the iteration loop using while_loop
    init_state = (0, x, x - 1.0, False)
    final_i, final_x, _, converged = jax.lax.while_loop(
        _cond_fun, _body_fun, init_state
    )

    return final_x, converged


def demean_jax(
    x: Union[np.ndarray, jnp.ndarray],
    flist: Union[np.ndarray, jnp.ndarray],
    weights: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[jnp.ndarray, bool]:
    """
    Demean array using JAX implementation of alternating projections.
    
    Parameters
    ----------
    x : array-like
        Input array of shape (n_samples, n_features) to demean.
    flist : array-like
        Fixed effects array of shape (n_samples, n_factors) with integer factor IDs.
    weights : array-like, optional
        Weights array of shape (n_samples,). If None, uses uniform weights.
    tol : float, optional
        Tolerance for convergence. Default is 1e-08.
    maxiter : int, optional
        Maximum number of iterations. Default is 100_000.
    
    Returns
    -------
    tuple[jnp.ndarray, bool]
        Tuple of (demeaned_array, converged).
    """
    # Enable float64 precision for better numerical stability
    config.update("jax_enable_x64", True)
    
    # Convert inputs to JAX arrays
    x = jnp.asarray(x, dtype=jnp.float64)
    flist = jnp.asarray(flist, dtype=jnp.int32)
    
    # Handle weights
    if weights is None:
        weights = jnp.ones(x.shape[0], dtype=jnp.float64)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float64)
    
    # Ensure x is 2D
    if x.ndim == 1:
        x = x[:, None]
    
    # Ensure flist is 2D
    if flist.ndim == 1:
        flist = flist[:, None]
    
    # Compute number of groups across all factors
    n_groups = int(jnp.max(flist) + 1)
    
    # Call the JIT-compiled implementation
    result, converged = _demean_jax_impl(
        x, flist, weights, n_groups, tol, maxiter
    )
    
    return result, converged


def prepare_fixed_effects(fe_vars: list) -> jnp.ndarray:
    """
    Prepare fixed effects variables for demeaning.
    
    Parameters
    ----------
    fe_vars : list
        List of arrays containing fixed effects variables.
        
    Returns
    -------
    jnp.ndarray
        Array of shape (n_samples, n_factors) with integer factor IDs.
    """
    if not fe_vars:
        return None
    
    # Convert each FE variable to consecutive integers starting from 0
    fe_arrays = []
    offset = 0
    
    for fe_var in fe_vars:
        fe_array = jnp.asarray(fe_var)
        
        # Get unique values and create mapping
        unique_vals = jnp.unique(fe_array)
        n_unique = len(unique_vals)
        
        # Create consecutive integer mapping
        fe_mapped = jnp.searchsorted(unique_vals, fe_array) + offset
        fe_arrays.append(fe_mapped)
        offset += n_unique
    
    # Stack all FE variables
    return jnp.stack(fe_arrays, axis=1)