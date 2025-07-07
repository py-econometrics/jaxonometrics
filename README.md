# `jaxonometrics`: Econometrics in jax

![Tests](https://github.com/py-econometrics/jaxonometrics/workflows/Tests/badge.svg)
![Fixed Effects Tests](https://github.com/py-econometrics/jaxonometrics/workflows/Fixed%20Effects%20Tests/badge.svg)

Simple library that provides performant implementations of standard econometrics routines in the JAX ecosystem.

- `jax` arrays everywhere
- `lineax` for solving linear systems
- `jaxopt` and `optax` for numerical optimization (Levenberg–Marquardt for NNLS-type problems and SGD for larger problems)

## Features

- **Linear Regression** with multiple solver backends (lineax, JAX, numpy)
- **Fixed Effects Regression** with JAX-accelerated alternating projections
- **GMM and IV Estimation**
- **Causal Inference** (IPW, AIPW, Entropy Balancing)
- **Maximum Likelihood Estimation** (Logistic, Poisson)

### Fixed Effects

jaxonometrics supports high-performance fixed effects regression with multiple FE variables:

```python
from jaxonometrics import LinearRegression
import jax.numpy as jnp

# Your data
X = jnp.asarray(data)  # (n_obs, n_features)
y = jnp.asarray(target)  # (n_obs,)
firm_ids = jnp.asarray(firm_identifiers, dtype=jnp.int32)
year_ids = jnp.asarray(year_identifiers, dtype=jnp.int32)

# Two-way fixed effects
model = LinearRegression(solver="lineax")
model.fit(X, y, fe=[firm_ids, year_ids])
coefficients = model.params["coef"]
```

## Installation and Development

### Install

```bash
uv pip install git+https://github.com/py-econometrics/jaxonometrics
```

or clone the repository and install in editable mode.


### Testing

Run the full test suite:
```bash
pytest tests/ -v
```

Run only fixed effects tests:
```bash
pytest tests/ -m fe -v
```

Run tests excluding slow ones:
```bash
pytest tests/ -m "not slow" -v
```
