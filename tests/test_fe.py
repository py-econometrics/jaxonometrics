"""
Comprehensive Fixed Effects Tests for jaxonometrics vs pyfixest comparison.
Tests multiple combinations of fixed effects with the new JAX API using proper assertions.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
import pytest
import pyfixest as pf
from jaxonometrics import LinearRegression

np.random.seed(42)


@pytest.fixture
def panel_data():
    """Generate realistic panel data for testing."""
    # Panel dimensions
    n_firms = 1000
    n_years = 20
    n_regions = 5
    n_obs = n_firms * n_years

    # Generate panel structure
    firm_id = np.repeat(np.arange(n_firms), n_years)
    year_id = np.tile(np.arange(n_years), n_firms)
    region_id = np.random.choice(n_regions, n_firms)  # Each firm belongs to one region
    region_id = np.repeat(region_id, n_years)  # Repeat for panel structure

    # Covariates
    X = np.random.randn(n_obs, 3)
    X[:, 0] = np.random.randn(n_obs) * 0.5  # Log sales
    X[:, 1] = np.random.randn(n_obs) * 0.3  # Log employment
    X[:, 2] = np.random.randn(n_obs) * 0.2  # R&D intensity

    # True effects
    true_coef = np.array([0.6, 0.4, 0.3])
    firm_effects = np.random.randn(n_firms) * 0.4
    year_effects = np.random.randn(n_years) * 0.2
    region_effects = np.random.randn(n_regions) * 0.3

    # Generate outcome (log productivity)
    y = (
        X @ true_coef
        + firm_effects[firm_id]
        + year_effects[year_id]
        + region_effects[region_id]
        + np.random.randn(n_obs) * 0.1
    )

    # Create DataFrame for pyfixest
    df = pd.DataFrame(
        {
            "log_sales": X[:, 0],
            "log_employment": X[:, 1],
            "rd_intensity": X[:, 2],
            "log_productivity": y,
            "firm_id": firm_id,
            "year_id": year_id,
            "region_id": region_id,
        }
    )

    # Convert to JAX arrays for jaxonometrics
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)
    firm_jax = jnp.asarray(firm_id, dtype=jnp.int32)
    year_jax = jnp.asarray(year_id, dtype=jnp.int32)
    region_jax = jnp.asarray(region_id, dtype=jnp.int32)

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_jax": X_jax,
        "y_jax": y_jax,
        "firm_id": firm_id,
        "year_id": year_id,
        "region_id": region_id,
        "firm_jax": firm_jax,
        "year_jax": year_jax,
        "region_jax": region_jax,
        "true_coef": true_coef,
        "n_firms": n_firms,
        "n_years": n_years,
        "n_regions": n_regions,
        "n_obs": n_obs,
    }


@pytest.mark.unit
def test_simple_ols(panel_data):
    """Test simple OLS (pooled regression with intercept)."""
    data = panel_data
    
    # jaxonometrics (needs intercept for OLS)
    X_ols_jax = jnp.column_stack([jnp.ones(data["n_obs"]), data["X_jax"]])
    jax_ols = LinearRegression(solver="lineax")
    jax_ols.fit(X_ols_jax, data["y_jax"])
    jax_ols_coef = jax_ols.params["coef"][1:]  # Skip intercept

    # pyfixest
    pf_ols = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity", data["df"]
    )
    pf_ols_coef = pf_ols.coef().values[1:]  # Skip intercept

    # Assert coefficients are close (slightly relaxed tolerance for OLS)
    np.testing.assert_allclose(
        jax_ols_coef, pf_ols_coef, rtol=1e-6, atol=1e-6,
        err_msg="Simple OLS coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
def test_firm_fixed_effects(panel_data):
    """Test one-way fixed effects (firm)."""
    data = panel_data
    
    # jaxonometrics (no intercept for FE)
    jax_firm_fe = LinearRegression(solver="lineax")
    jax_firm_fe.fit(data["X_jax"], data["y_jax"], fe=[data["firm_jax"]])
    jax_firm_coef = jax_firm_fe.params["coef"]

    # pyfixest
    pf_firm_fe = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | firm_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_firm_coef = pf_firm_fe.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_firm_coef, pf_firm_coef, rtol=1e-8, atol=1e-8,
        err_msg="Firm FE coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
def test_year_fixed_effects(panel_data):
    """Test one-way fixed effects (year)."""
    data = panel_data
    
    # jaxonometrics
    jax_year_fe = LinearRegression(solver="lineax")
    jax_year_fe.fit(data["X_jax"], data["y_jax"], fe=[data["year_jax"]])
    jax_year_coef = jax_year_fe.params["coef"]

    # pyfixest
    pf_year_fe = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | year_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_year_coef = pf_year_fe.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_year_coef, pf_year_coef, rtol=1e-8, atol=1e-8,
        err_msg="Year FE coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
def test_region_fixed_effects(panel_data):
    """Test one-way fixed effects (region)."""
    data = panel_data
    
    # jaxonometrics
    jax_region_fe = LinearRegression(solver="lineax")
    jax_region_fe.fit(data["X_jax"], data["y_jax"], fe=[data["region_jax"]])
    jax_region_coef = jax_region_fe.params["coef"]

    # pyfixest
    pf_region_fe = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | region_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_region_coef = pf_region_fe.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_region_coef, pf_region_coef, rtol=1e-8, atol=1e-8,
        err_msg="Region FE coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
def test_firm_year_two_way_fe(panel_data):
    """Test two-way fixed effects (firm + year)."""
    data = panel_data
    
    # jaxonometrics
    jax_twoway = LinearRegression(solver="lineax")
    jax_twoway.fit(
        data["X_jax"], data["y_jax"], fe=[data["firm_jax"], data["year_jax"]]
    )
    jax_twoway_coef = jax_twoway.params["coef"]

    # pyfixest
    pf_twoway = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | firm_id + year_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_twoway_coef = pf_twoway.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_twoway_coef, pf_twoway_coef, rtol=1e-8, atol=1e-8,
        err_msg="Firm + Year FE coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
def test_firm_region_two_way_fe(panel_data):
    """Test two-way fixed effects (firm + region)."""
    data = panel_data
    
    # jaxonometrics
    jax_firm_region = LinearRegression(solver="lineax")
    jax_firm_region.fit(
        data["X_jax"], data["y_jax"], fe=[data["firm_jax"], data["region_jax"]]
    )
    jax_firm_region_coef = jax_firm_region.params["coef"]

    # pyfixest
    pf_firm_region = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | firm_id + region_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_firm_region_coef = pf_firm_region.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_firm_region_coef, pf_firm_region_coef, rtol=1e-8, atol=1e-8,
        err_msg="Firm + Region FE coefficients do not match between jaxonometrics and pyfixest"
    )


@pytest.mark.fe
@pytest.mark.slow
def test_three_way_fixed_effects(panel_data):
    """Test three-way fixed effects (firm + year + region)."""
    data = panel_data
    
    # jaxonometrics
    jax_threeway = LinearRegression(solver="lineax")
    jax_threeway.fit(
        data["X_jax"],
        data["y_jax"],
        fe=[data["firm_jax"], data["year_jax"], data["region_jax"]],
    )
    jax_threeway_coef = jax_threeway.params["coef"]

    # pyfixest
    pf_threeway = pf.feols(
        "log_productivity ~ log_sales + log_employment + rd_intensity | firm_id + year_id + region_id",
        data["df"],
        demeaner_backend="jax",
    )
    pf_threeway_coef = pf_threeway.coef().values

    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_threeway_coef, pf_threeway_coef, rtol=1e-8, atol=1e-8,
        err_msg="Three-way FE coefficients do not match between jaxonometrics and pyfixest"
    )


class TestFixedEffectsComprehensive:
    """Comprehensive test class for all fixed effects combinations."""
    
    def test_coefficient_shapes_consistency(self, panel_data):
        """Test that all FE regressions return the expected number of coefficients."""
        data = panel_data
        expected_n_coef = 3  # 3 covariates
        
        # Test various FE combinations
        fe_combinations = [
            ([data["firm_jax"]], "firm FE"),
            ([data["year_jax"]], "year FE"), 
            ([data["region_jax"]], "region FE"),
            ([data["firm_jax"], data["year_jax"]], "firm + year FE"),
            ([data["firm_jax"], data["region_jax"]], "firm + region FE"),
            ([data["firm_jax"], data["year_jax"], data["region_jax"]], "three-way FE"),
        ]
        
        for fe_vars, description in fe_combinations:
            model = LinearRegression(solver="lineax")
            model.fit(data["X_jax"], data["y_jax"], fe=fe_vars)
            coef = model.params["coef"]
            
            assert len(coef) == expected_n_coef, (
                f"{description} should return {expected_n_coef} coefficients, "
                f"but got {len(coef)}"
            )
    
    def test_convergence_flags(self, panel_data):
        """Test that demeaning converges for all FE combinations."""
        data = panel_data
        
        # Test that demeaning converges (this is tested internally but we can't access it directly)
        # Instead, we test that the fit method doesn't raise any convergence warnings
        
        fe_combinations = [
            [data["firm_jax"]],
            [data["year_jax"]],
            [data["region_jax"]],
            [data["firm_jax"], data["year_jax"]],
            [data["firm_jax"], data["region_jax"]],
            [data["firm_jax"], data["year_jax"], data["region_jax"]],
        ]
        
        for fe_vars in fe_combinations:
            model = LinearRegression(solver="lineax")
            # Should not raise any exceptions
            model.fit(data["X_jax"], data["y_jax"], fe=fe_vars)
            # Should have computed coefficients
            assert "coef" in model.params
            assert model.params["coef"] is not None


@pytest.mark.fe
def test_minimal_example():
    """Test with a minimal example to ensure basic functionality."""
    np.random.seed(123)
    
    # Simple test data
    n_obs = 100
    X = np.random.randn(n_obs, 2)
    group_ids = np.repeat(np.arange(10), 10)  # 10 groups, 10 obs each
    y = X @ np.array([0.5, -0.3]) + np.random.randn(n_obs) * 0.1
    
    # Convert to JAX
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)
    group_jax = jnp.asarray(group_ids, dtype=jnp.int32)
    
    # Create DataFrame for pyfixest
    df = pd.DataFrame({
        'y': y,
        'x1': X[:, 0],
        'x2': X[:, 1],
        'group': group_ids
    })
    
    # Test FE
    jax_fe = LinearRegression(solver="lineax")
    jax_fe.fit(X_jax, y_jax, fe=[group_jax])
    jax_coef = jax_fe.params["coef"]
    
    pf_fe = pf.feols("y ~ x1 + x2 | group", df, demeaner_backend="jax")
    pf_coef = pf_fe.coef().values
    
    # Assert coefficients are close
    np.testing.assert_allclose(
        jax_coef, pf_coef, rtol=1e-8, atol=1e-8,
        err_msg="Minimal FE example coefficients do not match"
    )


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])