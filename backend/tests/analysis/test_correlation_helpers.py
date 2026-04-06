import numpy as np
import pytest

from app.analysis.correlation import (
    _default_price_column,
    _ols_beta_y_on_x,
    _resolve_price_metric,
)


def test_ols_beta_simple():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x
    assert abs(_ols_beta_y_on_x(y, x) - 2.0) < 1e-9


def test_default_price_column():
    assert _default_price_column("same_day", "none") == "returns"
    assert _default_price_column("same_day", "spy_beta_residual") == "excess_returns"
    assert _default_price_column("sentiment_leads_1d", "none") == "forward_1d_return"
    assert (
        _default_price_column("sentiment_leads_1d", "spy_beta_residual")
        == "forward_excess_return"
    )


def test_resolve_auto():
    assert _resolve_price_metric(None, "same_day", "none") == "returns"
    assert _resolve_price_metric("auto", "sentiment_leads_1d", "none") == "forward_1d_return"


def test_resolve_returns_maps_to_forward_when_lead():
    assert (
        _resolve_price_metric("returns", "sentiment_leads_1d", "none")
        == "forward_1d_return"
    )


def test_resolve_invalid_price_metric():
    with pytest.raises(ValueError):
        _resolve_price_metric("volume", "same_day", "none")
