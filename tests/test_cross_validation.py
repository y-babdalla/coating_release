"""Tests for the nested cross-validation module.

Run these tests with:
    pytest test_cross_validation.py
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.cross_validation import apply_pls, get_model_and_params


def test_get_model_and_params_valid() -> None:
    """Test that get_model_and_params returns the correct model type and parameter dict."""
    valid_models = ["lightGBM", "XGBoost", "KNN", "SVM", "RF"]
    for model_name in valid_models:
        model, params = get_model_and_params(model_name)
        assert isinstance(
            model, BaseEstimator
        ), f"Returned model is not a scikit-learn estimator for {model_name}."
        assert isinstance(params, dict), f"Parameters are not a dictionary for {model_name}."
        assert len(params) > 0, f"Parameter dictionary is empty for {model_name}."


def test_get_model_and_params_invalid() -> None:
    """Test that get_model_and_params raises a ValueError for an unsupported model name."""
    with pytest.raises(ValueError):
        get_model_and_params("InvalidModel")


def test_apply_pls() -> None:
    """Test PLS transformation on a scaled dataset."""
    columns = [2001.063477, 1500.0, 1000.0, 158.482422, "medium", "time"]
    x_train_scaled = pd.DataFrame(np.random.rand(5, 6), columns=columns)
    x_test_scaled = pd.DataFrame(np.random.rand(3, 6), columns=columns)

    y_train = np.random.rand(5)

    x_train_pls, x_test_pls, plsr = apply_pls(
        x_train_scaled, x_test_scaled, y_train, pls_components=2
    )

    assert x_train_pls.shape == (5, 4), "PLS transform output shape for training is incorrect."
    assert x_test_pls.shape == (3, 4), "PLS transform output shape for test is incorrect."
    assert hasattr(plsr, "coef_"), "PLS model appears to be untrained."
