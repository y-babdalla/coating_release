"""Unit tests for process_spectrum_dataframe and process_individual_spectrum."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.process_spectra import process_individual_spectrum, process_spectrum_dataframe


@pytest.fixture
def mock_x_values() -> np.ndarray:
    """Fixture to return a small array of synthetic x-values (wavelengths).

    This is used to patch pd.read_excel(...) calls in process_individual_spectrum.
    """
    return np.linspace(400, 4000, num=3601)  # e.g., 3601 points from 400 to 4000 cm^-1


@pytest.fixture
def sample_spectrum_df() -> pd.DataFrame:
    """Create a sample DataFrame with spectral data and minimal metadata columns.

    The numeric columns emulate spectral intensities at a few arbitrary wavelengths.
    """
    data = {
        "medium": ["A", "B", "A", "B", "A"],
        "time": [0, 1, 2, 3, 4],
        "release": [10, 20, 15, 25, 18],
        "index": [0, 1, 2, 3, 4],
        # A few numeric columns, simulating wavelengths
        400.0: [1.0, 2.0, 3.0, 4.0, 5.0],
        500.0: [2.0, 3.0, 4.0, 5.0, 6.0],
        600.0: [3.0, 4.0, 5.0, 6.0, 7.0],
        700.0: [4.0, 5.0, 6.0, 7.0, 8.0],
    }
    return pd.DataFrame(data)


@patch("src.process_spectra.pd.read_excel")
def test_process_individual_spectrum_no_downsample(
    mock_read_excel: patch, mock_x_values: np.ndarray
) -> None:
    """Test process_individual_spectrum without downsampling.

    We mock pd.read_excel to return our synthetic x-values.
    """
    mock_read_excel.return_value = pd.DataFrame({"x": mock_x_values})
    y = np.sin(mock_x_values / 200) + 0.5 * np.random.rand(len(mock_x_values))

    result = process_individual_spectrum(y=y, show_plot=False, downsample=None)

    assert isinstance(
        result, np.ndarray
    ), "Expected a NumPy array from process_individual_spectrum."
    assert len(result) == len(y), "When downsample is None, length of output should match input."


@patch("src.process_spectra.pd.read_excel")
def test_process_individual_spectrum_with_downsample(
    mock_read_excel: patch, mock_x_values: np.ndarray
) -> None:
    """Test process_individual_spectrum with a specific downsample factor.

    We mock pd.read_excel to return our synthetic x-values.
    """
    mock_read_excel.return_value = pd.DataFrame({"x": mock_x_values})

    y = np.random.rand(len(mock_x_values))
    downsample_factor = 10

    result = process_individual_spectrum(y=y, show_plot=False, downsample=downsample_factor)

    expected_length = len(mock_x_values[::downsample_factor])
    assert (
        len(result) == expected_length
    ), f"Output length should be original length divided by {downsample_factor}."


@patch("src.process_spectra.pd.read_excel")
def test_process_individual_spectrum_baseline_correction(
    mock_read_excel: patch, mock_x_values: np.ndarray
) -> None:
    """Test that baseline correction is applied (i.e., output differs from raw data in some way).

    We mock pd.read_excel to return our synthetic x-values.
    """
    mock_read_excel.return_value = pd.DataFrame({"x": mock_x_values})

    y = 0.0001 * (mock_x_values - 2000) ** 2 + 5

    result = process_individual_spectrum(y=y, show_plot=False, downsample=None)

    # We expect baseline correction + smoothing => result should differ from the original
    # polynomial (especially since Savitzky-Golay smoothing is applied). We'll just check that
    # they're not identical.
    assert not np.allclose(
        result, y
    ), "After baseline correction & smoothing, result should differ."


@patch("src.process_spectra.process_individual_spectrum")
def test_process_spectrum_dataframe_basic(
    mock_process_individual_spectrum: patch, sample_spectrum_df: pd.DataFrame
) -> None:
    """Test process_spectrum_dataframe with basic usage.

    We mock process_individual_spectrum to isolate testing this function's logic.
    """
    mock_process_individual_spectrum.return_value = np.array([0.5, 0.6, 0.7, 0.8])

    processed = process_spectrum_dataframe(
        spectrum_df=sample_spectrum_df, downsample=None, encode_labels=False
    )

    # We expect the same number of rows as the input
    assert len(processed) == len(sample_spectrum_df), "Output row count must match input."

    # The function appends the processed spectral data plus the original metadata columns
    # Since each row returned a [0.5, 0.6, 0.7, 0.8], and the original 4 columns are 'medium',
    # 'time', 'release', 'index'
    # total columns = 4 processed spectral columns + 4 metadata = 8
    assert processed.shape[1] == 8, "Expected processed columns plus metadata columns."

    assert mock_process_individual_spectrum.call_count == len(
        sample_spectrum_df
    ), "process_individual_spectrum should be called once per row."


@patch("src.process_spectra.process_individual_spectrum")
def test_process_spectrum_dataframe_downsample_mismatch(
    mock_process_individual_spectrum: patch, sample_spectrum_df: pd.DataFrame
) -> None:
    """Test that ValueError is raised when the data length does not match column count."""
    mock_process_individual_spectrum.return_value = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError, match="Processed data length does not match"):
        _ = process_spectrum_dataframe(
            spectrum_df=sample_spectrum_df, downsample=2, encode_labels=False
        )
