import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np
from pandas.testing import assert_frame_equal
from components.feature_engineer import FeatureEngineer, LagFeatureEngineer, FillNAFeatures
from stages.feature_stage import FeatureStage


class TestFillNAFeatures:
    """Tests for the FillNAFeatures class."""

    @pytest.fixture
    def fill_na_engineer(self):
        """Create a FillNAFeatures instance for testing."""
        return FillNAFeatures()

    @pytest.fixture
    def sample_df_with_nas(self):
        """Create a sample DataFrame with NAs for testing."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, np.nan, 4.0, np.nan],
            'B': [np.nan, 2.0, 3.0, np.nan, np.nan],
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        return df

    def test_transform_fills_na_with_forward_fill(self, fill_na_engineer, sample_df_with_nas):
        """Test that transform fills NAs with forward fill strategy."""
        result = fill_na_engineer.transform(sample_df_with_nas.copy())

        # # Expected result after ffill()
        # expected_after_ffill = pd.DataFrame({
        #     'A': [1.0, 2.0, 2.0, 4.0, 4.0],  # ffill propagates the last valid value
        #     'B': [np.nan, 2.0, 3.0, 3.0, 3.0],  # first NA can't be forward filled
        #     'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        # })

        # After ffill, remaining NAs should be filled with 0
        expected_after_fillna = pd.DataFrame({
            'A': [1.0, 2.0, 2.0, 4.0, 4.0],
            'B': [0.0, 2.0, 3.0, 3.0, 3.0],  # first NA becomes 0
            'C': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        # Check final result matches expected output after both operations
        assert_frame_equal(result, expected_after_fillna)

    def test_transform_with_all_na_column(self, fill_na_engineer):
        """Test transform handles columns with all NAs."""
        df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1.0, np.nan, 3.0]
        })

        result = fill_na_engineer.transform(df.copy())

        expected = pd.DataFrame({
            'A': [0.0, 0.0, 0.0],  # All NAs become 0
            'B': [1.0, 1.0, 3.0]  # NA gets forward filled
        })

        assert_frame_equal(result, expected)

    def test_transform_with_empty_df(self, fill_na_engineer):
        """Test transform handles empty DataFrames."""
        df = pd.DataFrame()
        result = fill_na_engineer.transform(df.copy())

        # Empty DataFrame should remain empty
        assert result.empty
        assert_frame_equal(result, df)

    def test_transform_with_no_nas(self, fill_na_engineer):
        """Test transform works correctly when there are no NAs."""
        df = pd.DataFrame({
            'A': [1.0, 2.0, 3.0],
            'B': [4.0, 5.0, 6.0]
        })

        result = fill_na_engineer.transform(df.copy())

        # DataFrame should remain unchanged
        assert_frame_equal(result, df)

    def test_transform_doesnt_modify_original(self, fill_na_engineer, sample_df_with_nas):
        """Test that transform doesn't modify the original DataFrame."""
        original = sample_df_with_nas.copy()
        result = fill_na_engineer.transform(sample_df_with_nas)

        # Check that the original DataFrame is unchanged
        assert_frame_equal(sample_df_with_nas, original)

        # Check that the result is different from the original
        with pytest.raises(AssertionError):
            assert_frame_equal(result, original)