import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy import stats
# Import the classes to test
# Update these imports to match your actual module structure
from components.model import MLModel

class TestMLModel:
    """Tests for the MLModel class."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        df['target'] = y
        return df

    def test_init(self):
        """Test MLModel initialization."""
        model = MLModel()
        assert isinstance(model.pipeline, Pipeline)
        assert isinstance(model.pipeline.named_steps['scaler'], StandardScaler)
        assert isinstance(model.pipeline.named_steps['model'], Ridge)

    def test_predict(self, sample_df):
        """Test MLModel predict method."""
        model = MLModel()

        # Test the predict method
        predictions, score = model.predict(sample_df)

        # Check predictions type and structure
        assert isinstance(predictions, pd.DataFrame)
        assert 'predictions' in predictions.columns
        assert len(predictions) == 20  # 20% of 100 rows for test set

        # Check score
        assert isinstance(score, float)
        assert 0 <= score <= 1  # R² score should be between 0 and 1

    def test_predict_without_target_column(self):
        """Test MLModel predict method with DataFrame missing target column."""
        model = MLModel()
        df_without_target = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6]
        })

        with pytest.raises(KeyError):
            model.predict(df_without_target)

    def test_predict_splitting(self, mock_loaded_data):
        """Test MLModel test-data corresponds to train-data"""
        df = mock_loaded_data.copy()
        df.rename(columns={'^GSPC': 'target'}, inplace=True)
        X = df.drop(columns=['target'])
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        p_value_list = []
        for ticker_name in X.columns:
            _, p_value = stats.ks_2samp(X_train[ticker_name], X_test[ticker_name])
            p_value_list.append(round(p_value, 4))
        print(p_value_list)
        assert np.min(p_value_list) > 0.05

        _, p_value = stats.ks_2samp(y_train, y_test)
        assert p_value > 0.05

