"""
Unit Tests for the Demand Forecasting Pipeline.
Run with: pytest tests/ -v
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_demand_data():
    """Create a small synthetic demand dataset for testing."""
    dates = pd.date_range("2023-01-01", periods=365, freq="D")
    products = ["ProductA", "ProductB"]
    rows = []
    for product in products:
        for d in dates:
            rows.append({
                "date": d,
                "product": product,
                "demand": np.random.randint(50, 200),
                "price": round(np.random.uniform(5, 50), 2),
                "promotion": np.random.choice([0, 1], p=[0.85, 0.15]),
                "day_of_week": d.dayofweek,
                "month": d.month,
                "is_weekend": int(d.dayofweek >= 5),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_demand_csv(sample_demand_data, tmp_path):
    """Save sample data to a CSV and return the path."""
    csv_path = tmp_path / "demand_data.csv"
    sample_demand_data.to_csv(csv_path, index=False)
    return str(csv_path)


# ─── Data Loading Tests ──────────────────────────────────────────────────────

class TestDataLoading:
    def test_generated_data_file_exists(self):
        """Check that the data file exists after a pipeline run."""
        # This test only passes after train.py has been run
        if not os.path.exists(config.GENERATED_DATA_FILE):
            pytest.skip("Data file not found - run train.py first")
        df = pd.read_csv(config.GENERATED_DATA_FILE)
        assert len(df) > 0
        assert "date" in df.columns
        assert "product" in df.columns
        assert "demand" in df.columns

    def test_data_has_expected_columns(self, sample_demand_data):
        expected = ["date", "product", "demand", "price", "promotion",
                     "day_of_week", "month", "is_weekend"]
        for col in expected:
            assert col in sample_demand_data.columns, f"Missing column: {col}"

    def test_data_has_no_nulls(self, sample_demand_data):
        assert sample_demand_data.isnull().sum().sum() == 0

    def test_demand_is_positive(self, sample_demand_data):
        assert (sample_demand_data["demand"] >= 0).all()

    def test_price_is_positive(self, sample_demand_data):
        assert (sample_demand_data["price"] > 0).all()

    def test_promotion_is_binary(self, sample_demand_data):
        assert set(sample_demand_data["promotion"].unique()).issubset({0, 1})


# ─── Preprocessing Tests ─────────────────────────────────────────────────────

class TestPreprocessing:
    def test_add_features(self, sample_demand_data):
        from src.preprocessing import add_features
        df = add_features(sample_demand_data)
        # Check that lag features were added
        assert "lag_1" in df.columns
        assert "lag_7" in df.columns
        assert "rolling_mean_7" in df.columns

    def test_split_data(self, sample_demand_data):
        from src.preprocessing import add_features, split_data
        df = add_features(sample_demand_data)
        train, test = split_data(df, product="ProductA")
        assert len(train) > 0
        assert len(test) > 0
        assert train["date"].max() <= test["date"].min()

    def test_feature_columns(self, sample_demand_data):
        from src.preprocessing import add_features, get_feature_columns
        df = add_features(sample_demand_data)
        cols = get_feature_columns()
        assert isinstance(cols, list)
        assert len(cols) > 0
        assert "demand" not in cols  # target should not be a feature
        assert "date" not in cols
        assert "product" not in cols


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:
    def test_xgboost_train_predict(self, sample_demand_data, monkeypatch):
        from src.preprocessing import add_features, split_data, get_feature_columns
        from src.models.xgboost_model import train_xgboost, predict_xgboost

        monkeypatch.setattr(config, "TEST_SIZE_DAYS", 30)
        df = add_features(sample_demand_data)
        train, test = split_data(df, product="ProductA")
        train = train.dropna()
        test = test.dropna()
        feature_cols = get_feature_columns()

        model = train_xgboost(train, feature_cols, product_name="test_product")
        preds = predict_xgboost(model, test, feature_cols)

        assert len(preds) == len(test)
        assert (preds >= 0).all()  # predictions should be non-negative

    def test_xgboost_feature_importance(self, sample_demand_data, monkeypatch):
        from src.preprocessing import add_features, split_data, get_feature_columns
        from src.models.xgboost_model import train_xgboost, get_feature_importance

        monkeypatch.setattr(config, "TEST_SIZE_DAYS", 30)
        df = add_features(sample_demand_data)
        train, test = split_data(df, product="ProductA")
        train = train.dropna()
        feature_cols = get_feature_columns()

        model = train_xgboost(train, feature_cols, product_name="test_product")
        importance = get_feature_importance(model, feature_cols)

        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == len(feature_cols)
        assert (importance["importance"] >= 0).all()

    def test_ensemble_weights(self):
        from src.models.ensemble_model import find_optimal_weights

        y_true = np.array([100, 150, 200, 250, 300])
        pred_a = np.array([110, 140, 210, 240, 310])
        pred_b = np.array([90, 160, 190, 260, 290])

        alpha, rmse = find_optimal_weights(y_true, pred_a, pred_b)
        assert 0 <= alpha <= 1
        assert rmse >= 0

        # Ensemble should be at least as good as the worse model
        rmse_a = np.sqrt(np.mean((y_true - pred_a) ** 2))
        rmse_b = np.sqrt(np.mean((y_true - pred_b) ** 2))
        assert rmse <= max(rmse_a, rmse_b)


# ─── Config Tests ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_products_list_not_empty(self):
        assert len(config.PRODUCTS) > 0

    def test_directories_exist(self):
        assert os.path.isdir(config.DATA_DIR)
        assert os.path.isdir(config.MODEL_DIR)
        assert os.path.isdir(config.RESULTS_DIR)

    def test_data_mode_flag(self):
        assert isinstance(config.USE_REAL_DATA, bool)


# ─── Report Generator Tests ──────────────────────────────────────────────────

class TestReportGenerator:
    def test_generate_report_returns_html(self):
        from src.report_generator import generate_report

        if not os.path.exists(config.GENERATED_DATA_FILE):
            pytest.skip("Data file not found - run train.py first")

        html, filename = generate_report()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert filename.endswith(".html")

    def test_report_contains_sections(self):
        from src.report_generator import generate_report

        if not os.path.exists(config.GENERATED_DATA_FILE):
            pytest.skip("Data file not found - run train.py first")

        html, _ = generate_report()
        assert "Dataset Overview" in html
        assert "Model Performance" in html
