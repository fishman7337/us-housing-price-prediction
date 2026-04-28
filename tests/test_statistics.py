from us_housing_price_prediction.data import load_housing_data
from us_housing_price_prediction.statistics import (
    paired_error_t_test,
    run_feature_significance_tests,
)


def test_feature_significance_tests_return_valid_p_values() -> None:
    df = load_housing_data()
    tests = run_feature_significance_tests(df)

    assert {"feature", "test", "statistic", "p_value", "significant_at_alpha"}.issubset(
        tests.columns
    )
    assert tests["p_value"].between(0, 1).all()
    assert {"city", "renovation_status"}.issubset(set(tests["feature"]))


def test_paired_error_t_test_reports_error_delta() -> None:
    result = paired_error_t_test(
        y_true=[100.0, 200.0, 300.0],
        candidate_predictions=[100.0, 210.0, 290.0],
        baseline_predictions=[150.0, 250.0, 250.0],
    )

    assert result["test"] == "paired_t_test_absolute_error"
    assert result["mean_error_delta"] < 0
    assert result["p_value"] is not None
