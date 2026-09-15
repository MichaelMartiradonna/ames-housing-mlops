"""Validate the unchanged control and the intended synthetic drift signal."""

from src.monitor_drift import run_monitor


def test_identical_control_has_no_drift(config):
    summary = run_monitor(config, "control", write_reports=False)
    assert summary["feature_count"] == 14
    assert summary["drifted_features"] == []
    assert summary["drift_share"] == 0
    assert summary["alert"] is False


def test_simulated_shift_triggers_expected_feature_alerts(config):
    summary = run_monitor(config, "drifted", write_reports=False)
    expected = set(config["monitoring"]["numeric_multipliers"]) | {"Neighborhood"}
    assert set(summary["drifted_features"]) == expected
    assert summary["drift_share"] == 5 / 14
    assert summary["alert"] is True
