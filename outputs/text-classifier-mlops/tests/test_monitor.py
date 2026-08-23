from text_classifier.monitor import analyze, population_stability_index


def test_identical_distributions_have_zero_psi():
    assert population_stability_index([0.2, 0.8], [0.2, 0.8]) == 0.0


def test_monitor_requires_minimum_events():
    config = {"monitoring": {"min_events": 2}, "serving": {}}
    report = analyze([], {}, config)
    assert report["status"] == "insufficient_data"
    assert report["retrain"] is False
