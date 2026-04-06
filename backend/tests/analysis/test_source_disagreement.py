"""Unit tests for cross-source disagreement metrics."""

from app.analysis.source_disagreement import disagreement_metrics


def test_disagreement_requires_two_channels():
    assert disagreement_metrics({"reddit": 0.5}) == (None, None)
    assert disagreement_metrics({}) == (None, None)


def test_disagreement_range_and_std():
    r, s = disagreement_metrics({"reddit": 0.2, "news": -0.4, "twitter": 0.1})
    assert r == 0.6  # 0.2 - (-0.4)
    vals = [0.2, -0.4, 0.1]
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    assert s == round(var**0.5, 4)
