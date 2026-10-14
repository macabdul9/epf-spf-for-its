import pytest

from epf_spf.config import PRESETS, build_config


def test_every_preset_builds():
    for name in PRESETS:
        assert build_config(name).method == name


def test_presets_encode_the_documented_differences():
    paper, itshub = build_config("epf-paper"), build_config("epf-itshub")
    project, spf = build_config("epf-project"), build_config("spf")
    assert (paper.weight_transform, paper.weight_temperature) == ("log", 1.0)
    assert paper.resample_from_step == 2
    assert (paper.anneal_ess_threshold, paper.anneal_window) == (0.5, 0.5)
    assert (itshub.weight_transform, itshub.score_aggregation) == ("logit", "prod")
    assert (itshub.anneal_ess_threshold, itshub.anneal_window) == (0.5, 0.5)
    assert (project.weight_transform, project.weight_temperature) == ("logit", 3.0)
    assert (spf.scorer, spf.resample_rule, spf.gate_threshold) == ("probe", "ess_gate", 0.5)


def test_overrides_replace_preset_fields_and_none_keeps_them():
    config = build_config("epf-project", weight_temperature=1.0, score_aggregation=None)
    assert config.weight_temperature == 1.0
    assert config.score_aggregation == "last"


@pytest.mark.parametrize(
    "override",
    [
        {"weight_temperature": 0.0},
        {"weight_transform": "softmax"},
        {"resample_from_step": 0},
        {"anneal_window": 1.5},
        {"resample_rule": "ess_gate"},  # epf-paper has no gate threshold to gate on
    ],
)
def test_invalid_configs_are_rejected(override):
    with pytest.raises(ValueError):
        build_config("epf-paper", **override)
