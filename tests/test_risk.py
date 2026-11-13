from copy import deepcopy
import json

import pytest
import yaml

from ml import risk
from ml.calibrate_risk import evaluate, violation
from ml.rules import RULES


@pytest.fixture
def config():
    return dict(weights=dict(sms=.5, url=.3, social_engineering=.2), levels=dict(medium=35, high=70),
                overrides=dict(code_request_min=70, combined_sms_threshold=.8, combined_min=85, url_mask_min=60))


def test_absent_url_excludes_its_weight_but_zero_probability_includes_it(config):
    absent = risk.compute_risk(.7, None, [], [], config=config)
    present = risk.compute_risk(.7, 0, [], [], config=config)
    assert absent.risk_score == 50
    assert present.risk_score == 35


def test_no_groups_no_flags_has_neutral_reason_and_recommendation(config):
    result = risk.compute_risk(0, None, [], [], config=config)
    assert (result.risk_score, result.risk_level, result.category) == (0, "LOW", "SAFE")
    assert result.explanation and "Выраженных" in result.explanation[0]
    assert result.recommendation and "вероятностный" in result.disclaimer
    assert result.to_dict()["risk_score"] == 0


@pytest.mark.parametrize("groups,expected", [(["URGENCY"],10), (["URGENCY","PRIZE"],19),
    (["URGENCY","PRIZE","BANK_CARD"],29), (["URGENCY","PRIZE","BANK_CARD","ACCOUNT_BLOCK"],29),
    (["URGENCY","URGENCY"],10)])
def test_unique_groups_and_saturation_at_three(config, groups, expected):
    assert risk.compute_risk(0, None, groups, [], config=config).risk_score == expected


def test_code_request_has_minimum_seventy(config):
    result = risk.compute_risk(0, None, ["CODE_REQUEST"], [], config=config)
    assert result.risk_score == 70 and result.risk_level == "HIGH"
    assert any("Правило CODE_REQUEST" in reason for reason in result.explanation)


def test_combined_override_uses_metadata_t2_not_hardcoded_number(config, tmp_path, monkeypatch):
    path = tmp_path / "url_meta.json"
    monkeypatch.setattr(risk, "URL_META_PATH", path)
    path.write_text(json.dumps({"t2":.4}), encoding="utf-8")
    assert risk.compute_risk(.8, .4, [], [], config=config).risk_score == 85
    path.write_text(json.dumps({"t2":.5}), encoding="utf-8")
    assert risk.compute_risk(.8, .4, [], [], config=config).risk_score == 52


@pytest.mark.parametrize("flags,expected", [({"is_ip":1},60), ({"has_punycode":1},8),
    ({"is_brand_similar":1},8), ({"has_punycode":1,"is_brand_similar":1},60),
    ([{"has_punycode":1},{"is_brand_similar":1}],8)])
def test_ip_and_same_url_punycode_brand_masking(config, flags, expected):
    assert risk.compute_risk(.1, .1, [], flags, config=config).risk_score == expected


def test_all_overrides_use_maximum_not_sum_and_ignore_order(config):
    flags = [{"is_ip":1}, {"has_punycode":1,"is_brand_similar":1}]
    a = risk.compute_risk(.8,.4,["URGENCY","CODE_REQUEST"],flags,config=config,t2=.4)
    b = risk.compute_risk(.8,.4,["CODE_REQUEST","URGENCY"],flags[::-1],config=config,t2=.4)
    assert a.risk_score == b.risk_score == 85
    assert a.category == b.category == "FRAUDULENT"


@pytest.mark.parametrize("value,score,level", [(34,34,"LOW"),(34.49,34,"LOW"),(34.5,35,"MEDIUM"),
    (35,35,"MEDIUM"),(69,69,"MEDIUM"),(69.49,69,"MEDIUM"),(69.5,70,"HIGH"),(70,70,"HIGH"),(100,100,"HIGH")])
def test_boundaries_and_half_up_rounding(config, value, score, level):
    config["weights"] = dict(sms=1.,url=0.,social_engineering=0.)
    result = risk.compute_risk(value/100,None,[],[],config=config)
    assert result.risk_score == score and result.risk_level == level


def test_yaml_changes_weights_levels_and_override_without_python_changes(config, tmp_path, monkeypatch):
    path = tmp_path / "risk.yaml"
    monkeypatch.setattr(risk, "CONFIG_PATH", path)
    path.write_text(yaml.safe_dump(config),encoding="utf-8")
    assert risk.compute_risk(.5,None,[],[]).risk_score == 36
    config["weights"] = dict(sms=.1,url=.3,social_engineering=.9)
    config["levels"] = dict(medium=3,high=80)
    config["overrides"]["code_request_min"] = 90
    path.write_text(yaml.safe_dump(config),encoding="utf-8")
    assert risk.compute_risk(.5,None,[],[]).risk_score == 5
    assert risk.compute_risk(.5,None,[],[]).risk_level == "MEDIUM"
    assert risk.compute_risk(0,None,["CODE_REQUEST"],[]).risk_score == 90


@pytest.mark.parametrize("section,key,value", [("weights","sms",-1),("weights","url",float("inf")),
    ("weights","sms",True),("levels","medium",70),("levels","high",101),("levels","medium",0),
    ("levels","high",69.5),("overrides","combined_sms_threshold",1.1),("overrides","code_request_min",-1),
    ("overrides","combined_min",101),("overrides","url_mask_min",60.1)])
def test_invalid_configuration_is_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        risk.validate_config(config)


def test_zero_denominator_is_rejected_without_urls(config):
    config["weights"] = dict(sms=0,url=1,social_engineering=0)
    with pytest.raises(ValueError):
        risk.validate_config(config)


@pytest.mark.parametrize("value", [-.1,1.1,float("nan"),float("inf"),None,"0.5",True])
def test_invalid_sms_probability(config, value):
    with pytest.raises(ValueError):
        risk.compute_risk(value,None,[],[],config=config)


@pytest.mark.parametrize("value", [-.1,1.1,float("nan"),"0.5",False])
def test_invalid_url_probability(config, value):
    with pytest.raises(ValueError):
        risk.compute_risk(.1,value,[],[],config=config)


def test_reasons_sorted_by_contribution_and_limited_to_six(config):
    result = risk.compute_risk(.6,.9,["URGENCY"],[],config=config)
    assert "SMS-модель" in result.explanation[0]  # 30 баллов > 27 баллов URL.
    assert "URL-модель" in result.explanation[1]
    many = risk.compute_risk(.8,.9,list(RULES),{"is_ip":1,"has_at":1,"is_shortener":1,
                             "has_punycode":1,"is_brand_similar":1},config=config,t2=.4)
    assert 1 <= len(many.explanation) <= 6
    assert many.recommendation


def test_calibration_evaluation_uses_medium_recall_and_safe_high_fraction(config):
    rows = [dict(label=0,p_sms=0,p_url_max=None,se_groups=[],url_flags=[]),
            dict(label=0,p_sms=0,p_url_max=None,se_groups=["CODE_REQUEST"],url_flags=[]),
            dict(label=1,p_sms=.7,p_url_max=None,se_groups=[],url_flags=[]),
            dict(label=1,p_sms=0,p_url_max=None,se_groups=[],url_flags=[])]
    m = evaluate(rows,config,.5)
    assert m["recall_medium_high"] == .5 and m["false_high_rate"] == .5
    assert m["detected_medium_high"] == m["false_high"] == 1
    assert not m["meets_targets"]
    assert violation(dict(en_validation=m,ru_kk_development=m)) > 0
