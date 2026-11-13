import json
import socket
from urllib import request

import pandas as pd
import pytest
import requests

from ml import url_dataset as module
from ml.url_features import extract_url_features


def raw_frame(urls, labels):
    return pd.DataFrame({"URL": urls, "label": labels})


def test_labels_are_reversed_and_source_features_are_ignored():
    raw = raw_frame(["https://example.com", "http://example.org"], [1, 0])
    raw["URLLength"] = [-999, -999]
    clean, _, _ = module.clean_dataset(raw)
    assert clean.label.tolist() == [0, 1]
    assert "URLLength" not in clean
    assert clean.source_id.tolist() == [0, 1]


@pytest.mark.parametrize("labels", [[0, 2], [1, ""], [1, -1]])
def test_invalid_labels_are_not_silently_remapped(labels):
    with pytest.raises(ValueError, match="метки"):
        module.clean_dataset(raw_frame(["example.com", "example.org"], labels))


def test_empty_invalid_and_valid_urls_have_stage_counts():
    raw = raw_frame(["", "  ", None, "ftp://example.com", "http://999.0.0.1", "example.com"], [1] * 6)
    clean, stages, reasons = module.clean_dataset(raw)
    assert clean.url.tolist() == ["http://example.com"]
    assert [s["removed"] for s in stages] == [3, 2, 0, 0]
    assert sum(reasons.values()) == 2
    assert stages[-1]["class_counts"] == {"0": 1, "1": 0}


def test_hostname_normalization_does_not_lower_path_query_userinfo():
    url = "HTTPS://User:Secret@LOGIN.Example.COM:8080/Path/AB?Token=XY#Fragment"
    assert module.normalize_url(url) == "https://User:Secret@login.example.com:8080/Path/AB?Token=XY#Fragment"
    assert module.normalize_url("https://[2001:DB8::1]:443/AB?q=XY") == "https://[2001:db8::1]:443/AB?q=XY"


def test_duplicate_urls_after_hostname_normalization_keep_first_id():
    clean, stages, _ = module.clean_dataset(raw_frame(
        ["https://EXAMPLE.com/AB?q=X", "https://example.COM/AB?q=X", "https://example.com/ab?q=X"], [1, 1, 1]))
    assert clean.source_id.tolist() == [0, 2]
    assert stages[-1]["removed"] == 1


def test_all_conflicting_rows_are_removed_before_deduplication():
    clean, stages, _ = module.clean_dataset(raw_frame(
        ["https://EXAMPLE.com", "https://example.com", "https://example.com", "https://example.org"], [0, 1, 1, 1]))
    assert clean.url.tolist() == ["https://example.org"]
    assert stages[-2]["removed"] == 3
    assert stages[-1]["removed"] == 0


def test_www_correction_can_create_conflicts_and_preserves_raw_url():
    assert module.normalize_url("https://www.example.com/?Token=XY", strip_www=True) == "https://example.com?Token=XY"
    assert module.normalize_url("https://www.com/A", strip_www=True) == "https://www.com/A"
    assert module.normalize_url("https://www.gov.uk/A", strip_www=True) == "https://www.gov.uk/A"
    for url in ("https://www.example.com/A", "https://www.gov.uk/A", "https://a.example.test"):
        assert module.domain_group(url) == module.domain_group(module.normalize_url(url, strip_www=True))
    clean, _, _ = module.clean_dataset(raw_frame(["https://www.example.com/A", "https://example.com/A"], [0, 1]))
    originals = clean.url_original.tolist()
    clean["url"] = clean.url.map(lambda u: module.normalize_url(u, strip_www=True))
    assert clean.url_original.tolist() == originals
    stages = []
    assert module.remove_conflicts_and_duplicates(clean, stages).empty
    assert stages[0]["removed"] == 2


def test_group_registered_domain_idna_and_ip_canonicalization():
    assert module.domain_group("https://a.b.example.co.uk/A") == "example.co.uk"
    assert module.domain_group("https://a.example.test") == "example.test"
    assert module.domain_group("http://192.0.2.1:8080") == "192.0.2.1"
    assert module.domain_group("http://[2001:0db8:0:0:0:0:0:1]") == "2001:db8::1"
    assert module.domain_group("https://пример.рф") == module.domain_group("https://" + "пример.рф".encode("idna").decode("ascii"))


def grouped_frame():
    urls = [f"https://{sub}.site{i}.test/Path" for i in range(30) for sub in ("a", "b")]
    clean, _, _ = module.clean_dataset(raw_frame(urls, [i % 2 for i in range(len(urls))]))
    clean["group"] = clean.url.map(module.domain_group)
    return clean


def test_group_split_is_reproducible_disjoint_and_complete():
    frame = grouped_frame()
    splits = module.split_dataset(frame)
    assert splits == module.split_dataset(frame)
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [42, 8, 10]
    groups = [set(frame.iloc[splits[k]].group) for k in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sorted(i for part in splits.values() for i in part) == list(range(60))


def test_overlap_guard_rejects_domains_even_when_positions_do_not_overlap():
    frame = grouped_frame()
    bad = {"train": list(range(0, 60, 2)), "validation": list(range(1, 30, 2)), "test": list(range(31, 60, 2))}
    with pytest.raises(ValueError, match="домены"):
        module.assert_no_group_overlap(frame, bad)


def test_stratified_sampling_is_reproducible():
    frame = grouped_frame()
    a, b = module.sample_dataset(frame, size=40), module.sample_dataset(frame, size=40)
    pd.testing.assert_frame_equal(a, b)
    assert a.label.value_counts().to_dict() == {0: 20, 1: 20}
    with pytest.raises(ValueError, match="строк"):
        module.sample_dataset(frame, size=100)


def test_parquet_round_trip_feature_order_and_saved_split(tmp_path):
    frame = grouped_frame()
    splits = module.split_dataset(frame)
    table = module.build_feature_table(frame, splits)
    expected = list(extract_url_features("https://example.com"))
    assert table.columns[:22].tolist() == expected
    for key, value in extract_url_features(frame.url.iloc[0]).items():
        assert table[key].iloc[0] == value
    module.save_dataset(table, splits, tmp_path)
    restored = pd.read_parquet(tmp_path / "url_features.parquet")
    pd.testing.assert_frame_equal(table, restored)
    assert json.loads((tmp_path / "url_split.json").read_text()) == splits
    for name, positions in splits.items():
        assert table.iloc[positions].split.eq(name).all()


def test_artifact_profile_describes_collection_format():
    frame, _, _ = module.clean_dataset(raw_frame(["https://www.example.com/", "example.org/Login?x=1"], [1, 0]))
    profile = module.artifact_profile(frame)
    assert profile["0"]["www"] == profile["0"]["scheme_present"] == 1
    assert profile["1"]["www"] == profile["1"]["scheme_present"] == 0
    assert profile["1"]["has_path"] == profile["1"]["has_query"] == 1


def test_local_preparation_never_visits_dataset_urls(tmp_path, monkeypatch):
    raw = raw_frame([f"https://{'www.' if i % 2 else ''}site{i}.test/A?Token=XY" for i in range(30)],
                    [i % 2 for i in range(30)])
    raw["WebsiteContentFeature"] = -999
    source = tmp_path / "source.csv"
    raw.to_csv(source, index=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Подготовка локального CSV не должна обращаться к сети")

    monkeypatch.setattr(module, "urlopen", forbidden)
    monkeypatch.setattr(request, "urlopen", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    meta = module.prepare_dataset(source, tmp_path / "out", size=30, strip_www=True)
    assert meta["selected_rows"] == 30 and meta["group_overlap"] == 0
    table = pd.read_parquet(tmp_path / "out/url_features.parquet")
    assert "WebsiteContentFeature" not in table
    assert not table.url.str.contains("//www.", regex=False).any()
    assert table.url.str.endswith("/A?Token=XY").all()
