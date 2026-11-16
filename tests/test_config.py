from copy import deepcopy

import pytest
import yaml

from glorp.config import ConfigError, load_config

BASE = {
    "provider_list": [
        {
            "name": "local",
            "base_url": "localhost:11434/v1",
            "protocol": "openai-completions",
            "api_key": "API_KEY",
            "models": [{"id": "actual/id", "name": "Display"}],
        }
    ]
}


def load(tmp_path, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    return load_config(path)


def test_defaults_literals_and_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "NOT_THE_LITERAL")
    monkeypatch.chdir("/")
    config = load(tmp_path, BASE)
    assert config.providers[0].api_key == "API_KEY"
    assert config.providers[0].base_url == "http://localhost:11434/v1"
    assert config.database == tmp_path / "data/glorp.sqlite3"
    assert config.routes["local/actual/id"].name == "local"
    assert "actual/id" not in config.routes
    assert "API_KEY" not in repr(config)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1"),
        ("[::1]:8001/v1/", "http://[::1]:8001/v1"),
        ("https://remote.invalid/v1", "https://remote.invalid/v1"),
    ],
)
def test_urls(tmp_path, url, expected):
    raw = deepcopy(BASE)
    raw["provider_list"][0]["base_url"] = url
    assert load(tmp_path, raw).providers[0].base_url == expected


@pytest.mark.parametrize(
    "url",
    [
        "remote.invalid/v1",
        "http://remote.invalid/v1",
        "http://localhost:0/v1",
        "http://localhost:99999",
        "https://user:SECRET@remote.invalid/v1",
        "https://remote.invalid/v1?api_key=SECRET",
        "ftp://localhost/v1",
    ],
)
def test_bad_urls(tmp_path, url):
    raw = deepcopy(BASE)
    raw["provider_list"][0]["base_url"] = url
    with pytest.raises(ConfigError) as exc:
        load(tmp_path, raw)
    assert "SECRET" not in str(exc.value)


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(provider_list=[]),
        lambda x: x["provider_list"][0].update(protocol="anthropic"),
        lambda x: x["provider_list"][0].pop("api_key"),
        lambda x: x["provider_list"][0].update(models=[{"id": "x"}]),
        lambda x: x["provider_list"].append(deepcopy(x["provider_list"][0])),
        lambda x: x["provider_list"][0]["models"].append(deepcopy(x["provider_list"][0]["models"][0])),
        lambda x: x["provider_list"][0].update(name="nested/provider"),
        lambda x: x.update(server={"host": "0.0.0.0"}),
        lambda x: x.update(server={"host": "remote.invalid"}),
        lambda x: x.update(server={"port": True}),
        lambda x: x.update(server={"port": 65536}),
        lambda x: x.update(server={"port": "8000"}),
        lambda x: x.update(server={"unknown": 1}),
    ],
)
def test_bad_schema(tmp_path, change):
    raw = deepcopy(BASE)
    change(raw)
    with pytest.raises(ConfigError):
        load(tmp_path, raw)


@pytest.mark.parametrize("contents", ["[", "!!python/object/apply:os.system ['false']", "null", "[]"])
def test_bad_yaml(tmp_path, contents):
    path = tmp_path / "bad.yaml"
    path.write_text(contents)
    with pytest.raises(ConfigError):
        load_config(path)


def test_same_upstream_model_allowed_across_providers(tmp_path):
    raw = deepcopy(BASE)
    raw["provider_list"].append(dict(raw["provider_list"][0], name="other"))
    config = load(tmp_path, raw)
    assert set(config.routes) == {"local/actual/id", "other/actual/id"}
    assert config.routes["local/actual/id"].name == "local"
    assert config.routes["other/actual/id"].name == "other"
    assert all(p.models[0].id == "actual/id" for p in config.providers)


def test_environment_and_remote_binding(tmp_path, monkeypatch):
    raw = deepcopy(BASE)
    raw["provider_list"][0]["api_key"] = "${UPSTREAM_TEST_KEY}"
    for value in (None, "", "   "):
        if value is None:
            monkeypatch.delenv("UPSTREAM_TEST_KEY", raising=False)
        else:
            monkeypatch.setenv("UPSTREAM_TEST_KEY", value)
        with pytest.raises(ConfigError):
            load(tmp_path, raw)
    monkeypatch.setenv("UPSTREAM_TEST_KEY", "secret-upstream")
    raw["server"] = {"host": "0.0.0.0", "api_key_env": "INBOUND_TEST_KEY"}
    with pytest.raises(ConfigError):
        load(tmp_path, raw)
    monkeypatch.setenv("INBOUND_TEST_KEY", "secret-inbound")
    config = load(tmp_path, raw)
    assert config.api_key == "secret-inbound"
    assert config.providers[0].api_key == "secret-upstream"
    assert "secret" not in repr(config)
