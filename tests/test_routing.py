"""多端点模型路由测试（纯解析逻辑，离线可跑）。"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag.config as cfg # noqa: E402
from rag.config import resolve_model # noqa: E402


def test_default_endpoint(monkeypatch):
    monkeypatch.setattr(cfg, "BASE_URL", "https://default.example/v1")
    monkeypatch.setattr(cfg, "API_KEY", "sk-default")
    monkeypatch.setattr(cfg, "LLM_MODEL", "default-model")
    bare, base, key = resolve_model(None)
    assert bare == "default-model" and base == "https://default.example/v1"
    bare, base, key = resolve_model("plain-model")
    assert bare == "plain-model" and key == "sk-default"


def test_alias_routing(monkeypatch):
    monkeypatch.setenv("MYEP__BASE_URL", "https://myep.example/v1")
    monkeypatch.setenv("MYEP__API_KEY", "sk-myep")
    bare, base, key = resolve_model("some-model@myep")
    assert bare == "some-model"
    assert base == "https://myep.example/v1" and key == "sk-myep"
    # 别名大小写不敏感
    bare, base, _ = resolve_model("m2@MyEp")
    assert bare == "m2" and base == "https://myep.example/v1"


def test_missing_alias_raises(monkeypatch):
    monkeypatch.delenv("NOPE__BASE_URL", raising=False)
    monkeypatch.delenv("NOPE__API_KEY", raising=False)
    with pytest.raises(ValueError, match="NOPE__BASE_URL"):
        resolve_model("m@nope")


def test_alias_takes_last_at(monkeypatch):
    # 模型名理论上不含 @；rsplit 保证只按最后一个 @ 切
    monkeypatch.setenv("B__BASE_URL", "https://b.example/v1")
    monkeypatch.setenv("B__API_KEY", "sk-b")
    bare, base, key = resolve_model("a@b")
    assert bare == "a" and base == "https://b.example/v1" and key == "sk-b"
