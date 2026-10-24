"""Tests du client LLM (openai mocke) et de l'export d'annotation (dans tmp_path)."""
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.annotation import agreement, export
from biasdetect.llm import client


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(client, "_load_dotenv", lambda: None)
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _install_fake_openai(monkeypatch, payload):
    import openai

    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content=payload)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return FakeOpenAI, calls


# ---------- llm/client.py ----------

def test_load_config_has_expected_sections():
    cfg = client.load_config()
    assert {"models", "judge", "generators"} <= set(cfg)
    assert "groq" in cfg["models"]


def test_available_models_only_when_key_is_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    models = client.available_models()
    assert set(models) == {"groq"}
    assert models["groq"] == client.load_config()["models"]["groq"]


def test_available_models_empty_without_keys():
    assert client.available_models() == {}


def test_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        client.LLMClient("unknown", "some-model")


def test_llm_client_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.LLMClient("openai", "gpt-4o-mini")


def test_chat_json_with_mocked_openai(monkeypatch):
    payload = '{"label": 1, "confidence": 0.9}'
    fake_cls, calls = _install_fake_openai(monkeypatch, payload)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    llm = client.LLMClient("openai", "gpt-4o-mini")
    assert isinstance(llm._client, fake_cls)
    assert llm._client.kwargs == {"api_key": "fake-key", "base_url": None}

    out = llm.chat_json("system prompt", "user prompt")
    assert out == {"label": 1, "confidence": 0.9}
    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_chat_json_parses_markdown_fenced_payload(monkeypatch):
    _install_fake_openai(monkeypatch, '```json\n{"label": 0}\n```')
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    assert client.LLMClient("openai", "gpt-4o-mini").chat_json("s", "u") == {"label": 0}


# ---------- annotation/export.py ----------

def test_export_writes_sheets_in_tmp_path(monkeypatch, tmp_path):
    src = tmp_path / "generations.csv"
    rows = [
        {"domain": domain, "generator": gen, "text": f"{domain}-{gen}-{k}"}
        for domain in ("gender", "race")
        for gen in ("gpt", "llama")
        for k in range(2)
    ]
    pd.DataFrame(rows).to_csv(src, index=False)
    out_dir = tmp_path / "annotation"

    monkeypatch.setattr(export, "SRC", src)
    monkeypatch.setattr(export, "OUT", out_dir)
    monkeypatch.setattr(sys, "argv", ["export", "--n", "4"])

    export.main()

    master = pd.read_csv(out_dir / "items_master.csv")
    assert len(master) == 4
    assert list(master["item_id"]) == ["it0000", "it0001", "it0002", "it0003"]
    assert {"item_id", "text"} <= set(master.columns)

    for name in ("A1", "A2"):
        sheet = pd.read_csv(out_dir / f"to_annotate_{name}.csv", keep_default_na=False)
        assert list(sheet.columns) == ["item_id", "text", "label", "category", "comment"]
        assert (sheet["label"] == "").all()
        assert set(sheet["item_id"]) == set(master["item_id"])


def test_export_then_agreement_keeps_group_columns(monkeypatch, tmp_path):
    src = tmp_path / "generations.csv"
    rows = [
        {"domain": domain, "generator": gen, "text": f"{domain}-{gen}-{k}"}
        for domain in ("gender", "race")
        for gen in ("gpt", "llama")
        for k in range(2)
    ]
    pd.DataFrame(rows).to_csv(src, index=False)
    ann = tmp_path / "annotation"

    monkeypatch.setattr(export, "SRC", src)
    monkeypatch.setattr(export, "OUT", ann)
    monkeypatch.setattr(sys, "argv", ["export", "--n", "4"])
    export.main()

    master = pd.read_csv(ann / "items_master.csv")
    assert {"item_id", "domain", "generator", "text"} <= set(master.columns)

    for name, labels in (("A1", ["0", "1", "0", "0"]), ("A2", ["0", "1", "0", "1"])):
        sheet = pd.read_csv(ann / f"to_annotate_{name}.csv", keep_default_na=False)
        sheet["label"] = labels
        sheet.to_csv(ann / f"to_annotate_{name}.csv", index=False)

    monkeypatch.setattr(agreement, "ANN", ann)
    monkeypatch.setattr(agreement, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["agreement"])
    agreement.main()

    assert (ann / "disagreements.csv").exists()
