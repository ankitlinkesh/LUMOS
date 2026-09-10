import importlib

from triad.llm import doctor


def test_import_does_no_io(monkeypatch, tmp_path):
    # Re-importing must not touch the real secrets file or network; if it did,
    # this would either raise (no real keys on a fresh checkout) or hang.
    importlib.reload(doctor)


def test_no_keys_prints_friendly_message_and_returns_nonzero(tmp_path, monkeypatch, capsys):
    from triad import config

    monkeypatch.setattr(config, "KEYS_FILE", tmp_path / "nope.txt")
    monkeypatch.delenv("GROQ_API_KEYS", raising=False)
    # doctor.load_keys uses the default arg bound at import time (config.KEYS_FILE
    # at import), so call main() with a patched load_keys default via env instead:
    monkeypatch.setattr(doctor, "load_keys", lambda: (_ for _ in ()).throw(
        __import__("triad.llm.keys", fromlist=["KeysMissing"]).KeysMissing(
            f"no Groq API keys found: {tmp_path / 'nope.txt'} does not exist."
        )
    ))
    code = doctor.main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "Paste" in out
    assert str(config.KEYS_FILE) in out


def test_report_with_real_keys_masks_values(tmp_path, monkeypatch, capsys):
    keys_file = tmp_path / "groq_keys.txt"
    keys_file.write_text("gsk_supersecretvalueaaaa1111\n", encoding="utf-8")

    def fake_load_keys():
        from triad.llm.keys import load_keys as real_load_keys
        return real_load_keys(path=keys_file, env={})

    monkeypatch.setattr(doctor, "load_keys", fake_load_keys)
    code = doctor.main([])
    out = capsys.readouterr().out
    assert code == 0
    assert "supersecretvalueaaaa1111" not in out
    assert "Loaded 1 key" in out
