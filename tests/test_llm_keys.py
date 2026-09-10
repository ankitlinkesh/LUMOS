import logging

import pytest

from triad.llm.keys import ApiKey, KeysMissing, load_keys


def write_keys(tmp_path, text):
    p = tmp_path / "groq_keys.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_plain_keys(tmp_path):
    p = write_keys(tmp_path, "gsk_aaaaaaaaaaaaaaaa1234\ngsk_bbbbbbbbbbbbbbbb5678\n")
    keys = load_keys(path=p, env={})
    assert len(keys) == 2
    assert keys[0].value == "gsk_aaaaaaaaaaaaaaaa1234"


def test_ignores_comments_and_blanks_and_placeholder(tmp_path):
    p = write_keys(
        tmp_path,
        "# this is a comment\n\ngsk_realkeyvalueaaaa1111\nPASTE_KEY_HERE\n   \n",
    )
    keys = load_keys(path=p, env={})
    assert len(keys) == 1
    assert keys[0].value == "gsk_realkeyvalueaaaa1111"


def test_label_and_org_parsed(tmp_path):
    p = write_keys(tmp_path, "gsk_xxxxxxxxxxxxxxxx0001 prod org=teamA\n")
    keys = load_keys(path=p, env={})
    assert keys[0].label == "prod"
    assert keys[0].org == "teamA"


def test_org_only_no_label(tmp_path):
    p = write_keys(tmp_path, "gsk_xxxxxxxxxxxxxxxx0002 org=teamB\n")
    keys = load_keys(path=p, env={})
    assert keys[0].label is None
    assert keys[0].org == "teamB"


def test_malformed_key_rejected(tmp_path):
    p = write_keys(tmp_path, "not_a_real_key_at_all\n")
    with pytest.raises(ValueError, match="line 1"):
        load_keys(path=p, env={})


def test_no_file_raises_keys_missing_with_path(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(KeysMissing) as ei:
        load_keys(path=missing, env={})
    assert str(missing) in str(ei.value)


def test_empty_file_raises_keys_missing(tmp_path):
    p = write_keys(tmp_path, "# only comments\nPASTE_KEY_HERE\n")
    with pytest.raises(KeysMissing):
        load_keys(path=p, env={})


def test_env_keys_comma_separated(tmp_path):
    missing = tmp_path / "unused.txt"  # file need not exist when env wins
    keys = load_keys(path=missing, env={"GROQ_API_KEYS": "gsk_envkeyaaaaaaaaaa1,gsk_envkeybbbbbbbbbb2"})
    assert len(keys) == 2
    assert keys[1].value == "gsk_envkeybbbbbbbbbb2"


def test_env_keys_with_label_and_org(tmp_path):
    missing = tmp_path / "unused.txt"
    keys = load_keys(path=missing, env={"GROQ_API_KEYS": "gsk_envkeyaaaaaaaaaa1 mine org=orgX"})
    assert keys[0].org == "orgX"
    assert keys[0].label == "mine"


def test_same_org_keys_share_bucket(tmp_path):
    p = write_keys(
        tmp_path,
        "gsk_keyoneaaaaaaaaaaaa1 org=shared\ngsk_keytwobbbbbbbbbbbb2 org=shared\n",
    )
    keys = load_keys(path=p, env={})
    assert keys[0].bucket == keys[1].bucket


def test_keys_without_org_have_distinct_buckets(tmp_path):
    p = write_keys(tmp_path, "gsk_keyoneaaaaaaaaaaaa1\ngsk_keytwobbbbbbbbbbbb2\n")
    keys = load_keys(path=p, env={})
    assert keys[0].bucket != keys[1].bucket


def test_repr_and_str_mask_the_value(tmp_path):
    p = write_keys(tmp_path, "gsk_supersecretvalue9999\n")
    keys = load_keys(path=p, env={})
    key = keys[0]
    assert "supersecretvalue9999" not in repr(key)
    assert "supersecretvalue9999" not in str(key)
    assert "supersecretvalue9999" not in f"{key}"
    assert "supersecretvalue9999" not in f"{(key,)}"
    assert key.value.endswith(repr(key)[-6:-2]) or "9999" in repr(key)


def test_keys_missing_message_never_contains_file_body(tmp_path):
    p = write_keys(tmp_path, "gsk_thisshouldneverleak12\n# comment body text\n")
    # Force a failure path (env empty, but point at an empty file instead) and check
    # the exception message never contains any line from a real keys file.
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(KeysMissing) as ei:
        load_keys(path=empty, env={})
    assert "thisshouldneverleak12" not in str(ei.value)
    assert "comment body text" not in str(ei.value)
