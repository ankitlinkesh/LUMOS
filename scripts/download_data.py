"""Download every dataset TRIAD-RAG evaluates on, into data/raw/.

Re-runnable: anything already present is skipped. Writes data/raw/MANIFEST.json
with byte sizes and sha256 so every teammate can confirm they have identical data.

    .venv/Scripts/python scripts/download_data.py            # everything
    .venv/Scripts/python scripts/download_data.py enronqa    # one source
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# (local dir, HF dataset repo, files). LLMail-Inject's per-attempt success labels
# (email.retrieved, defense.undetected, exfil.sent/destination/content) exist ONLY in
# raw_submissions_*.jsonl (1.9 GB). The labelled_unique files carry just
# {attack_attempt, reason: api_triggered | judge}, so the raw files are required for the
# "attacks that already beat the deployed defenses" subset.
HF_SOURCES = {
    "enronqa": ("MichaelR207/enron_qa_0922", [
        "data/train-00000-of-00002.parquet",
        "data/train-00001-of-00002.parquet",
        "data/dev-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
        "README.md",
    ]),
    "enron_corpus": ("MichaelR207/enron_corpus_0904", [
        "emails_adj_dedup.csv",
        "README.md",
    ]),
    "llmail_inject": ("microsoft/llmail-inject-challenge", [
        "data/labelled_unique_submissions_phase1.json",
        "data/labelled_unique_submissions_phase2.json",
        "data/raw_submissions_phase1.jsonl",
        "data/raw_submissions_phase2.jsonl",
        "data/emails_for_fp_tests.json",
        "data/scenarios.json",
        "data/system_prompt.json",
        "data/levels_descriptions.json",
        "data/objectives_descriptions.json",
        "README.md",
    ]),
}

GIT_SOURCES = {
    "PoisonedRAG": "https://github.com/thisxyz/PoisonedRAG.git",
    "BIPIA": "https://github.com/microsoft/BIPIA.git",
}

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
BEIR_SOURCES = ("nq",)  # PoisonedRAG's default corpus; hotpotqa/msmarco can be added here


def fetch_hf(name: str) -> None:
    repo, files = HF_SOURCES[name]
    dest = RAW / name
    for f in files:
        if (dest / f).exists():
            print(f"  skip  {name}/{f}")
            continue
        print(f"  get   {name}/{f}", flush=True)
        hf_hub_download(repo_id=repo, filename=f, repo_type="dataset", local_dir=dest)


def fetch_git(name: str) -> None:
    dest = RAW / name
    if dest.exists():
        print(f"  skip  {name} (already cloned)")
        return
    print(f"  clone {name}", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", GIT_SOURCES[name], str(dest)], check=True)


def fetch_beir(name: str) -> None:
    dest = RAW / "beir"
    if (dest / name / "corpus.jsonl").exists():
        print(f"  skip  beir/{name}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.zip"
    if not archive.exists():
        print(f"  get   beir/{name}.zip", flush=True)
        part = archive.with_suffix(".zip.part")
        with requests.get(BEIR_URL.format(name=name), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        part.rename(archive)
    print(f"  unzip beir/{name}.zip", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> None:
    entries = []
    for p in sorted(RAW.rglob("*")):
        if not p.is_file() or ".git" in p.parts or ".cache" in p.parts or p.name == "MANIFEST.json":
            continue
        entries.append({"path": p.relative_to(RAW).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})
    (RAW / "MANIFEST.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"  manifest: {len(entries)} files, {sum(e['bytes'] for e in entries) / 1e9:.2f} GB")


def main(selected: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    everything = [*HF_SOURCES, *GIT_SOURCES, *(f"beir:{n}" for n in BEIR_SOURCES)]
    for name in selected or everything:
        print(f"[{name}]", flush=True)
        if name in HF_SOURCES:
            fetch_hf(name)
        elif name in GIT_SOURCES:
            fetch_git(name)
        elif name.startswith("beir:"):
            fetch_beir(name.split(":", 1)[1])
        else:
            sys.exit(f"unknown source {name!r}; choose from {everything}")
    write_manifest()


if __name__ == "__main__":
    main(sys.argv[1:])
