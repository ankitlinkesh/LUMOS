"""Run the TRIAD-RAG demo API server.

    python -m triad.api                # fake demo service (default)
    python -m triad.api --real         # real pipeline (requires triad.pipeline.Pipeline)
    python -m triad.api --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m triad.api", description=__doc__)
    parser.add_argument("--real", action="store_true",
                         help="use the real pipeline instead of the fake demo service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.real:
        try:
            from triad.pipeline import Pipeline
        except ImportError as exc:
            print(
                "error: --real requires triad.pipeline.Pipeline, which does not exist "
                "yet (the real pipeline is being built separately).\n"
                "  Run `python -m triad.api` without --real to use the fake demo "
                "service, or wait for the pipeline to land and finish "
                "triad/api/real_adapter.py.\n"
                f"  (import error: {exc})",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        # Building a genuinely working real pipeline needs a real LLM client
        # too -- Pipeline.ask() raises if llm is None. Constructing that here
        # (and refusing to start on any failure) is what keeps the promise in
        # real_adapter.py's meta(): the "DEMO MODE -- FAKE DATA" banner is OFF
        # in --real mode, so it must never be OFF while endpoints 500 because
        # the LLM, or the demo corpus build, never actually came up.
        try:
            from triad.llm.cache import DiskCache
            from triad.llm.client import GroqClient
            from triad.llm.keys import load_keys
            from triad.llm.limiter import RateLimiter
            from triad.pipeline import DefenseConfig

            keys = load_keys()
            llm = GroqClient(keys=keys, limiter=RateLimiter(), cache=DiskCache())
            # Stage 3 is measured now (see README's "Stage 3 -- output and
            # egress" section) -- enabling it here is what makes the demo's
            # live trace match that claim instead of contradicting it.
            # Confined to the demo server: DefenseConfig.stage3_enabled's own
            # default stays False (triad.eval harnesses read that default,
            # and every persisted results/ number was measured against it).
            # A 10-question live sample of this exact demo corpus (real Groq
            # calls, real egress checks, defended path) found egress ran on
            # 10/10 and inspect_answer blocked or rewrote 0/10 -- enabling
            # it does not visibly mangle the demo, it just makes the trace
            # show real egress decisions.
            pipeline = Pipeline.demo(llm=llm, defense=DefenseConfig(stage3_enabled=True))
        except Exception as exc:
            print(
                "error: --real could not build a working pipeline, so refusing to "
                "start (never serves with the DEMO MODE banner off and broken "
                "endpoints).\n"
                f"  cause: {type(exc).__name__}: {exc}\n"
                "  Run `python -m triad.api` without --real to use the fake demo "
                "service instead.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc

        from triad.api.real_adapter import RealDemoService
        service = RealDemoService(pipeline)
    else:
        from triad.api.service import FakeDemoService
        service = FakeDemoService()

    import uvicorn

    from triad.api.app import create_app

    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
