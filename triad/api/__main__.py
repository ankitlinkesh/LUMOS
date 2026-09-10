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
            from triad.pipeline import Pipeline  # noqa: F401
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

        from triad.api.real_adapter import RealDemoService
        service = RealDemoService(Pipeline.demo())
    else:
        from triad.api.service import FakeDemoService
        service = FakeDemoService()

    import uvicorn

    from triad.api.app import create_app

    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
