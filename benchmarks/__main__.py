"""``python -m benchmarks`` entry point."""

from __future__ import annotations

import argparse
import platform
import sys

from benchmarks.runner import render, render_markdown, summarize
from benchmarks.suite import bench


def main() -> int:
    parser = argparse.ArgumentParser(description="koe performance benchmarks")
    parser.add_argument("--only", help="substring filter on group or benchmark name")
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown table")
    args = parser.parse_args()

    from koe.text.tokenize import mecab_available

    header = (
        f"koe benchmarks — Python {platform.python_version()} on "
        f"{platform.system()} {platform.machine()}, "
        f"MeCab {'available' if mecab_available() else 'unavailable'}"
    )

    results = bench.run(only=args.only)
    if args.markdown:
        print(f"<!-- {header} -->")
        print(render_markdown(results))
    else:
        print(header)
        print(render(results))
        print(summarize(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
