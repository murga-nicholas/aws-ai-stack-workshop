"""Separate-process entry point for restoring a Strands session."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from awsai_demo.cli import CommandParser
from awsai_demo.decision_demo import ResumeRequest, resume_from_disk
from awsai_demo.network import install_from_environment
from awsai_demo.redact import quiet_sdk_logging, redact, sanitize_exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TextIO


@quiet_sdk_logging()
def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Restore from disk and emit redacted JSON or a diagnostic."""
    output = sys.stdout if stdout is None else stdout
    diagnostic = sys.stderr if stderr is None else stderr
    try:
        install_from_environment()
        parser = CommandParser(prog="awsai-resume-worker")
        parser.add_argument("--request", required=True)
        args = parser.parse_args(argv)
        request = ResumeRequest.model_validate_json(args.request)
        report = resume_from_disk(request)
        print(
            json.dumps(redact(report.model_dump()), sort_keys=True),
            file=output,
        )
    except Exception as exc:
        print(sanitize_exception(exc), file=diagnostic)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
