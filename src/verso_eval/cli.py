"""Helpers shared by the command-line entry points (python -m verso_eval.<command>)."""

import sys


def configure_output():
    """Write stdout and stderr as UTF-8.

    On Windows, output redirected to a file or a pipe defaults to the ANSI code page, which cannot encode
    Vietnamese dataset text (answers, units) and makes print() fail.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
