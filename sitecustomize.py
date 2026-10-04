import sys


def _ensure_utf8(stream):
    if stream is None:
        return
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


_ensure_utf8(sys.stdout)
_ensure_utf8(sys.stderr)
