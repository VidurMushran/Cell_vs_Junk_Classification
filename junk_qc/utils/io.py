import os
import re
from pathlib import Path
from typing import List


def ensure_dir(path: str) -> None:
    """Create directory `path` (and parents) if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def pretty_time(seconds: float) -> str:
    """Return a human-readable string for a duration in seconds."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = int(seconds - 60 * minutes)
    return f"{minutes}m{rem}s"


def find_h5s(root: str, subfolder: str) -> List[str]:
    """
    Find all `.hdf5` files in `root/subfolder` (non-recursive), sorted alphabetically.

    Args:
        root: Root directory path.
        subfolder: Subdirectory inside `root` (e.g., `junk_annotated`).

    Returns:
        Sorted list of absolute paths to `.hdf5` files.
    """
    d = os.path.join(root, subfolder)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".hdf5") or name.lower().endswith(".h5"):
            out.append(os.path.join(d, name))
    return out


def sanitize_name(s: str) -> str:
    """
    Sanitize a string into a lowercase, filesystem-safe identifier.

    - Strip leading/trailing whitespace.
    - Replace whitespace with `_`.
    - Replace non-alphanumeric/`._-` characters with `_`.
    - Strip leading/trailing `._-`.

    Args:
        s: Input string.

    Returns:
        Sanitized string.
    """
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    return s.strip("._-")
