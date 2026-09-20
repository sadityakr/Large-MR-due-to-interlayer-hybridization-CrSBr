"""Lightweight reader for Bruker NanoScope .000 AFM height images (numpy-only).

Replaces the pySPM dependency for ingesting Bruker NanoScope CIAO files. pySPM
pins an old Python ceiling (<3.13) and on Python 3.13 its installer falls back to
a release that forces a broken numpy build, so this self-contained parser keeps
the AFM notebook runnable in the numpy-2 / Python-3.13 `data_analysis` kernel.

It parses the ASCII CIAO header, resolves the z-scale chain
(raw LSB -> volts via the hard scale, volts -> nm via the referenced sensitivity)
exactly as Gwyddion/pySPM do, and returns the first Height channel.

Input:
    A path to a Bruker NanoScope .000 file. The file must contain at least one
    "Ciao image list" block whose "@2:Image Data" channel is "Height".

Process:
    Reads the ASCII header, locates the first Height image block, computes
    height_nm = raw_int * (Z-scale value [V]) * (sensitivity [nm/V]) / 2**(8*bytes),
    and reshapes the little-endian integer image. Row order is kept as stored,
    which reproduces pySPM's pixel orientation bit-for-bit (validated).

Output:
    read_bruker_height() -> (data_nm, sx_um, sy_um):
        data_nm : 2-D float64 array of height in nm (no leveling applied)
        sx_um   : fast-axis scan size in micrometres
        sy_um   : slow-axis scan size in micrometres
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
# ]
# ///

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_HEADER_END = b"\\*File list end"
_UNIT_TO_UM = {"~m": 1.0, "um": 1.0, "µm": 1.0, "nm": 1e-3, "m": 1e6}


def _read_header(raw: bytes) -> list[str]:
    """Return the CIAO header as a list of de-escaped lines."""
    end = raw.find(_HEADER_END)
    blob = raw[: end if end != -1 else 1 << 16]
    text = blob.decode("latin-1", errors="replace")
    return [ln.strip().lstrip("\\") for ln in text.splitlines()]


def _first_float(s: str) -> float:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m is None:
        raise ValueError(f"no number in {s!r}")
    return float(m.group())


def _lookup_sensitivity(lines: list[str], ref: str) -> float:
    """Find the soft-scale sensitivity (nm/V) named `ref`, e.g. 'Sens. Zscan'."""
    key = re.compile(r"^@(?:\d+:)?" + re.escape(ref) + r":")
    for ln in lines:
        if key.match(ln):
            return _first_float(ln.split(":", 1)[1])
    raise ValueError(f"sensitivity reference {ref!r} not found in header")


def _image_blocks(lines: list[str]) -> list[dict[str, str]]:
    """Split the header into per-'Ciao image list' key/value dicts."""
    blocks: list[dict[str, str]] = []
    cur: dict[str, str] | None = None
    for ln in lines:
        if ln.startswith("*"):
            if ln.startswith("*Ciao image list"):
                cur = {}
                blocks.append(cur)
            else:
                cur = None
            continue
        # Split on the first ": " (colon-space). CIAO keys carry an internal
        # "@N:" prefix with no trailing space (e.g. "@2:Image Data"), so a plain
        # split on ":" would truncate the key.
        if cur is not None and ": " in ln:
            k, v = ln.split(": ", 1)
            cur[k.strip()] = v.strip()
    return blocks


def _scan_size_um(block: dict[str, str]) -> tuple[float, float]:
    """Parse 'Scan size: 20 20 ~m' (or single value) into (sx, sy) in µm."""
    raw = block.get("Scan size") or block.get("Scan Size")
    if raw is None:
        raise ValueError("no 'Scan size' field in image block")
    toks = raw.split()
    nums = [float(t) for t in toks if re.fullmatch(r"[-+]?\d*\.?\d+", t)]
    unit = next((t for t in toks if t in _UNIT_TO_UM), "~m")
    f = _UNIT_TO_UM[unit]
    if len(nums) >= 2:
        return nums[0] * f, nums[1] * f
    return nums[0] * f, nums[0] * f


def read_bruker_height(filepath: str | Path) -> tuple[np.ndarray, float, float]:
    """Read the first Height channel of a Bruker NanoScope .000 file.

    Returns (data_nm, sx_um, sy_um); see module docstring.
    """
    raw = Path(filepath).read_bytes()
    lines = _read_header(raw)

    block = next(
        (
            b
            for b in _image_blocks(lines)
            if b.get("@2:Image Data", "").endswith('"Height"')
        ),
        None,
    )
    if block is None:
        raise ValueError("no Height image found in file")

    offset = int(block["Data offset"])
    bpp = int(block["Bytes/pixel"])
    nx = int(block["Samps/line"])
    ny = int(block["Number of lines"])

    # @2:Z scale: V [Sens. Zscan] (0.006713765 V/LSB) 20.11444 V
    zline = block["@2:Z scale"]
    ref = re.search(r"\[([^\]]+)\]", zline).group(1)
    z_value_v = float(re.search(r"\)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", zline).group(1))
    sens_nm_per_v = _lookup_sensitivity(lines, ref)

    scale_nm = z_value_v * sens_nm_per_v / (2 ** (8 * bpp))

    dtype = np.dtype("<i2" if bpp == 2 else "<i4")
    n = nx * ny
    flat = np.frombuffer(raw, dtype=dtype, count=n, offset=offset).astype(np.float64)
    data = flat.reshape(ny, nx) * scale_nm  # row order matches pySPM (no flip)

    sx, sy = _scan_size_um(block)
    return data, sx, sy
