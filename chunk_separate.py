"""Validation harness for chunk-parallel audio separation — NOT wired into production.

Goal: prove that splitting audio into OVERLAPPING chunks, separating each chunk
independently, discarding the padded edges, and stitching the fully-contexted centers
produces output IDENTICAL to whole-file separation — for any deterministic,
context-bounded separator (which mel_band_roformer is: it separates each output sample
from a bounded temporal window). If that holds, chunk-parallel is safe and
content-independent; the only failure vectors (too-little overlap, per-chunk
normalization, misalignment) are all detectable by the seam-diff below.

Design (see the accompanying design discussion):
  - KEEP regions tile [0, total) exactly (contiguous, no gaps/overlap) -> stitching is a
    plain copy, no overlap-add, no crossfade (avoids phase/level seams entirely).
  - Each chunk is fed to the separator with `overlap` samples of CONTEXT padding on both
    sides (clamped at the file edges). After separation we discard that padding and keep
    only the center, which had the exact same surrounding audio it would have had in the
    whole-file pass. With `overlap >= the separator's context radius`, the kept center is
    bit-identical to whole-file — regardless of content.

`run_seam_validation(separate_fn, ...)` is the reusable gate: pass a synthetic
context-bounded separator (the tests do) to prove the METHOD, or pass a function that
calls the real Modal roformer on a chunk to prove it for the REAL model before rollout.
The separator is treated as a black box `np.ndarray -> np.ndarray` of equal length.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    idx: int
    sep_start: int   # first sample fed to the separator (incl. context padding)
    sep_end: int     # one-past-last sample fed (exclusive)
    keep_start: int  # first sample of the fully-contexted region we keep (absolute)
    keep_end: int    # one-past-last kept sample (exclusive)

    @property
    def keep_offset(self) -> int:
        """Where the kept region begins inside this chunk's separated output."""
        return self.keep_start - self.sep_start


def plan_chunks(total: int, chunk_len: int, overlap: int) -> list[Chunk]:
    """Tile [0, total) into contiguous KEEP blocks of `chunk_len` samples; each is fed to
    the separator with `overlap` samples of context padding on both sides, clamped at the
    file edges. KEEP blocks tile exactly, so stitching is a plain copy."""
    if total <= 0:
        return []
    if chunk_len <= 0:
        raise ValueError("chunk_len must be positive")
    chunks: list[Chunk] = []
    idx = 0
    keep_start = 0
    while keep_start < total:
        keep_end = min(keep_start + chunk_len, total)
        sep_start = max(0, keep_start - overlap)
        sep_end = min(total, keep_end + overlap)
        chunks.append(Chunk(idx, sep_start, sep_end, keep_start, keep_end))
        keep_start = keep_end
        idx += 1
    return chunks


def separate_chunked(separate_fn, audio: np.ndarray, chunk_len: int,
                     overlap: int) -> tuple[np.ndarray, list[Chunk]]:
    """Run `separate_fn` on each padded chunk, keep the fully-contexted center, stitch.
    Returns (stitched_output, plan). `separate_fn(x)` must return an array as long as x."""
    total = len(audio)
    plan = plan_chunks(total, chunk_len, overlap)
    out = np.zeros_like(audio)
    for c in plan:
        sep = separate_fn(audio[c.sep_start:c.sep_end])
        if len(sep) != c.sep_end - c.sep_start:
            raise ValueError(
                f"separate_fn changed length: got {len(sep)}, "
                f"expected {c.sep_end - c.sep_start}")
        keep = sep[c.keep_offset:c.keep_offset + (c.keep_end - c.keep_start)]
        out[c.keep_start:c.keep_end] = keep
    return out, plan


def _rms_db(x: np.ndarray) -> float:
    """RMS of x in dBFS (full-scale = 1.0). Exact-zero -> -inf."""
    if x.size == 0:
        return -np.inf
    r = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
    return 20.0 * np.log10(r) if r > 0 else -np.inf


def seam_diff(reference: np.ndarray, chunked: np.ndarray, plan: list[Chunk],
              seam_window: int) -> dict:
    """Numerically compare whole-file `reference` vs stitched `chunked`.

    Returns:
      global_db   - RMS error over the whole signal (dBFS)
      max_seam_db - worst RMS error within +/- seam_window samples of any interior
                    KEEP boundary (where chunking artifacts would appear)
      seams       - [(boundary_sample, rms_db), ...] per interior boundary
    A truly seam-clean run has global_db and max_seam_db at the float noise floor."""
    if reference.shape != chunked.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {chunked.shape}")
    diff = reference.astype(np.float64) - chunked.astype(np.float64)
    seams: list[tuple[int, float]] = []
    for c in plan[:-1]:                       # interior boundaries only
        b = c.keep_end
        lo = max(0, b - seam_window)
        hi = min(len(reference), b + seam_window)
        seams.append((b, _rms_db(diff[lo:hi])))
    max_seam_db = max((d for _, d in seams), default=-np.inf)
    return {"global_db": _rms_db(diff), "max_seam_db": max_seam_db, "seams": seams}


def run_seam_validation(separate_fn, audio: np.ndarray, chunk_len: int, overlap: int,
                        seam_window: int | None = None) -> tuple[dict, list[Chunk]]:
    """THE GATE. Separate `audio` whole-file and chunked, then seam-diff them.

    `separate_fn`: deterministic separator, np.ndarray -> np.ndarray (same length). Use a
    synthetic context-bounded op (tests) to prove the method, or a real per-chunk Modal
    roformer call to gate a production rollout. For stereo, validate each channel (or a
    stem) separately — the method is per-channel identical.
    """
    if seam_window is None:
        seam_window = max(1, overlap)
    reference = separate_fn(audio)
    chunked, plan = separate_chunked(separate_fn, audio, chunk_len, overlap)
    return seam_diff(reference, chunked, plan, seam_window), plan
