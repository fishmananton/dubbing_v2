"""Harness tests for chunk-parallel separation (chunk_separate.py). GPU-free.

The centerpiece is a synthetic *context-bounded* separator — a symmetric boxcar FIR whose
output[n] depends only on input[n-R : n+R]. mel_band_roformer is likewise context-bounded
(it separates each sample from a bounded window), so if the chunk/overlap/stitch method is
bit-identical to whole-file for the FIR at overlap >= R, the method is correct for the real
model too — and it's content-independent (the tests show it across diverse signals). The
failure test proves the seam-diff actually CATCHES insufficient overlap, so it can gate a
real rollout.
"""
import numpy as np

from chunk_separate import (Chunk, plan_chunks, separate_chunked, seam_diff,
                            run_seam_validation)


def _fir(radius):
    """Deterministic, context-bounded separator: symmetric boxcar of the given radius.
    output[n] uses only input[n-radius : n+radius] (zero-padded at array edges)."""
    kernel = np.ones(2 * radius + 1, dtype=np.float64)

    def sep(x):
        return np.convolve(x, kernel, mode="same")

    return sep


# ----------------------------- plan / stitch mechanics -----------------------------

def test_plan_tiles_keeps_exactly_and_pads_with_overlap():
    plan = plan_chunks(total=1000, chunk_len=300, overlap=50)
    assert plan[0].keep_start == 0
    assert plan[-1].keep_end == 1000
    # KEEP regions are contiguous and gap-free (they tile the whole file)
    for a, b in zip(plan, plan[1:]):
        assert a.keep_end == b.keep_start
    # interior chunk padded by overlap on both sides
    mid = plan[1]
    assert mid.sep_start == mid.keep_start - 50
    assert mid.sep_end == mid.keep_end + 50
    # file edges clamp the padding
    assert plan[0].sep_start == 0
    assert plan[-1].sep_end == 1000


def test_plan_empty_and_single_chunk():
    assert plan_chunks(0, 100, 10) == []
    plan = plan_chunks(50, 100, 10)
    assert len(plan) == 1 and plan[0].keep_start == 0 and plan[0].keep_end == 50


def test_stitch_reconstructs_with_identity_separator():
    audio = np.random.RandomState(0).randn(1000)
    out, plan = separate_chunked(lambda x: x, audio, chunk_len=137, overlap=20)
    assert np.array_equal(out, audio)          # identity separator -> exact reconstruction
    assert plan[-1].keep_end == 1000


# --------------------------------- THE PROOF ---------------------------------

def test_overlap_ge_radius_is_bit_identical_across_content_types():
    # With overlap >= the separator's context radius, chunked == whole-file, for ANY
    # content. Diverse synthetic "content" to demonstrate file-independence.
    rs = np.random.RandomState(1)
    signals = {
        "white_noise": rs.randn(6000),
        "sine": np.sin(2 * np.pi * 5 * np.linspace(0, 1, 6000)),
        "sustained_dc": np.full(6000, 0.7),          # worst case for context
        "sparse_impulses": (rs.rand(6000) > 0.99).astype(np.float64),
        "chirp": np.sin(np.cumsum(np.linspace(0.01, 0.5, 6000))),
    }
    R = 32
    sep = _fir(R)
    for name, audio in signals.items():
        report, _ = run_seam_validation(sep, audio, chunk_len=700, overlap=R)  # overlap==R
        assert report["global_db"] < -100, f"{name}: global {report['global_db']}"
        assert report["max_seam_db"] < -100, f"{name}: seam {report['max_seam_db']}"


def test_more_overlap_than_needed_also_identical():
    audio = np.random.RandomState(9).randn(6000)
    R = 16
    report, _ = run_seam_validation(_fir(R), audio, chunk_len=500, overlap=R * 4)
    assert report["global_db"] < -100 and report["max_seam_db"] < -100


def test_edge_chunks_match_whole_file():
    # first/last samples clamp context in BOTH whole-file and chunked -> identical there.
    audio = np.random.RandomState(3).randn(2000)
    report, plan = run_seam_validation(_fir(16), audio, chunk_len=400, overlap=16)
    assert report["global_db"] < -100
    assert plan[0].sep_start == 0 and plan[-1].sep_end == 2000


# ------------------------- the gate CATCHES failures -------------------------

def test_insufficient_overlap_is_detected_at_seams():
    # overlap < context radius MUST show up as seam error — proves the gate works.
    audio = np.random.RandomState(2).randn(6000)
    R = 40
    report, _ = run_seam_validation(_fir(R), audio, chunk_len=700, overlap=R // 4)
    assert report["max_seam_db"] > -40, "seam-diff missed under-provisioned overlap"
    # and the damage is concentrated at the seams, not uniform across the file
    assert report["max_seam_db"] > report["global_db"] + 6


def test_seam_diff_localizes_injected_click():
    ref = np.zeros(1000)
    got = np.zeros(1000)
    plan = plan_chunks(1000, 250, 10)
    boundary = plan[0].keep_end            # a real interior seam position
    got[boundary] += 0.5                   # inject a click exactly at the seam
    rep = seam_diff(ref, got, plan, seam_window=20)
    assert rep["max_seam_db"] > -40        # clearly caught vs the <-100 clean floor
    assert any(b == boundary for b, _ in rep["seams"])


def test_per_chunk_normalization_shows_as_seam_error():
    # A separator that normalizes PER INPUT (a real risk) produces level jumps at seams;
    # the gate must flag it even though each chunk in isolation "looks fine".
    def normalizing_sep(x):
        peak = np.max(np.abs(x)) or 1.0
        return x / peak                      # per-chunk gain -> different scale per chunk
    rs = np.random.RandomState(7)
    audio = rs.randn(6000) * rs.uniform(0.2, 1.0, 6000)  # varying level across the file
    report, _ = run_seam_validation(normalizing_sep, audio, chunk_len=700, overlap=64)
    assert report["max_seam_db"] > -40, "per-chunk normalization drift not detected"
