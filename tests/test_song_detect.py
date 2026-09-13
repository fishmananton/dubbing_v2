from datetime import timedelta

import srt

from song_detect import Chunk, chunk_subs


def _sub(idx, start_s, end_s, text="x"):
    return srt.Subtitle(index=idx,
                        start=timedelta(seconds=start_s),
                        end=timedelta(seconds=end_s),
                        content=f"Speaker: {text}")


def test_short_input_is_single_chunk():
    subs = [_sub(1, 0, 2), _sub(2, 100, 102), _sub(3, 470, 480)]
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 1
    assert chunks[0].offset_s == 0.0
    assert [s.index for s in chunks[0].subs] == [1, 2, 3]


def test_splits_long_input_at_nearest_gap_over_threshold():
    subs = []
    t = 0.0
    idx = 1
    while t < 1200:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 5.0 if 600 <= t < 601 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 2
    assert chunks[1].offset_s > 600
    all_idx = [s.index for c in chunks for s in c.subs]
    assert all_idx == [s.index for s in subs]


def test_no_qualifying_gap_allows_oversized_chunk():
    subs = []
    t = 0.0
    idx = 1
    while t < 1500:
        subs.append(_sub(idx, t, t + 1.0))
        gap = 3.0 if 1080 <= t < 1081 else 1.0
        t += 1.0 + gap
        idx += 1
    chunks = chunk_subs(subs, target_min=10, min_split_min=12, gap_min_s=2.0)
    assert len(chunks) == 2
    assert chunks[0].subs[-1].end.total_seconds() <= 1081
    assert chunks[1].offset_s > 1080


from song_detect import build_chunk_script


def test_build_chunk_script_rebases_times_keeps_ids():
    subs = [_sub(41, 605, 607, "hello"), _sub(42, 610, 612, "world")]
    chunk = Chunk(subs=subs, offset_s=600.0, end_s=612.0)
    script = build_chunk_script(chunk)
    # original IDs preserved
    assert "[41]" in script and "[42]" in script
    # times rebased by -600s: 605->5.00, 610->10.00
    assert "5.00-7.00" in script
    assert "10.00-12.00" in script
    # no absolute 605/610 leaked
    assert "605" not in script and "610" not in script
