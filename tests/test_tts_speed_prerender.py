"""#2: pre-render the placement-INDEPENDENT (non-overflow) atempo calls in parallel.

The final tts_build_final speed factor is placement-independent only when
needed = raw_len/subtitle_duration <= speaker_base (the clamp branch). Overflow lines
(needed > speaker_base) depend on actual_available_duration → next_free_start_ms and
CANNOT be known before the sequential loop. These tests pin the non-overflow factor to
match tts_v2.py:363-390 exactly and prove parallel pre-render is byte-identical to the
serial inline adjust_speed it replaces.
"""
from pydub.generators import Sine

from tts_v2 import adjust_speed, _nonoverflow_speed_factor, _prerender_nonoverflow


def _seg(ms, freq=440):
    return Sine(freq).to_audio_segment(duration=ms).set_frame_rate(48000).set_channels(1)


def test_factor_clamps_up_to_floor_092():
    # needed = 500/1000 = 0.5 < 0.92 floor -> clamp up to 0.92 (mirrors max(0.92, ...))
    assert _nonoverflow_speed_factor(500, 1000, speaker_base=1.2,
                                     max_speed_factor=1.3) == 0.92


def test_factor_uses_needed_when_between_floor_and_base():
    # needed = 1.1, <= base 1.2 -> factor = 1.1
    assert abs(_nonoverflow_speed_factor(1100, 1000, 1.2, 1.3) - 1.1) < 1e-9


def test_factor_none_when_overflow():
    # needed = 1.5 > base 1.2 -> overflow branch (placement-dependent) -> not prerenderable
    assert _nonoverflow_speed_factor(1500, 1000, 1.2, 1.3) is None


def test_factor_respects_max_speed_cap():
    # needed 1.25 <= base 1.3 -> 1.25, but max_speed 1.1 caps it
    assert _nonoverflow_speed_factor(1250, 1000, 1.3, 1.1) == 1.1


def test_factor_zero_subtitle_duration_is_neutral():
    # guard: subtitle_duration <= 0 -> needed treated as 1.0 (mirrors loop)
    assert _nonoverflow_speed_factor(1000, 0, 1.2, None) == 1.0


def test_prerender_is_byte_identical_to_serial_inline():
    segs = {1: _seg(500, 440), 2: _seg(800, 330), 3: _seg(1200, 550),
            4: _seg(700, 220)}
    factors = {1: 0.92, 2: 1.1, 3: 1.0, 4: 0.95}  # idx 3 is a no-op, must be skipped
    items = [(i, segs[i], factors[i]) for i in sorted(segs)]

    # serial reference: exactly what the loop does inline today
    serial = {i: adjust_speed(segs[i], f) for i, f in factors.items() if f != 1.0}
    par = _prerender_nonoverflow(items)

    assert set(par.keys()) == set(serial.keys())        # no-op idx 3 excluded
    for i in serial:
        assert par[i].raw_data == serial[i].raw_data     # byte-identical
