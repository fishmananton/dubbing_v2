"""#4: parallelize loudness_adjust's per-line measure and apply loops (cross-line gain
smoothing stays serial in between). Pure refactor — this characterization test must pass
before AND after: it pins the output contract (all lines produced, sorted by start/idx,
each with a smoothed gain and a written output file)."""
import numpy as np
import soundfile as sf
import srt as _srt
from datetime import timedelta

from loudness_adjust import run_line_loudness_stage, LoudnessConfig

SR = 48000


def _wav(path, seconds, freq=220.0, amp=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sf.write(str(path), (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32),
             SR, format="WAV", subtype="FLOAT")


def test_all_lines_measured_smoothed_and_written(tmp_path):
    # vocals covering both sub windows
    _wav(tmp_path / "vocals.wav", 2.0, freq=180.0, amp=0.15)
    tts_dir = tmp_path / "segs"
    _wav(tts_dir / "SPEAKER_00" / "1.wav", 0.6, freq=220.0)
    _wav(tts_dir / "SPEAKER_00" / "2.wav", 0.6, freq=260.0)

    subs = [
        _srt.Subtitle(index=1, start=timedelta(seconds=0.0), end=timedelta(seconds=0.6),
                      content="SPEAKER_00: hello"),
        _srt.Subtitle(index=2, start=timedelta(seconds=1.0), end=timedelta(seconds=1.6),
                      content="SPEAKER_00: world"),
    ]
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text(_srt.compose(subs, reindex=False), encoding="utf-8")

    out = run_line_loudness_stage(str(srt_path), str(tts_dir), str(tmp_path / "vocals.wav"),
                                  LoudnessConfig())

    assert [e["idx"] for e in out] == [1, 2]                      # sorted by start/idx
    for e in out:
        assert "smoothed_gain_db" in e and e["smoothed_gain_db"] is not None
        assert (tts_dir / "SPEAKER_00" / f"{e['idx']}_loudness_out.wav").exists()


def test_missing_segment_is_skipped(tmp_path):
    _wav(tmp_path / "vocals.wav", 2.0)
    tts_dir = tmp_path / "segs"
    _wav(tts_dir / "SPEAKER_00" / "1.wav", 0.6)   # idx 2 intentionally absent

    subs = [
        _srt.Subtitle(index=1, start=timedelta(seconds=0.0), end=timedelta(seconds=0.6),
                      content="SPEAKER_00: hi"),
        _srt.Subtitle(index=2, start=timedelta(seconds=1.0), end=timedelta(seconds=1.6),
                      content="SPEAKER_00: gone"),
    ]
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text(_srt.compose(subs, reindex=False), encoding="utf-8")

    out = run_line_loudness_stage(str(srt_path), str(tts_dir), str(tmp_path / "vocals.wav"),
                                  LoudnessConfig())
    assert [e["idx"] for e in out] == [1]     # missing idx 2 skipped, no crash
