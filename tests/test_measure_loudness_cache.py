"""#3: memoize measure_loudness. The source video's loudness is invariant within a run,
but build_audio scans it 2-4x. Cache by (path, mtime, size) so ffmpeg runs once; a
changed file recomputes. Behavior-preserving: identical LUFS/LRA returned."""
import time

import final_audio


def test_second_call_served_from_cache(tmp_path, monkeypatch):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"video-bytes")
    calls = {"n": 0}

    def fake(inp):
        calls["n"] += 1
        return {"i": -18.5, "lra": 9.0}

    monkeypatch.setattr(final_audio, "_measure_loudness_uncached", fake)
    final_audio._loudness_cache.clear()

    a = final_audio.measure_loudness(str(f))
    b = final_audio.measure_loudness(str(f))
    assert a == b == {"i": -18.5, "lra": 9.0}
    assert calls["n"] == 1  # ffmpeg scan ran once, second call cached


def test_recomputes_when_file_changes(tmp_path, monkeypatch):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"video-bytes")
    calls = {"n": 0}

    def fake(inp):
        calls["n"] += 1
        return {"i": -16.0, "lra": 7.0}

    monkeypatch.setattr(final_audio, "_measure_loudness_uncached", fake)
    final_audio._loudness_cache.clear()

    final_audio.measure_loudness(str(f))
    time.sleep(0.01)
    f.write_bytes(b"different-longer-video-bytes")  # changes size + mtime
    final_audio.measure_loudness(str(f))
    assert calls["n"] == 2  # cache invalidated by the changed file
