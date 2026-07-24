import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)


def standardize_wav(input_path: str, output_path: str, target_sr: int = 48000, mono: bool = False):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(target_sr),
        "-c:a", "pcm_f32le",
        "-ac", "1" if mono else "2",
        output_path,
    ]
    run(cmd)


def duck_background(
    background_wav: str,
    dialog_wav: str,
    output_wav: str,
    bg_gain_db: float = -3.0,
    threshold: float = 0.03,
    ratio: float = 6.0,
    attack: int = 15,
    release: int = 280,
):
    filter_complex = (
        f"[0:a]volume={bg_gain_db}dB[bg];"
        f"[1:a]apad[sc];"
        f"[bg][sc]sidechaincompress="
        f"threshold={threshold}:ratio={ratio}:attack={attack}:release={release}"
        f"[ducked]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", background_wav,
        "-i", dialog_wav,
        "-filter_complex", filter_complex,
        "-map", "[ducked]",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_f32le",
        output_wav,
    ]
    run(cmd)


def mix_stems(
    dialog_wav: str,
    background_mix_wav: str,
    output_wav: str,
    non_speech_layer: str | None = None,
    original_speech_layer: str | None = None,
    background_gain_db: float = 0.0,
    dialog_gain_db: float = 0.0,
    non_speech_gain_db: float = 0.0,
    original_underlay_gain_db: float = -16.0,
    enable_compression: bool = True
):
    inputs = [
        "-i", dialog_wav,
        "-i", background_mix_wav,
    ]

    stream_defs = [
        f"[0:a]volume={dialog_gain_db}dB,"
        f"aecho=0.8:0.9:35|70:0.08|0.04"
        f"pan=stereo|c0=c0|c1=c0[dlg]",
        f"[1:a]volume={background_gain_db}dB[bg]",
    ]
    mix_inputs = ["[bg]"]

    next_input_idx = 2

    if original_speech_layer is not None:
        inputs += ["-i", original_speech_layer]
        stream_defs.append(
            f"[{next_input_idx}:a]volume={original_underlay_gain_db}dB,pan=stereo|c0=c0|c1=c0[orig]"
        )
        mix_inputs.append("[orig]")
        next_input_idx += 1

    if non_speech_layer is not None:
        inputs += ["-i", non_speech_layer]
        stream_defs.append(
            f"[{next_input_idx}:a]volume={non_speech_gain_db}dB,pan=stereo|c0=c0|c1=c0[ns]"
        )
        mix_inputs.append("[ns]")

    mix_inputs.append("[dlg]")
    amix_inputs = len(mix_inputs)

    if enable_compression:
        mix_filter = (
            f"{''.join(mix_inputs)}amix=inputs={amix_inputs}:normalize=0,"
            f"acompressor=threshold=0.125:ratio=2.5:attack=20:release=150[m]"
        )
    else:
        mix_filter = (
            f"{''.join(mix_inputs)}amix=inputs={amix_inputs}:normalize=0[m]"
        )

    filter_complex = ";".join(stream_defs + [mix_filter])

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[m]",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_f32le",
        output_wav,
    ]
    run(cmd)


def loudnorm_pass1(input_wav: str, target_i: float = -16.0, target_lra: float = 7.0, target_tp: float = -1.5):
    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:print_format=json",
        "-f", "null",
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    match = re.search(r"\{\s*\"input_i\".*?\}", proc.stderr, flags=re.S)
    if not match:
        raise RuntimeError("Could not parse loudnorm JSON from ffmpeg output.")
    return json.loads(match.group(0))


def loudnorm_pass2(
    input_wav: str,
    output_wav: str,
    stats: dict,
    target_i: float = -16.0,
    target_lra: float = 7.0,
    target_tp: float = -1.5,
):
    af = (
        f"loudnorm=I={target_i}:LRA={target_lra}:TP={target_tp}:"
        f"measured_I={stats['input_i']}:"
        f"measured_LRA={stats['input_lra']}:"
        f"measured_TP={stats['input_tp']}:"
        f"measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:"
        f"linear=true:print_format=summary,"
        f"alimiter=limit=0.80:attack=5:release=50:level=disabled"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_wav,
        "-af", af,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_f32le",
        output_wav,
    ]
    run(cmd)
    #
    # background_gain_db: float = -3.0,
    # dialog_gain_db: float = 0.0,
    # non_speech_gain_db: float = -14.0,
    # original_underlay_gain_db: float = -16.0,
    # duck_threshold: float = 0.03,
    # duck_ratio: float = 6.0,
    # duck_attack: int = 15,
    # duck_release: int = 280,

def build_audio(
    tts_segments_folder: str,
    background_file: str,
    output_audio_wav: str,
    non_speech_layer: str | None = None,
    original_speech_layer: str | None = None,
    background_gain_db: float = -6.0,
    dialog_gain_db: float = -4.0,
    non_speech_gain_db: float = -4.0,
    original_underlay_gain_db: float = -22.0,
    duck_threshold: float = 0.08,
    duck_ratio: float = 1.6,
    duck_attack: int = 60,
    duck_release: int = 500,
    stem_background_out: str | None = None,
    stem_dialog_out: str | None = None,
    stem_original_out: str | None = None,
):
    """
    Final mix pipeline:
    1) standardize all stems to 48k
    2) duck background by dialog
    3) mix ducked background + dialog + optional layers
    4) final 2-pass loudnorm

    Inputs:
    - tts_segments_folder/final.wav : built dialog stem
    - background_file               : demucs background/music stem
    - non_speech_layer              : optional non-speech vocal residue
    - original_speech_layer         : optional quiet original speech underlay
    """

    dialog_src = str(Path(tts_segments_folder) / "final.wav")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        dialog_wav = str(tmpdir / "dialog.wav")
        background_wav = str(tmpdir / "background.wav")
        ducked_bg_wav = str(tmpdir / "background_ducked.wav")
        premaster_wav = str(tmpdir / "premaster.wav")

        ns_wav = str(tmpdir / "non_speech.wav") if non_speech_layer is not None else None
        orig_wav = str(tmpdir / "original_speech.wav") if original_speech_layer is not None else None

        # Phase 1: standardize all inputs in parallel
        with ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(standardize_wav, dialog_src, dialog_wav, 48000, True),
                ex.submit(standardize_wav, background_file, background_wav, 48000, False),
            ]
            if ns_wav is not None:
                futures.append(ex.submit(standardize_wav, non_speech_layer, ns_wav, 48000, True))
            if orig_wav is not None:
                futures.append(ex.submit(standardize_wav, original_speech_layer, orig_wav, 48000, True))
            for f in futures:
                f.result()

        # Phase 2: duck background, export dialog stem, export original stem — all independent
        def export_dialog_stem():
            if not stem_dialog_out:
                return
            if ns_wav is not None:
                run(["ffmpeg", "-y", "-i", dialog_wav, "-i", ns_wav,
                     "-filter_complex", "amix=inputs=2:normalize=0[m]",
                     "-map", "[m]", "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k",
                     stem_dialog_out])
            else:
                run(["ffmpeg", "-y", "-i", dialog_wav,
                     "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k",
                     stem_dialog_out])

        def export_original_stem():
            if stem_original_out and orig_wav is not None:
                run(["ffmpeg", "-y", "-i", orig_wav,
                     "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k",
                     stem_original_out])

        with ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(duck_background,
                          background_wav=background_wav,
                          dialog_wav=dialog_wav,
                          output_wav=ducked_bg_wav,
                          bg_gain_db=0.0,
                          threshold=duck_threshold,
                          ratio=duck_ratio,
                          attack=duck_attack,
                          release=duck_release),
                ex.submit(export_dialog_stem),
                ex.submit(export_original_stem),
            ]
            for f in futures:
                f.result()

        # Phase 3: export background stem + mix_stems — both need ducked_bg_wav
        def export_background_stem():
            if stem_background_out:
                run(["ffmpeg", "-y", "-i", ducked_bg_wav,
                     "-ac", "1", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "96k",
                     stem_background_out])

        with ThreadPoolExecutor() as ex:
            futures = [
                ex.submit(export_background_stem),
                ex.submit(mix_stems,
                          dialog_wav=dialog_wav,
                          background_mix_wav=ducked_bg_wav,
                          output_wav=premaster_wav,
                          non_speech_layer=ns_wav,
                          original_speech_layer=orig_wav,
                          background_gain_db=background_gain_db,
                          dialog_gain_db=dialog_gain_db,
                          non_speech_gain_db=non_speech_gain_db,
                          original_underlay_gain_db=original_underlay_gain_db,
                          enable_compression=False),
            ]
            for f in futures:
                f.result()

        stats = loudnorm_pass1(
            premaster_wav,
            target_i=-16.0,
            target_lra=7.0,
            target_tp=-1.5,
        )

        loudnorm_pass2(
            premaster_wav,
            output_audio_wav,
            stats,
            target_i=-16.0,
            target_lra=7.0,
            target_tp=-1.5,
        )
