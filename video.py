import subprocess
import tempfile
import os
import srt


def cut_video(input_video: str, output_video: str, duration_sec: float):
    """Cut the first N seconds of a video for test mode."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-t", str(duration_sec),
        "-c", "copy",
        output_video
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return output_video


def clear_substitles(subtitles_file: str):
    with open(subtitles_file, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    for sub in subs:
        content = sub.content.strip()
        speaker, text = content.split(":", 1)
        sub.content = text.strip()
    result_srt = srt.compose(subs, reindex=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8") as tmp:
        tmp.write(result_srt)
        return tmp.name



def encode_preview_base(input_video: str, output_path: str) -> subprocess.Popen:
    """Start background low-priority encode of scaled-down video (no audio, no subs)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-an",
        "-vf", "scale=-2:min(480\\,ih)",
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "ultrafast",
        "-threads", "2",
        output_path,
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=lambda: os.nice(10),
    )


_EXPORT_CRF = {
    "480":  23,
    "720":  22,
    "1080": 20,
}

EXPORT_RESOLUTIONS = {"original", "480", "720", "1080"}


def generate_videos(
    original_video: str,
    final_audio: str,
    subtitles: str,
    output_name: str,
    preview: bool = False,
    resolution: str = "original",
    pre_encoded_video: str | None = None,
):
    if resolution not in EXPORT_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {EXPORT_RESOLUTIONS}")
    subtitles = clear_substitles(subtitles)

    use_pre_encoded = preview and pre_encoded_video and os.path.exists(pre_encoded_video)
    print(f"generate_videos: use_pre_encoded={use_pre_encoded}, pre_encoded_video={pre_encoded_video}")

    video_source = pre_encoded_video if use_pre_encoded else original_video

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_source,
        "-i", final_audio,
        "-i", subtitles,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "2:s:0",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=eng",
        "-c:a", "aac",
        "-b:a", "128k" if preview else "320k",
    ]

    if use_pre_encoded:
        cmd += ["-c:v", "copy"]
    elif preview:
        cmd += [
            "-vf", "scale=-2:min(480\\,ih)",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "ultrafast",
        ]
    else:
        if resolution == "original":
            cmd += ["-c:v", "copy"]
        else:
            cmd += [
                "-vf", f"scale=-2:{resolution}",
                "-c:v", "libx264",
                "-crf", str(_EXPORT_CRF[resolution]),
                "-preset", "medium",
            ]

    cmd += [output_name]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

