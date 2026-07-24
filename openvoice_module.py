
import tarfile
from pathlib import Path
import boto3
import torch
import torchaudio
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from runpod_utils import run_runpod_job
import shutil

from modal_utils import run_modal_job

ckpt_converter = 'openvoice/OpenVoice/checkpoints/converter'
device="cuda:0" if torch.cuda.is_available() else "cpu"

def make_tar_archive(speaker_name:str, file_list: list, output_dir:str="tmp"):
    output_dir = Path(output_dir)
    archive_path = output_dir / f"{speaker_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_path in file_list:
            tar.add(file_path, arcname=Path(file_path).name)
    return archive_path

def extract_tar_archive(archive_path, output_dir):
    """
    Extracts a .tar.gz archive to a new directory.
    """
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    return output_dir


def s3_file_exists(s3, bucket, key) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as e:
        return False

def make_16k_mono_wav(src_path: str, dst_path: str):
    audio, sr = sf.read(src_path, dtype="float32", always_2d=True)  # shape: (T, C)
    audio = audio.mean(axis=1, keepdims=True)  # mono -> (T, 1)

    wav = torch.from_numpy(audio.T)  # (1, T)

    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000

    sf.write(dst_path, wav.squeeze(0).numpy(), sr, subtype="PCM_16")


def resample_wav(
        input_path: str,
        output_path: str,
        target_sr: int = 48000,
        mono: bool = False,
):

    audio, sr = sf.read(input_path, dtype="float32")

    # shape → [channels, time]
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T

    tensor = torch.tensor(audio)

    # mono conversion (optional)
    if mono and tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)

    # resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        tensor = resampler(tensor)
        sr = target_sr

    # back to numpy [time, channels]
    out = tensor.numpy().T

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, out, sr, format="WAV", subtype="FLOAT")


def audio_precheck(path: str, min_sec=1.0, rms_min=1e-4):
    if not os.path.exists(path):
        return False, "missing file"
    if os.path.getsize(path) < 4096:
        return False, "file too small"

    try:
        info = sf.info(path)
    except Exception as e:
        return False, f"unreadable audio: {e}"

    dur = info.frames / float(info.samplerate)
    if dur < min_sec:
        return False, f"too short: {dur:.2f}s"

    # Read a small chunk to estimate loudness (avoid full read)
    # 2 seconds from the middle
    mid = max(0, info.frames // 2 - int(info.samplerate))
    frames = int(min(info.samplerate * 2, info.frames - mid))
    audio, _ = sf.read(path, start=mid, frames=frames, dtype="float32", always_2d=True)

    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < rms_min:
        return False, f"too quiet (RMS={rms:.6f})"

    return True, f"ok dur={dur:.2f}s sr={info.samplerate} ch={info.channels} rms={rms:.6f}"


def openvoice_convert_runpod(speakers: dict, voice_profile: dict, temp_output_folder:str, reference_speakers_folder: str, base_speaker_folder: str, input_files_folder:str, output_files_folder: str,boto_session: boto3.Session, bucket_name: str, runpod_key: str, runpod_template_id: str,changed_list: list| None =None, run_id:str='', ttsmodel:int=0, total_sec_thresh: float = 5.5):
    if changed_list:
        changed = set(changed_list)

        speakers = {
            speaker: {
                "gender": data["gender"],
                "groups": {
                    emotion: {
                        **group,
                        "idxs": [i for i in group["idxs"] if i in changed]
                    }
                    for emotion, group in data["groups"].items()
                    if any(i in changed for i in group["idxs"])
                }
            }
            for speaker, data in speakers.items()
            if any(i in changed for g in data["groups"].values() for i in g["idxs"])
        }

#fishaudio
    if ttsmodel==3:
        for speaker_name, speaker_data in speakers.items():
            groups = speaker_data.get("groups", {})
            for emotion, group_data in groups.items():
                for idx in group_data['idxs']:
                    in_path = f"{input_files_folder}/{speaker_name}/{idx}.wav"
                    out_path = f"{output_files_folder}/{speaker_name}/{idx}_out.wav"
                    shutil.copy(in_path, out_path)
        return

    s3 = boto_session.client("s3")
    for speaker_name, speaker_data in speakers.items():
        file_list = []
        groups = speaker_data.get("groups", {})

        for emotion, group_data in groups.items():
            reference_file = f"{reference_speakers_folder}/{speaker_name}_{emotion}.wav"
            if group_data['total_sec'] < total_sec_thresh:
                for idx in group_data['idxs']:
                    in_path =f"{input_files_folder}/{speaker_name}/{idx}.wav"
                    out_path=f"{output_files_folder}/{speaker_name}/{idx}_out.wav"
                    align_short_segment(
                        in_path,
                        out_path,
                        voice_profile[idx]
                    )
                continue


            tmp_file = f"{temp_output_folder}/{speaker_name}_{emotion}.wav"
            make_16k_mono_wav(reference_file, tmp_file)
            file_list.append(tmp_file)
            for idx in group_data['idxs']:
                file_list.append(f"{input_files_folder}/{speaker_name}/{idx}.wav")
        if len(file_list) == 0:
            continue
        base_file = f"{base_speaker_folder}/{speaker_name}_combined.wav"
        tmp_file = f"{temp_output_folder}/{speaker_name}_combined.wav"
        make_16k_mono_wav(base_file, tmp_file)
        file_list.append(tmp_file)

        archive_path = make_tar_archive(speaker_name, file_list, temp_output_folder)
        s3_path = f"{run_id}/openvoice/input/{speaker_name}.tar.gz"
        s3.upload_file(archive_path, bucket_name, s3_path)
        input_url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket_name, "Key": s3_path},
            ExpiresIn=3600)
        s3_output_path = f"{run_id}/openvoice/output/{speaker_name}_out.tar.gz"
        output_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket_name, "Key": s3_output_path},
            ExpiresIn=3600
        )
        speakers[speaker_name]["s3_input_obj"] = input_url
        speakers[speaker_name]["s3_output_obj"] = output_url
    runpod_payload = {
        "total_sec_thresh": total_sec_thresh,
        "speakers": speakers
    }
    #
    # result = run_runpod_job(
    #     runpod_key=runpod_key,
    #     runpod_template_id=runpod_template_id,
    #     payload=runpod_payload,
    #     job_name="openvoice_tts",
    #     timeout_minutes=10
    # )
    result = run_modal_job(
        app_name="openvoice",
        function_name="openvoice_job",
        timeout_minutes=30,
        poll_delay_sec=5,
        total_sec_thresh=total_sec_thresh,
        speakers=speakers,
        max_workers=2,
    )
    if result["status"] == "COMPLETED":
        for speaker_name, speaker_data in speakers.items():
            s3_output_path = f"{run_id}/openvoice/output/{speaker_name}_out.tar.gz"
            try:
                if not s3_file_exists(s3, bucket_name, s3_output_path):
                    continue
                s3.download_file(bucket_name, s3_output_path, f"{temp_output_folder}/{speaker_name}_out.tar.gz")
                extract_tar_archive(f"{temp_output_folder}/{speaker_name}_out.tar.gz", f"{temp_output_folder}/{speaker_name}")
                src_dir = f"{temp_output_folder}/{speaker_name}"
                dst_dir = f"{output_files_folder}/{speaker_name}"
                for filename in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, filename)
                    dst_path = os.path.join(dst_dir, filename)
                    if os.path.isfile(src_path):
                        resample_wav(src_path, dst_path, target_sr=48000, mono=True)
            except Exception as e:
                print(f"Error processing speaker {speaker_name}: {e}")


    for speaker_name, _ in speakers.items():
        s3.delete_object(Bucket=bucket_name, Key=f"{run_id}/openvoice/input/{speaker_name}.tar.gz")
        s3.delete_object(Bucket=bucket_name, Key=f"{run_id}/openvoice/output/{speaker_name}_out.tar.gz")

    if result["status"] != "COMPLETED":
        raise Exception(f"Didn't complete openvoice TTS. job_id {result['job_id']}")



def get_median_pitch(y, sr):
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    if len(f0) == 0:
        return None
    return float(np.median(f0))

def soft_pitch_align(y, sr, target_pitch_hz):
    current_pitch = get_median_pitch(y, sr)
    if current_pitch is None:
        return y  # no pitch → leave untouched

    semitones = 12 * np.log2(target_pitch_hz / current_pitch)

    # hard clamp
    semitones = np.clip(semitones, -1.5, 1.5)

    # soft compression
    if abs(semitones) > 0.8:
        semitones = np.sign(semitones) * (
            0.8 + (abs(semitones) - 0.8) * 0.5
        )

    if abs(semitones) < 0.15:
        return y  # too small to matter

    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)

def match_loudness(seg: AudioSegment, target_dbfs: float):
    if seg.dBFS == float("-inf"):
        return seg
    gain = target_dbfs - seg.dBFS
    return seg.apply_gain(gain)

def align_short_segment(in_wav: str, out_wav: str, profile: dict):
    y, sr = librosa.load(in_wav, sr=None)

    # Pitch
    if profile["pitch_median_hz"] is not None:
        y = soft_pitch_align(
            y,
            sr,
            np.clip(
                profile["pitch_median_hz"],
                120,
                280
            )
        )

    # Resample to 48kHz before writing
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000

    sf.write(out_wav, y, sr)

    # Loudness
    if profile["energy_dbfs"] is None:
        return
    seg = AudioSegment.from_file(out_wav)
    seg = match_loudness(seg, profile["energy_dbfs"])
    seg.export(out_wav, format="wav")

