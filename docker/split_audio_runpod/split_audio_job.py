import torch
import torch.nn.functional as F
import requests
import soundfile as sf
import runpod
import torchaudio
import os
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile
import tempfile

_DIARIZATION_PIPELINE = None


_DEMUCS_MODEL = None


def save_as_wav(tensor, sr, filename, target_sr=48000, mono=False):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    tensor = tensor.detach().cpu().float()

    # Downmix to mono if requested
    if mono and tensor.size(0) > 1:
        tensor = tensor.mean(dim=0, keepdim=True)


    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor = resampler(tensor)
        sr = target_sr

    # Save as WAV (format inferred from extension)
    audio_np = tensor.T.numpy()  # soundfile expects shape [time, channels]
    sf.write(filename, audio_np, sr, format="WAV")


def download_file(url, path):
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)

def upload_file(url, path):
    with open(path, "rb") as f:
        requests.put(url, data=f)



def db_to_amp(db: float) -> float:
    return 10 ** (db / 20.0)


def gaussian_kernel1d(sigma: float, device, dtype, radius: int | None = None):
    if sigma <= 0:
        return None

    if radius is None:
        radius = max(1, int(4 * sigma + 0.5))

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def smooth_2d_mask(mask: torch.Tensor, smooth_freq: float, smooth_time: float):
    """
    mask shape: [freq, frames]
    """
    x = mask[None, None, :, :]  # [1, 1, F, T]

    kf = gaussian_kernel1d(smooth_freq, mask.device, mask.dtype)
    if kf is not None:
        kf = kf.view(1, 1, -1, 1)
        pad = (0, 0, kf.shape[2] // 2, kf.shape[2] // 2)
        x = F.pad(x, pad, mode="replicate")
        x = F.conv2d(x, kf)

    kt = gaussian_kernel1d(smooth_time, mask.device, mask.dtype)
    if kt is not None:
        kt = kt.view(1, 1, 1, -1)
        pad = (kt.shape[3] // 2, kt.shape[3] // 2, 0, 0)
        x = F.pad(x, pad, mode="replicate")
        x = F.conv2d(x, kt)

    return x[0, 0]


def soft_vocal_guided_cleanup_tensor(
    vocals: torch.Tensor,
    music: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    win_length: int = 4096,
    power: float = 2.0,
    mask_floor: float = 0.0,
    max_attenuation_db: float = -6.0,
    smooth_freq: float = 1.2,
    smooth_time: float = 2.0,
    vocal_sensitivity: float = 1.0,
    quiet_gate_db: float = -55.0,
    quiet_gate_reduction_db: float = -3.0,
):
    """
    vocals/music shape: [channels, samples]
    returns cleaned music: [channels, samples]
    """

    if vocals.ndim != 2 or music.ndim != 2:
        raise ValueError("Expected vocals and music tensors shaped [channels, samples]")

    min_len = min(vocals.shape[-1], music.shape[-1])
    vocals = vocals[:, :min_len]
    music = music[:, :min_len]

    if vocals.shape[0] != music.shape[0]:
        if vocals.shape[0] == 1 and music.shape[0] == 2:
            vocals = vocals.repeat(2, 1)
        elif vocals.shape[0] == 2 and music.shape[0] == 1:
            music = music.repeat(2, 1)
        else:
            raise ValueError("Channel mismatch cannot be resolved automatically")

    device = music.device
    dtype = music.dtype

    window = torch.hann_window(win_length, device=device, dtype=dtype)

    max_att = db_to_amp(max_attenuation_db)
    quiet_gate_att = db_to_amp(quiet_gate_reduction_db)

    cleaned_channels = []

    for ch in range(music.shape[0]):
        y_bg = music[ch]
        y_v = vocals[ch]

        D_bg = torch.stft(
            y_bg,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        D_v = torch.stft(
            y_v,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )

        mag_bg = D_bg.abs()
        mag_v = D_v.abs()

        eps = 1e-8
        vocal_ratio = mag_v / (mag_v + mag_bg + eps)

        mask = torch.clamp(vocal_ratio * vocal_sensitivity, 0.0, 1.0)
        mask = mask.pow(power)

        mask = smooth_2d_mask(mask, smooth_freq=smooth_freq, smooth_time=smooth_time)
        mask = torch.clamp(mask, mask_floor, 1.0)

        gain = 1.0 - mask * (1.0 - max_att)
        D_clean = D_bg * gain

        y_clean = torch.istft(
            D_clean,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            length=y_bg.shape[-1],
        )

        # Light quiet-frame gate
        frames = y_clean.unfold(0, win_length, hop_length)
        frame_rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
        frame_rms_db = 20.0 * torch.log10(frame_rms + 1e-8)

        quiet = frame_rms_db < quiet_gate_db
        if quiet.any():
            gate_gain = torch.ones_like(frame_rms)
            gate_gain[quiet] = quiet_gate_att

            kt = gaussian_kernel1d(1.0, device, dtype)
            if kt is not None:
                x = gate_gain[None, None, :]
                x = F.pad(x, (kt.numel() // 2, kt.numel() // 2), mode="replicate")
                gate_gain = F.conv1d(x, kt.view(1, 1, -1))[0, 0]

            frame_positions = torch.arange(
                gate_gain.numel(), device=device, dtype=dtype
            ) * hop_length

            sample_positions = torch.arange(
                y_clean.numel(), device=device, dtype=dtype
            )

            # torch has no simple np.interp equivalent, so use bucket interpolation
            idx = torch.searchsorted(frame_positions, sample_positions, right=True) - 1
            idx = torch.clamp(idx, 0, gate_gain.numel() - 2)

            x0 = frame_positions[idx]
            x1 = frame_positions[idx + 1]
            y0 = gate_gain[idx]
            y1 = gate_gain[idx + 1]

            t = (sample_positions - x0) / torch.clamp(x1 - x0, min=1.0)
            sample_gain = y0 + t * (y1 - y0)

            y_clean = y_clean * sample_gain

        cleaned_channels.append(y_clean)

    return torch.stack(cleaned_channels, dim=0)



def split_audio(input_audio_path: str, output_vocal: str, output_music: str):
    global _DEMUCS_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _DEMUCS_MODEL is None:
        _DEMUCS_MODEL = get_model('htdemucs').to(device).eval()
    model = _DEMUCS_MODEL

    audio_file = AudioFile(input_audio_path)
    sr = model.samplerate
    channels = model.audio_channels
    wav = audio_file.read(streams=0, samplerate=sr, channels=channels)
    wav = wav.to(device).float().unsqueeze(0)

    with torch.no_grad():
        out = apply_model(
            model,
            wav,
            device=device,
            split=True,
            overlap=0.25,
            shifts=2,
        )

    sources = model.sources
    vocals = out[0, sources.index("vocals")]
    music = sum(out[0, i] for i, s in enumerate(sources) if s != "vocals")

    cleaned_music = soft_vocal_guided_cleanup_tensor(
        vocals=vocals,
        music=music,
        max_attenuation_db=-4.0,
        power=2.5,
        vocal_sensitivity=0.9,
        smooth_freq=1.0,
        smooth_time=2.5,
        quiet_gate_db=-58.0,
        quiet_gate_reduction_db=-2.0,
    )

    save_as_wav(vocals, sr, output_vocal, target_sr=48000, mono=True)
    save_as_wav(cleaned_music, sr, output_music, target_sr=48000, mono=False)

def handler(job):

    temp_folder = tempfile.mkdtemp()
    print("Job started")
    payload = job["input"]
    input_url = payload["input_url"]
    input_path = f"{temp_folder}/input.wav"
    download_file(input_url, input_path)
    print("Downloaded file")


    output_vocal_url = payload["output_vocal_url"]
    output_music_url = payload["output_music_url"]
    vocal_path = f"{temp_folder}/vocals.wav"
    music_path = f"{temp_folder}/music.wav"
    split_audio(input_path, vocal_path, music_path)
    print("Splitted audio")


    upload_file(output_vocal_url, vocal_path)
    upload_file(output_music_url, music_path)
    return {"message": "Processing done"}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

