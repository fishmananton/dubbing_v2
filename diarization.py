import os
import time
import requests
import boto3
import numpy as np
from pydub import AudioSegment
from collections import defaultdict
from scipy.spatial.distance import cosine
import soundfile as sf


def diarize (boto_session: boto3.Session, pyannote_key, bucket_name: str,  audio_path: str, num_speakers = None, run_id:str=''):

    if num_speakers == 1:
        file = sf.SoundFile(audio_path)
        duration = round(len(file) / file.samplerate, 3)
        speaker_segments = [
            {
            "start": 0.000,
            "end": duration,
            "speaker_raw": "SPEAKER_00",
            "speaker": "SPEAKER_00"
            }
        ]
        return speaker_segments

    s3 = boto_session.client("s3")
    s3_path = f"{run_id}/diarize/input/input.wav"
    audio_for_diarize = AudioSegment.from_file(audio_path).set_channels(1).set_frame_rate(16000)
    import tempfile
    tmp_diarize = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_for_diarize.export(tmp_diarize.name, format="wav")
    tmp_diarize.close()
    s3.upload_file(tmp_diarize.name, bucket_name, s3_path)
    os.unlink(tmp_diarize.name)
    input_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_path},
        ExpiresIn=3600
    )
    url = "https://api.pyannote.ai/v1/diarize"
    payload = {
        "url": input_url,
        "model": "precision-2",
    }
    if num_speakers:
        payload["numSpeakers"] = num_speakers
    headers = {
        "Authorization": f"Bearer {pyannote_key}", #sk_ada934ffb3eb4d5695e3e15f07f0cd5b
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()
    response_json = response.json()
    job_id = response_json.get("jobId")
    if not job_id:
        s3.delete_object(Bucket=bucket_name, Key=s3_path)
        raise Exception(f"Didn't start diarization job. response: {response_json['message']}")
    status = ''
    data = 'Timed Out'
    for i in range(120):
        url = f"https://api.pyannote.ai/v1/jobs/{job_id}"
        response = requests.get(url, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        data = response.json()
        if data.get("status") in ("succeeded", "canceled", 'failed'):
            status = data.get("status")
            break
        else:
            time.sleep(5)
    s3.delete_object(Bucket=bucket_name, Key=s3_path)
    if status != "succeeded":
        raise Exception(f"Didn't succeeded diarization. response: {data}")
    speaker_segments = []
    for item in data['output']['diarization']:
        speaker_segments.append({
            "start": item['start'],
            "end": item['end'],
            "speaker_raw": item['speaker']
        })
    calculated_speakers = len(set(seg["speaker_raw"] for seg in speaker_segments))
    if num_speakers is None and calculated_speakers <= 2:
        num_speakers = calculated_speakers
    if num_speakers is None:
        speaker_segments = assign_voice_identities(audio_path, speaker_segments)
    else:
        for seg in speaker_segments: seg["speaker"] = seg["speaker_raw"]
    return speaker_segments

def assign_voice_identities(audio_file, srt_segments):
    def get_audio_embedding(audio, start, end, model, target_sr=16000):
        seg = audio[int(start * 1000): int(end * 1000)]

        if len(seg) < 300:  # <300ms useless
            return None

        seg = seg.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= np.max(np.abs(samples)) + 1e-9

        waveform = torch.tensor(samples).unsqueeze(0)

        with torch.no_grad():
            emb = model.encode_batch(waveform)

        return emb.squeeze().cpu().numpy()

    SIM_SPLIT = 0.75  # если ниже — разные люди
    MIN_SEG_DUR = 0.2  # минимум для embedding

    MERGE_THRESHOLD = 0.80  # для слияния разных SPEAKER_X
    MIN_MERGE_DUR = 4.0  # минимум речи для якоря
    SHORT_ASSIGN_DUR = 0.8

    import torch
    from speechbrain.inference import EncoderClassifier
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/spkrec-ecapa"
    )



    audio = AudioSegment.from_file(audio_file)

    # 1️⃣ Собираем embeddings по diarization-speaker
    buckets = defaultdict(list)

    for seg in srt_segments:
        dur = seg["end"] - seg["start"]
        if dur < MIN_SEG_DUR:
            continue

        emb = get_audio_embedding(
            audio, seg["start"], seg["end"], speaker_model
        )
        if emb is None:
            continue

        buckets[seg["speaker_raw"]].append({
            "emb": emb,
            "dur": dur,
            "seg": seg
        })


    # 2️⃣ Проверяем каждый SPEAKER_X
    for raw_speaker, items in buckets.items():

        # ❗ слишком мало данных → доверяем diarization
        if len(items) < 3:
            for i in items:
                i["seg"]["speaker"] = raw_speaker
            continue

        # берём первый как reference
        ref = items[0]["emb"]
        group_a, group_b = [], []

        for i in items:
            sim = 1 - cosine(ref, i["emb"])
            if sim >= SIM_SPLIT:
                group_a.append(i)
            else:
                group_b.append(i)

        # 3️⃣ Решаем — split или нет
        if len(group_b) == 0 or sum(i["dur"] for i in group_b) < 3.0:
            # split не имеет смысла
            for i in items:
                i["seg"]["speaker"] = raw_speaker
        else:
            # 🔥 SPLIT: два реальных человека
            for i in group_a:
                i["seg"]["speaker"] = f"{raw_speaker}_A"
            for i in group_b:
                i["seg"]["speaker"] = f"{raw_speaker}_B"



    # ---------- 5️⃣ MERGE разных speaker по embeddings ----------
    speaker_buffers = defaultdict(list)

    # собираем embeddings по текущему speaker label
    for seg in srt_segments:
        dur = seg["end"] - seg["start"]
        if dur < MIN_SEG_DUR:
            continue

        emb = get_audio_embedding(
            audio, seg["start"], seg["end"], speaker_model
        )
        if emb is None:
            continue

        speaker_buffers[seg["speaker"]].append((emb, dur))

    # строим anchor для каждого speaker
    speaker_anchors = {}
    for speaker, items in speaker_buffers.items():
        total = sum(d for _, d in items)
        if total < MIN_MERGE_DUR:
            continue

        embs = np.array([e for e, _ in items])
        weights = np.array([d for _, d in items])
        speaker_anchors[speaker] = np.average(embs, axis=0, weights=weights)

    # кластеризация anchors → Person_X
    person_refs = {}
    speaker_to_person = {}
    person_counter = 1

    for speaker, anchor in speaker_anchors.items():
        best, best_score = None, 0.0

        for person, ref in person_refs.items():
            score = 1 - cosine(anchor, ref)
            if score > best_score:
                best, best_score = person, score

        if best and best_score >= MERGE_THRESHOLD:
            speaker_to_person[speaker] = best
        else:
            person = f"Person_{person_counter}"
            person_counter += 1
            person_refs[person] = anchor
            speaker_to_person[speaker] = person

    # применяем merge
    for seg in srt_segments:
        seg["speaker"] = speaker_to_person.get(seg["speaker"] if 'speaker' in seg else seg["speaker_raw"], None)

    # 6️⃣ Re-assign very short segments to closest Person anchor (if confident)
    for idx, seg in enumerate(srt_segments):
        dur = seg["end"] - seg["start"]
        if dur >= SHORT_ASSIGN_DUR:
            continue

        # if we have no person anchors, nothing to do
        if not person_refs:
            continue

        if seg['speaker'] is not None:
            continue

            # --- YOUR APPROACH: copy speaker from nearest neighbor with same speaker_raw ---
        raw = seg["speaker_raw"]

        left = srt_segments[idx - 1] if idx > 0 else None
        right = srt_segments[idx + 1] if idx + 1 < len(srt_segments) else None

        if left and left["speaker_raw"] == raw and (seg["start"] - left["end"]) <= 1.2:
            seg["speaker"] = left["speaker"]
            continue

        if right and right["speaker_raw"] == raw and (right["start"] - seg["end"]) <= 1.2:
            seg["speaker"] = right["speaker"]
            continue

        emb = get_audio_embedding(audio, seg["start"], seg["end"], speaker_model)
        if emb is None:
            continue

        best_person, best_score = None, -1.0
        for person, ref in person_refs.items():
            score = 1 - cosine(emb, ref)
            if score > best_score:
                best_person, best_score = person, score

        seg["speaker"] = best_person

    # 3.5️⃣ Assign remaining unmatched segments if confident
    for seg in srt_segments:
        if seg.get("speaker") is not None:
            continue  # already assigned

        assigned = try_assign_unmatched(
            seg,
            audio,
            speaker_model,
            person_refs,
            get_audio_embedding,
            min_sim=0.5,  # tune this
        )

        if assigned:
            seg["speaker"] = assigned

    # 4️⃣ Финальный проход: если саб не трогали — оставить как есть
    for seg in srt_segments:
        if seg['speaker'] is None:
            seg["speaker"] = seg["speaker_raw"]

    return srt_segments


def try_assign_unmatched(
    seg,
    audio,
    speaker_model,
    person_refs,
    get_audio_embedding,
    min_sim=0.7,        # <- YOU control this
):
    emb = get_audio_embedding(audio, seg["start"], seg["end"], speaker_model)
    if emb is None:
        return None

    best_person, best_score = None, -1.0
    for person, ref in person_refs.items():
        score = 1 - cosine(emb, ref)
        if score > best_score:
            best_person, best_score = person, score

    if best_score >= min_sim:
        return best_person

    return None
