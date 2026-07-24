
import boto3
import srt
from runpod_utils import run_runpod_job
from openai import OpenAI
import json
import tempfile
from modal_utils import run_modal_job

SYSTEM_PROMPT = """You are correcting subtitle emotion labels.

Task:
Fix only clearly wrong emotion labels using subtitle text, context, and speaker role.

Allowed emotions:
["angry", "fearful", "sad", "neutral", "happy", "surprised", "disgusted"]

Rules:
- Be conservative: keep labels that are plausible.
- Output only corrections.
- Use surrounding subtitles for context.
- Do not use "neutral" as a default fallback.
- Use "neutral" only if the line is clearly flat and emotionless even in context.
- Prefer a specific emotion if context supports it.
- Do not invent data.
- Return valid JSON only.

Emotion continuity:
- Do not evaluate lines in isolation.
- Check nearby lines from the same speaker.
- If a speaker has a consistent tone, align outlier lines to that tone unless there is clear evidence of change.

Output:
[
  {
    "speaker": "speaker name",
    "idx": number,
    "from": "old_emotion",
    "to": "new_emotion",
    "confidence": "high|medium|low",
    "reason": "brief reason"
  }
]

Decision:
- Keep original unless the new label is clearly better supported by context.

Before producing output, internally verify each correction against the original label and keep the original unless the change is clearly justified.
"""

def fix_emotion_detect_with_ai(client:OpenAI, model:str, speakers:dict, subs:dict):
    sub_list = [
        {
            "speaker": s.content.split(":", 1)[0].strip(),
            "start": s.start.total_seconds(),
            "end": s.end.total_seconds(),
            "text": s.content.split(":", 1)[1].strip() if ":" in s.content else s.content
        } for s in subs]
    for spk in speakers.values():
        for group in spk["groups"].values():
            group.pop("initial_total_sec", None)
    text = f"MAPPING:\n{speakers}, Subtitles:\n{sub_list}"
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
    )
    corrections = json.loads(response.choices[0].message.content.strip())
    for c in corrections:
        speaker = c["speaker"]
        idx = c["idx"]
        old = c["from"]
        new = c["to"]
        if c['confidence'] == 'low':
            continue

        groups = speakers[speaker]["groups"]

        # Remove from old group
        if old in groups and idx in groups[old]["idxs"]:
            groups[old]["idxs"].remove(idx)
        # 🔥 delete group if empty
        if not groups[old]["idxs"]:
            del groups[old]

        # Add to new group
        if new not in groups:
            groups[new] = {"idxs": []}

        groups[new]["idxs"].append(idx)

    durations = {sub.index: (sub.end - sub.start).total_seconds()for sub in subs}
    for speaker_data in speakers.values():
        for group in speaker_data["groups"].values():
            group["initial_total_sec"] = sum(
                durations[idx] for idx in group["idxs"] if idx in durations
            )

    return speakers


def apply_changed_mappings(speakers_array, new_speakers_array, changed_list, subs):
    durations = {
        sub.index: (sub.end - sub.start).total_seconds()
        for sub in subs
    }

    def remove_idx(mapping, idx, duration):
        for speaker_data in mapping.values():
            groups = speaker_data["groups"]

            for emotion in list(groups.keys()):
                group = groups[emotion]
                if idx in group["idxs"]:
                    group["idxs"].remove(idx)
                    group["initial_total_sec"] -= duration

                    if abs(group["initial_total_sec"]) < 1e-9:
                        group["initial_total_sec"] = 0.0

                    if not group["idxs"]:
                        del groups[emotion]

                    return

    def find_new_place(mapping, idx):
        for speaker, speaker_data in mapping.items():
            for emotion, group in speaker_data["groups"].items():
                if idx in group["idxs"]:
                    return speaker, emotion
        return None, None

    def add_idx(mapping, speaker, emotion, idx, duration):
        if speaker not in mapping:
            mapping[speaker] = {"groups": {}}

        if "groups" not in mapping[speaker]:
            mapping[speaker]["groups"] = {}

        groups = mapping[speaker]["groups"]

        if emotion not in groups:
            groups[emotion] = {
                "idxs": [],
                "initial_total_sec": 0.0
            }

        if idx not in groups[emotion]["idxs"]:
            groups[emotion]["idxs"].append(idx)
            groups[emotion]["idxs"].sort()
            groups[emotion]["initial_total_sec"] += duration

    for idx in changed_list:
        duration = durations.get(idx, 0.0)

        remove_idx(speakers_array, idx, duration)

        new_speaker, new_emotion = find_new_place(new_speakers_array, idx)
        if new_speaker is None:
            continue

        add_idx(speakers_array, new_speaker, new_emotion, idx, duration)

    return speakers_array

def emotion_detect_runpod(openai_client: OpenAI,
                          openai_model:str,
                          audio_file:str,
                          subtitles_file:str,
                          boto_session: boto3.Session,
                          bucket_name: str,
                          runpod_key: str,
                          runpod_template_id: str,
                          emotions_flag=True,
                          changed_list: list| None = None,
                          old_speakers_array = None,
                          run_id:str=''):
    with open(subtitles_file, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    if not emotions_flag:
        speakers = {}
        for sub in subs:
            speaker, _ = sub.content.strip().split(":", 1)
            idx = sub.index
            start, end = sub.start.total_seconds(), sub.end.total_seconds()
            if speaker not in speakers:
                speakers[speaker] = {'groups': {'neutral': {'idxs': [idx], 'initial_total_sec': end - start}}, }
            else:
                speakers[speaker]['groups']['neutral']['idxs'].append(idx)
                speakers[speaker]['groups']['neutral']['initial_total_sec'] += end - start
        return speakers

    if changed_list:
        subs = [s for s in subs if s.index in changed_list]
        srt_content = srt.compose(subs, reindex = False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8") as f:
            f.write(srt_content)
            temp_srt_path = f.name
        subtitles_file=temp_srt_path

    s3 = boto_session.client("s3")
    s3_subtitles_path = f"{run_id}/emotion/input/subtitle.srt"
    s3_audio_path = f"{run_id}/emotion/input/audio.mp3"
    s3.upload_file(subtitles_file, bucket_name, s3_subtitles_path)
    speakers= {}
    subtitle_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_subtitles_path},
        ExpiresIn=3600
    )


    s3.upload_file(audio_file, bucket_name, s3_audio_path)
    audio_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_audio_path},
        ExpiresIn=3600
    )

    runpod_payload = {
        "srt_url": subtitle_url,
        "audio_url": audio_url
    }

    result = run_modal_job(
        app_name="emotion_detect",
        function_name="emotion_job",
        timeout_minutes=30,
        poll_delay_sec=5,
        srt_url=subtitle_url,
        audio_url=audio_url
    )



    #
    # result = run_runpod_job(
    #     runpod_key=runpod_key,
    #     runpod_template_id=runpod_template_id,
    #     payload=runpod_payload,
    #     job_name="emotion_detect",
    #     timeout_minutes=10
    # )

    s3.delete_object(Bucket=bucket_name, Key=s3_audio_path)
    s3.delete_object(Bucket=bucket_name, Key=s3_subtitles_path)
    if result["status"] == "COMPLETED":
        speakers = result['output']['speakers']

    if result["status"] != "COMPLETED":
        raise Exception(f"Didn't detect emotions. job_id {result['job_id']}")

    if changed_list:
        speakers = apply_changed_mappings(
            speakers_array=old_speakers_array,
            new_speakers_array=speakers,
            changed_list=changed_list,
            subs=subs,
        )
    else:
        speakers = fix_emotion_detect_with_ai(openai_client,openai_model, speakers, subs)

    return speakers
