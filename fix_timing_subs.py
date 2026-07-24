import json
import re
from pathlib import Path
import srt
from openai import OpenAI
from difflib import SequenceMatcher



def clean_json_response(text: str) -> str:
    text = text.strip()
    match = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def split_label_text(content: str) -> tuple[str | None, str]:
    """
    Language-neutral split:
    'Speaker: text' -> ('Speaker', 'text')
    """
    if ":" not in content:
        return None, content.strip()

    label, text = content.split(":", 1)
    return label.strip(), text.strip()


def speaker_label(content: str) -> str | None:
    label, _ = split_label_text(content)
    return label


def text_body(content: str) -> str:
    _, body = split_label_text(content)
    return body.strip()


def rewrite_is_universally_valid(
    old_content: str,
    new_content: str,
    original_content: str,
) -> bool:
    """
    Global, language-neutral validation only.
    No hardcoded technical terms.
    No punctuation/dot/end-word assumptions.
    """

    old_label = speaker_label(old_content)
    new_label = speaker_label(new_content)

    # Must preserve speaker label exactly.
    if not old_label or new_label != old_label:
        return False

    old_body = text_body(old_content)
    new_body = text_body(new_content)
    source_body = text_body(original_content)

    # Do not turn a meaningful line into silence.
    # Keep this loose because some subtitles are tiny fragments.
    if source_body.strip() and len(old_body.strip()) > 3 and not new_body.strip():
        return False

    # Avoid obvious model explosions. This is intentionally loose and language-neutral.
    old_len = max(1, len(old_body.strip()))
    new_len = len(new_body.strip())

    if old_len > 15 and new_len > old_len * 4:
        return False

    return True


def build_timing_rewrite_prompt(
    source_language: str,
    target_language: str,
    punctuation: bool = False,
    prev_visibility_res_to_fix: list[dict] | None = None,
) -> str:
    return f"""
You are a professional dubbing subtitle editor.

Task:
Rewrite only editable translated subtitles to improve TTS timing while preserving meaning and natural target-language speech.

Languages:
- Source: {source_language}
- Target: {target_language}

Subtitle roles:
- Primary Target: subtitle with the timing problem. Usually rewrite this.
- Optional Adjacent Target: may be rewritten only when needed for grammar, sentence flow, or meaning across subtitle boundaries.
- Read-only Context: never rewrite.

Hard priorities, in order:

{
'''
1. Valid, natural target-language grammar across the local subtitle group.
2. Preserve source meaning, speaker label, names, numbers, and important terms.
3. Improve timing in the requested direction.
4. Keep speech natural for dubbing.''' 
if target_language != 'de' else '''
1. Fit the target timing window for TTS as closely as possible.
2. Preserve core source meaning, speaker label, names, numbers, and important terms.
3. Keep target-language speech natural enough for dubbing.
4. Grammar matters, but brevity wins when timing and perfect grammar conflict.
'''
}


Rules:
- Preserve speaker labels exactly.
- Return only subtitles whose text actually changes.
- Never return Read-only Context indices.
- Do not duplicate meaning across adjacent subtitles.
- Do not drop required verbs, objects, negation, names, numbers, or important terms.
{
'''
- Convert numbers, currencies, dates, percentages, abbreviations, units, and symbols into fully spoken forms appropriate for natural speech in the target language.
- Use the spoken form that fits the grammar of the rewritten sentence, including case, agreement, gender, classifiers, counters, or declension when the target language requires it.
- Do not use digit/symbol shorthand only to make timing shorter.
''' if target_language != 'de' else '''
- Use compact written forms for numbers and percentages when they are naturally speakable by TTS.
- Prefer: 100.000, 240.000, 3 Milliarden, 30 %, 79,9 %, 19,6 %, 0,5 %, AV1, CPU.
- Do not spell out long numbers or percentages unless absolutely necessary.
- Do not expand technical abbreviations.
- Keep terms compact: FFmpeg, VLC, AV1, CPU, Codecs, Compiler, Intrinsics.
- For German shortening, remove filler and softeners such as: ja, also, eben, halt, wirklich, vielleicht, irgendwie, eigentlich, im Grunde.
- Remove repeated ideas when the source repeats itself.
- Preserve the core point, not every word.
- It is allowed to paraphrase strongly if the core meaning remains.
'''
}

- Preserve product names, person names, brand names, code identifiers, and technical terms in their normal target-language spoken form.
- Do not create dangling fragments unless the source itself is intentionally a fragment.
- If the Primary Target cannot be shortened or lengthened naturally by itself, adjust Optional Adjacent Target subtitles in the same local group.
{
'''
- If timing and grammar conflict, grammar and meaning win.
''' if target_language != 'de' else '''
- If timing and perfect grammar conflict, preserve core meaning and make the shortest natural-enough German line.
'''
}
- If the requested timing change is impossible without damaging meaning, make the best safe improvement.

Timing instructions:
- Change = sub: make the spoken text shorter if possible, by restructuring naturally.
- Change = add: make the spoken text longer if possible, by making existing source meaning more explicit.
{
'''
- Timing changes must still keep numbers, percentages, dates, currencies, abbreviations, units, and symbols speakable; do not compress them into written notation.
''' if target_language != 'de' else '''
- For German, compact written notation is allowed and preferred when TTS can speak it naturally.
- Prefer compact forms like 100.000, 240.000, 30 %, 79,9 %, AV1, CPU.
'''
}
- Do not add new facts, new opinions, or new tone only to change duration.

Before answering:
- Read previous context + editable targets + next context as one phrase.
- Make sure the editable group is grammatical, non-duplicated, and meaning-preserving.
- If you changed an adjacent subtitle, explain briefly why.

{'''Previous attempt:
Use the previous-attempt information to avoid repeating a failed wording. Make a better local rewrite while preserving grammar and meaning.''' if prev_visibility_res_to_fix else ""}

Output only valid JSON:
[
  {{
    "idx": <index>,
    "text": "<speaker label>: <rewritten text>",
    "changed": true,
    "role": "primary" | "adjacent",
    "reason": "<short reason>"
  }}
]
""".strip()


def repeated_short_sequence(text: str, min_tokens: int = 3) -> bool:
    tokens = normalize_text(text).split()
    if len(tokens) < min_tokens * 2:
        return False

    seen = set()
    for i in range(len(tokens) - min_tokens + 1):
        seq = tuple(tokens[i:i + min_tokens])
        if seq in seen:
            return True
        seen.add(seq)

    return False


def group_text_for_indices(subs_by_idx: dict[int, srt.Subtitle], indices: list[int]) -> str:
    bodies = []
    for idx in indices:
        if idx in subs_by_idx:
            bodies.append(text_body(subs_by_idx[idx].content))
    return " ".join(x for x in bodies if x).strip()

def build_timing_rewrite_user_content(
    non_translated_subs: list[srt.Subtitle],
    translated_subs: list[srt.Subtitle],
    visibility_res_to_fix: list[dict],
    context_radius: int = 3,
    editable_radius: int = 1,
    prev_visibility_res_to_fix: list[dict] | None = None,
    prev_translated_subs: list[srt.Subtitle] | None = None,
) -> tuple[str, set[int], dict[int, str]]:
    orig_by_idx = {sub.index: sub for sub in non_translated_subs}
    tr_by_idx = {sub.index: sub for sub in translated_subs}

    if prev_visibility_res_to_fix and prev_translated_subs:
        prev_tr_by_idx = {sub.index: sub for sub in prev_translated_subs}
        prev_visibility_res_to_fix_by_idx = {fix["idx"]: fix for fix in prev_visibility_res_to_fix}
    else:
        prev_tr_by_idx = {}
        prev_visibility_res_to_fix_by_idx = {}

    parts = []
    parts.append("Context and rewrite requests:\n")

    allowed_editable_idxs: set[int] = set()
    allowed_editable_roles: dict[int, str] = {}

    for fix in visibility_res_to_fix:
        idx = int(fix["idx"])
        change = fix["change"]
        cur_len = int(fix["len"])
        delta = int(fix["value"])

        center_label = speaker_label(tr_by_idx[idx].content) if idx in tr_by_idx else None

        editable_idxs = {
            j
            for j in range(idx - editable_radius, idx + editable_radius + 1)
            if (
                j in orig_by_idx
                and j in tr_by_idx
                and speaker_label(tr_by_idx[j].content) == center_label
            )
        }

        allowed_editable_idxs.update(editable_idxs)

        for j in editable_idxs:
            if j == idx:
                allowed_editable_roles[j] = "primary"
            elif allowed_editable_roles.get(j) != "primary":
                allowed_editable_roles[j] = "adjacent"

        target_len = cur_len + delta if change == "add" else cur_len - delta

        relative = (delta / cur_len) if cur_len > 0 else 0.0
        if change == "sub":
            relative = -relative
        relative_percent = round(relative * 100)

        parts.append("---")
        parts.append(f"Block for idx {idx}:\n")
        editable_group_indices = sorted(editable_idxs)
        parts.append(
            "Editable local group indices: "
            + ", ".join(str(x) for x in editable_group_indices)
        )
        parts.append(
            "If grammar or meaning spans these indices, rewrite the editable group as one local phrase, "
            "but return only the individual indices that actually changed."
        )
        parts.append("")

        for j in range(idx - context_radius, idx + context_radius + 1):
            if j not in orig_by_idx or j not in tr_by_idx:
                continue

            if j == idx:
                role = "Primary Target"
            elif j in editable_idxs:
                role = "Optional Adjacent Target"
            else:
                role = "Read-only Context"

            parts.append(f"{role}:")
            parts.append(f"idx {j}")
            parts.append(f"Original: {orig_by_idx[j].content.strip()}")
            parts.append(f"Translation: {tr_by_idx[j].content.strip()}")

            if j == idx:
                sign = "+" if change == "add" else "-"
                rel_sign = "+" if relative_percent >= 0 else ""

                parts.append(f"Change: {change}")
                parts.append(f"Current duration ms: {cur_len}")
                parts.append(f"Target adjustment ms: {sign}{delta}")
                parts.append(f"Relative change: {rel_sign}{relative_percent}%")
                parts.append(f"Approx target duration ms: {target_len}")

                if (
                    prev_visibility_res_to_fix
                    and idx in prev_visibility_res_to_fix_by_idx
                    and idx in prev_tr_by_idx
                ):
                    prev_text = prev_tr_by_idx[idx].content.strip()
                    prev_translated_text = tr_by_idx[j].content.strip()
                    prev_change = prev_visibility_res_to_fix_by_idx[idx]["change"]
                    prev_sign = "+" if prev_change == "add" else "-"
                    prev_len = prev_visibility_res_to_fix_by_idx[idx]["len"]
                    prev_delta = int(prev_visibility_res_to_fix_by_idx[idx]["value"])
                    prev_relative = (prev_delta / prev_len) if prev_len > 0 else 0.0

                    if prev_change == "sub":
                        prev_relative = -prev_relative

                    prev_relative_percent = round(prev_relative * 100)
                    prev_rel_sign = "+" if prev_relative_percent >= 0 else ""

                    current_error_vs_target = cur_len - target_len
                    abs_current_error = abs(current_error_vs_target)

                    if abs_current_error <= 400:
                        magnitude = "slightly"
                    elif abs_current_error <= 1200:
                        magnitude = "moderately"
                    else:
                        magnitude = "significantly"

                    if current_error_vs_target < 0:
                        correction_instruction = (
                            f"The current translation is still too short by {abs_current_error} ms versus the target. "
                            f"Lengthen it {magnitude}, but keep it close to the target duration and do not overshoot. "
                            "Do not repeat a similar sentence structure."
                        )
                    elif current_error_vs_target > 0:
                        correction_instruction = (
                            f"The current translation is too long by {abs_current_error} ms versus the target. "
                            f"Shorten it {magnitude}, but keep it close to the target duration and do not over-correct. "
                            "Do not repeat a similar sentence structure."
                        )
                    else:
                        correction_instruction = (
                            "Keep the duration close to the target and preserve the meaning. "
                            "Do not repeat a similar sentence structure."
                        )

                    if prev_text:
                        parts.append("Previous attempt context:")
                        parts.append(f"Translation before previous rewrite: {prev_text}")
                        parts.append(f"Result after previous rewrite / current translation now: {prev_translated_text}")
                        parts.append(f"Previous run requested change: {prev_change}")
                        parts.append(f"Previous run input duration ms: {prev_len}")
                        parts.append(f"Previous run requested adjustment ms: {prev_sign}{prev_delta}")
                        parts.append(f"Previous run requested relative change: {prev_rel_sign}{prev_relative_percent}%")
                        parts.append(f"Current input duration for this run: {cur_len}")
                        parts.append(f"Target duration for this run: {target_len}")
                        parts.append(f"Current requested adjustment from measured audio: {current_error_vs_target:+d} ms")
                        parts.append(f"Correction instruction for this run: {correction_instruction}")

            parts.append("")

    return "\n".join(parts).strip(), allowed_editable_idxs, allowed_editable_roles


def parse_timing_rewrite_response(response_text: str) -> dict[int, dict]:
    cleaned = clean_json_response(response_text)
    data = json.loads(cleaned)

    if not isinstance(data, list):
        raise ValueError("Rewrite response must be a JSON list")

    rewrites: dict[int, dict] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        idx = item.get("idx")
        text = item.get("text")
        changed = item.get("changed", True)
        role = item.get("role", "")
        reason = item.get("reason", "")

        if idx is None or text is None:
            continue

        if isinstance(changed, str):
            if changed.strip().lower() in {"false", "no", "0"}:
                continue
        elif changed is False:
            continue

        try:
            idx_int = int(idx)
        except Exception:
            continue

        rewrites[idx_int] = {
            "text": str(text).strip(),
            "role": str(role).strip().lower(),
            "reason": str(reason).strip(),
        }

    return rewrites


def apply_rewrites_to_subs(
    translated_subs: list[srt.Subtitle],
    rewrites: dict[int, dict],
) -> list[srt.Subtitle]:
    updated = []

    for sub in translated_subs:
        if sub.index in rewrites:
            updated.append(
                srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=rewrites[sub.index]["text"],
                )
            )
        else:
            updated.append(sub)

    return updated


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def rewrite_timing_mismatched_subtitles(
    client: OpenAI,
    non_translated_subs_file: str,
    translated_subs_file: str,
    visibility_res_to_fix: list[dict],
    source_language: str,
    target_language: str,
    model: str,
    result_file: str | None = None,
    punctuation: bool = False,
    context_radius: int = 3,
    editable_radius: int = 1,
    prev_visibility_res_to_fix: list[dict] | None = None,
    prev_translated_subs_file: str | None = None,
    return_debug: bool = False,
):
    with open(non_translated_subs_file, "r", encoding="utf-8") as f:
        non_translated_subs = list(srt.parse(f.read()))

    with open(translated_subs_file, "r", encoding="utf-8") as f:
        translated_subs = list(srt.parse(f.read()))

    if prev_visibility_res_to_fix and prev_translated_subs_file:
        with open(prev_translated_subs_file, "r", encoding="utf-8") as f:
            prev_translated_subs = list(srt.parse(f.read()))
    else:
        prev_translated_subs = None
        prev_visibility_res_to_fix = None

    if not visibility_res_to_fix:
        result_srt = srt.compose(translated_subs, reindex=False)
        if result_file:
            Path(result_file).write_text(result_srt, encoding="utf-8")

        if return_debug:
            return {
                "changed_ids": [],
                "rewrites": {},
                "allowed_editable_idxs": [],
                "raw_response": "",
            }

        return []

    system_prompt = build_timing_rewrite_prompt(
        source_language,
        target_language,
        punctuation,
        prev_visibility_res_to_fix,
    )

    user_content, allowed_editable_idxs, allowed_editable_roles = build_timing_rewrite_user_content(
        non_translated_subs=non_translated_subs,
        translated_subs=translated_subs,
        visibility_res_to_fix=visibility_res_to_fix,
        context_radius=context_radius,
        editable_radius=editable_radius,
        prev_visibility_res_to_fix=prev_visibility_res_to_fix,
        prev_translated_subs=prev_translated_subs,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    raw_response = response.choices[0].message.content.strip()
    try:
        raw_rewrites = parse_timing_rewrite_response(raw_response)
    except Exception as e:
        print(f"⚠️ Timing rewrite JSON parse failed: {e}")
        print(f"Raw response preview: {raw_response[:1000]}")
        raw_rewrites = {}

    orig_by_idx = {sub.index: sub for sub in non_translated_subs}
    tr_by_idx = {sub.index: sub for sub in translated_subs}

    rewrites: dict[int, dict] = {}

    for idx, item in raw_rewrites.items():
        if idx not in allowed_editable_idxs:
            continue

        if idx not in tr_by_idx or idx not in orig_by_idx:
            continue

        expected_role = allowed_editable_roles.get(idx, "adjacent")
        item["role"] = "primary" if expected_role == "primary" else "adjacent"

        new_text = item["text"]

        if not rewrite_is_universally_valid(
            old_content=tr_by_idx[idx].content,
            new_content=new_text,
            original_content=orig_by_idx[idx].content,
        ):
            print(f"⚠️ Rejected invalid rewrite idx={idx}: {new_text}")
            continue

        rewrites[idx] = item

    # Light group-level validation against obvious duplicate phrasing.
    # Language-neutral: only rejects repeated short token sequences introduced by rewrite.
    candidate_subs = apply_rewrites_to_subs(translated_subs, rewrites)
    candidate_by_idx = {sub.index: sub for sub in candidate_subs}
    original_tr_by_idx = {sub.index: sub for sub in translated_subs}

    bad_rewrite_idxs: set[int] = set()

    for fix in visibility_res_to_fix:
        center = int(fix["idx"])

        if center not in tr_by_idx:
            continue

        center_label = speaker_label(tr_by_idx[center].content)

        group_indices = [
            j
            for j in range(center - editable_radius, center + editable_radius + 1)
            if (
                    j in candidate_by_idx
                    and j in tr_by_idx
                    and speaker_label(tr_by_idx[j].content) == center_label
            )
        ]

        old_group = group_text_for_indices(original_tr_by_idx, group_indices)
        new_group = group_text_for_indices(candidate_by_idx, group_indices)

        # Reject only if the rewrite introduced obvious repetition.
        if not repeated_short_sequence(old_group) and repeated_short_sequence(new_group):
            for j in group_indices:
                if j in rewrites:
                    bad_rewrite_idxs.add(j)

    for idx in bad_rewrite_idxs:
        print(f"⚠️ Rejected rewrite idx={idx}: introduced repeated local phrase")
        rewrites.pop(idx, None)

    updated_subs = apply_rewrites_to_subs(translated_subs, rewrites)
    result_srt = srt.compose(updated_subs, reindex=False)

    changed_ids = sorted(
        idx
        for idx, item in rewrites.items()
        if idx in tr_by_idx and tr_by_idx[idx].content.strip() != item["text"].strip()
    )

    if result_file:
        Path(result_file).write_text(result_srt, encoding="utf-8")

    if return_debug:
        return {
            "changed_ids": changed_ids,
            "rewrites": rewrites,
            "allowed_editable_idxs": sorted(allowed_editable_idxs),
            "raw_response": raw_response,
        }

    return changed_ids