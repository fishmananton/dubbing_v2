from pydub import AudioSegment
import numpy as np
import srt


def detect_gender(audio_path: str, srt_path: str, MALE_STRONG: int=160, FEMALE_STRONG:int =210, sample_rate:int=16000):
    import parselmouth
    with open(srt_path, 'r', encoding='utf-8') as f:
        raw_subs = list(srt.parse(f.read()))
    speakers_set =  {}
    for sub in raw_subs:
        speaker, _ = sub.content.strip().split(":", 1)
        idx = sub.index
        start, end = sub.start.total_seconds(), sub.end.total_seconds()
        if speaker not in speakers_set:
            speakers_set[speaker] = {}
    for speaker, speaker_data in speakers_set.items():
        subs = [
            sub for sub in raw_subs
            if sub.content.strip().startswith(f"{speaker}:")
        ]
        if not subs:
            continue

        audio = AudioSegment.from_file(audio_path)

        def mean_f0_for_segment(start_ms, end_ms):
            """
            Compute mean F0 (fundamental frequency) for a given time segment in ms.
            Handles too-short or silent segments gracefully.
            """
            # --- 1️⃣ Duration sanity check ---
            duration_ms = end_ms - start_ms
            if duration_ms < 100:  # less than 0.1s
                return None  # too short for reliable pitch detection

            # --- 2️⃣ Extract audio segment safely ---
            seg = audio[start_ms:end_ms].set_channels(1).set_frame_rate(sample_rate)
            samples = np.array(seg.get_array_of_samples()).astype(np.float32)

            if len(samples) == 0:
                return None

            # --- 3️⃣ Normalize samples ---
            samples /= np.iinfo(seg.array_type).max
            samples = samples.astype(np.float64)

            # --- 4️⃣ Build parselmouth.Sound and compute pitch ---
            try:
                snd = parselmouth.Sound(samples, sampling_frequency=seg.frame_rate)
                pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
                mean_f0 = parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz")

                # Sometimes Praat returns undefined (NaN)
                if mean_f0 is None or np.isnan(mean_f0) or mean_f0 <= 0:
                    return None
                return mean_f0

            except parselmouth.PraatError as e:
                print(f"⚠️ Pitch analysis failed between {start_ms}–{end_ms} ms: {e}")
                return None

        # # --- Step 1: analyze only first sub ---
        # first = subs[0]
        # mean_f0 = mean_f0_for_segment(int(first.start.total_seconds() * 1000), int(first.end.total_seconds() * 1000))
        #
        # # --- Step 2: classify initial result ---
        # if mean_f0 is None or mean_f0 <= 0:
        #     gender = None
        # elif mean_f0 < male_thresh:
        #     gender = "male"
        # elif mean_f0 > female_thresh:
        #     gender = "female"
        # else:
        #     gender = None  # ambiguous
        # gender = None
        # # --- Step 3: if ambiguous, analyze up to 5 subs ---
        # if gender is None:
        f0_values = []
        if len(subs)< 2:
            mean_f0 = mean_f0_for_segment(int(subs[0].start.total_seconds() * 1000), int(subs[0].end.total_seconds() * 1000))

            # --- Step 2: classify initial result ---
            if mean_f0 is None or mean_f0 <= 0:
                gender = None
            elif mean_f0 < MALE_STRONG:
                gender = "male"
            elif mean_f0 > FEMALE_STRONG:
                gender = "female"
            else:
                gender = None  # ambiguous
        else:
            for sub in subs[:5]:
                f0 = mean_f0_for_segment(int(sub.start.total_seconds() * 1000),  int(sub.end.total_seconds() * 1000))
                if f0 is not None and not np.isnan(f0) and f0 > 0:
                    f0_values.append(f0)
            f0_values = [f for f in f0_values if 80 <= f <= 350]  # sanity clamp
            if not f0_values:
                gender = None
            else:
                f0_med = float(np.median(f0_values))



                high_cnt = sum(f >= FEMALE_STRONG for f in f0_values)
                low_cnt = sum(f <= MALE_STRONG for f in f0_values)

                # optional: detect "pitch doubling" style instability
                spread = np.percentile(f0_values, 90) - np.percentile(f0_values, 10)


                if high_cnt >= max(2, int(0.7 * len(f0_values))) and low_cnt == 0 and spread < 80:
                    gender = "female"

                elif low_cnt >= 1 or f0_med <= 170:
                    gender = "male"
                elif spread < 60 and f0_med >= 175:
                    gender = "female"

                # otherwise ambiguous
                else:
                    gender = None
            # speakers[speaker] = {"gender" : gender}
        if gender is None:
            gender = "male"
        speaker_data["gender"]=gender
    return speakers_set
