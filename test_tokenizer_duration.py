"""
Standalone test: estimate Index-TTS 2.5 audio duration from text.

Index-TTS 2.5 pipeline:
  text → tiktoken tokens → GPT mel codes → s2mel expansion → BigVGAN audio

Known constants from the model architecture:
  - output sample_rate = 22050 (s2mel preprocess_params.sr)
  - BigVGAN hop_length = 256 (bigvgan_v2_22khz_80band_256x)
  - s2mel expansion = 1.72 (code_lens * 1.72 * duration_factor = mel frames)
  - duration_factor = 1.0 (configurable 0.5-2.0)

So: duration_sec = num_mel_codes * 1.72 * duration_factor * 256 / 22050

The GPT's text-token-to-mel-code ratio is NOT deterministic — it's autoregressive.
We must calibrate it empirically from generated audio.

The auxiliary models (w2v-bert, BigVGAN, CAMPPlus) do NOT help with duration
estimation — they handle speaker embedding, mel→wav conversion, and speaker
verification respectively. None provide a token-to-time mapping.
"""

import os
import sys
import re
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "index-tts"))

import tiktoken

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "index-tts", "checkpoints")
TIKTOKEN_PATH = os.path.join(CHECKPOINTS_DIR, "multilingual_zh_ja_yue_char_del.tiktoken")

OUTPUT_SAMPLE_RATE = 22050
S2MEL_EXPANSION = 1.72
BIGVGAN_HOP_LENGTH = 256

# Empirical duration model (multivariate regression on 46 English segments).
# Model: duration = INTERCEPT + tokens * (BASE_RATE + sum(emo_adj_i * emo_i))
# MAE: 0.35s, 72% within 0.5s, 98% within 1.0s
INTERCEPT = 0.83
BASE_RATE = 0.276  # sec/token baseline

# Emotion adjustment coefficients (added to BASE_RATE, multiplied by tokens).
# emo_vector order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMO_RATE_ADJUSTMENTS = [
    +0.009,   # happy: negligible
    -0.071,   # angry: slightly faster
    +0.208,   # sad: much slower (+75%)
    -0.151,   # afraid: faster (panicked rush)
    -0.055,   # disgusted: slightly faster
    -0.058,   # melancholic: (correlated with sad, captured there)
    +0.039,   # surprised: slightly slower
    -0.091,   # calm: faster (efficient delivery)
]

# Punctuation replacement map (same as TextNormalizer.char_rep_map in v2.5)
CHAR_REP_MAP = {
    "：": ",", "；": ",", ";": ",", "，": ",", "。": ".",
    "！": "!", "？": "?", "\n": " ", "·": "-", "、": ",",
    "...": "…", ",,,": "…", "，，，": "…",
    "……": "…", "“": "'", "”": "'", '"': "'",
    "‘": "'", "’": "'", "（": "'", "）": "'",
    "(": "'", ")": "'", "《": "'", "》": "'", "【": "'",
    "】": "'", "[": "'", "]": "'", "—": "-", "～": "-",
    "~": "-", "「": "'", "」": "'", ":": ",",
}
_CLEAN_PATTERN = re.compile("|".join(re.escape(p) for p in CHAR_REP_MAP.keys()))

# Language tokens supported by v2.5
LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
]


def _build_tiktoken_encoding() -> tiktoken.Encoding:
    """Load the Index-TTS 2.5 tiktoken encoding (same logic as indextts/utils/tokenizer.py)."""
    ranks = {
        base64.b64decode(token): int(rank)
        for token, rank in (line.split() for line in open(TIKTOKEN_PATH) if line.strip())
    }
    n_vocab = len(ranks)
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES[:99]],
        *[f"<|{ev}|>" for ev in ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM",
                                   "Laughter", "/Laughter", "Applause", "/Applause"]],
        *[f"<|{em}|>" for em in ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 31)],
        *[f"<|{tts}|>" for tts in [
            "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
            *[f"TTS/SP{i:02d}" for i in range(1, 14)]
        ]],
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]

    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name="multilingual_zh_ja_yue_char_del",
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


class IndexTTS25Tokenizer:
    def __init__(self):
        self.encoding = _build_tiktoken_encoding()

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    def clean_text(self, text: str) -> str:
        """Apply the same char_rep_map cleanup as infer_v2_5.py line 702."""
        return _CLEAN_PATTERN.sub(lambda x: CHAR_REP_MAP[x.group()], text)

    def encode(self, text: str, lang: str = "en") -> list[int]:
        """Encode text the same way infer_v2_5.py does before passing to GPT."""
        text = self.clean_text(text)
        # v2.5 lowercases for zh/en/ja, uppercases for es
        if lang.lower() in ["zh", "zhen", "en", "ja"]:
            text = text.lower()
        elif lang.lower() == "es":
            text = text.upper()
        lang_prefix = f"<|{lang.lower()}|> "
        full_text = lang_prefix + text
        return self.encoding.encode(full_text, allowed_special="all")

    def count_tokens(self, text: str, lang: str = "en") -> int:
        """Count tokens (including lang prefix token)."""
        return len(self.encode(text, lang))

    def estimate_duration_sec(
        self,
        text: str,
        lang: str = "en",
        emo_vector: list[float] | None = None,
        duration_factor: float = 1.0,
    ) -> float:
        """
        Estimate audio duration in seconds.

        Model: duration = (INTERCEPT + tokens * effective_rate) * duration_factor
        Where: effective_rate = BASE_RATE + sum(emo_adj_i * emo_i)
        """
        n_tokens = self.count_tokens(text, lang)
        effective_rate = BASE_RATE
        if emo_vector:
            for adj, val in zip(EMO_RATE_ADJUSTMENTS, emo_vector):
                effective_rate += adj * val
        baseline = INTERCEPT + n_tokens * effective_rate
        return baseline * duration_factor

    def estimate_duration_factor(
        self,
        text: str,
        target_duration_sec: float,
        lang: str = "en",
        emo_vector: list[float] | None = None,
    ) -> float:
        """
        Given a target duration, calculate what duration_factor to pass to infer().
        Returns a value clamped to [0.5, 2.0].
        """
        baseline = self.estimate_duration_sec(text, lang, emo_vector, duration_factor=1.0)
        if baseline <= 0:
            return 1.0
        factor = target_duration_sec / baseline
        return max(0.5, min(2.0, factor))


def measure_from_output(audio_path: str, text: str, tokenizer: IndexTTS25Tokenizer, lang: str = "en") -> dict:
    """
    Given a generated audio file and its source text, return actual duration
    and token count for calibration.
    """
    import torchaudio

    audio, sr = torchaudio.load(audio_path)
    duration_sec = audio.shape[1] / sr
    n_tokens = tokenizer.count_tokens(text, lang)

    return {"duration_sec": duration_sec, "n_tokens": n_tokens}


if __name__ == "__main__":
    import json, srt, glob
    import torchaudio

    tokenizer = IndexTTS25Tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Duration model: {INTERCEPT}s + tokens * ({BASE_RATE} + emo_adjustments)")
    print(f"Emo adjustments: {dict(zip(['happy','angry','sad','afraid','disgusted','melancholic','surprised','calm'], EMO_RATE_ADJUSTMENTS))}")
    print()

    # Load real data
    subs_path = 'output/20260507_wdbk/data/subtitles_translated.srt'
    emo_path = 'output/20260507_wdbk/data/emotions_tags.json'

    with open(subs_path) as f:
        subs = {s.index: s for s in srt.parse(f.read())}
    with open(emo_path) as f:
        emotions = json.load(f)

    wavs = glob.glob('output/20260507_wdbk/audio/tts_segments/*/[0-9]*.wav')
    wavs = {int(os.path.basename(w).replace('.wav', '')): w for w in wavs if 'loudness' not in w}

    print(f"{'idx':>3} | {'tok':>3} | {'est':>5} | {'act':>5} | {'err':>6} | {'%err':>5} | {'emo_cat':10} | text")
    print("-" * 110)

    abs_errors = []
    for idx_str, emo_data in sorted(emotions.items(), key=lambda x: int(x[0])):
        idx = int(idx_str)
        if idx not in wavs or idx not in subs:
            continue
        sub = subs[idx]
        raw = sub.content.strip()
        text = raw.split(':', 1)[1].strip() if ':' in raw else raw
        if not text:
            continue

        audio, sr = torchaudio.load(wavs[idx])
        actual_dur = audio.shape[1] / sr
        emo_vec = emo_data.get('emo_vector', [0]*8)
        category = emo_data.get('category', '?')

        n_tokens = tokenizer.count_tokens(text, 'en')
        estimated = tokenizer.estimate_duration_sec(text, 'en', emo_vector=emo_vec)
        error = estimated - actual_dur
        pct_err = (error / actual_dur * 100) if actual_dur > 0 else 0
        abs_errors.append(abs(error))

        print(f"{idx:3d} | {n_tokens:3d} | {estimated:5.2f} | {actual_dur:5.2f} | {error:+6.2f} | {pct_err:+5.1f}% | {category:10} | {text[:45]}")

    print()
    print("=" * 110)
    import statistics
    print(f"Total: {len(abs_errors)} segments")
    print(f"MAE: {statistics.mean(abs_errors):.2f}s")
    print(f"Median AE: {statistics.median(abs_errors):.2f}s")
    print(f"Within 0.5s: {sum(1 for e in abs_errors if e <= 0.5)}/{len(abs_errors)} ({sum(1 for e in abs_errors if e <= 0.5)/len(abs_errors)*100:.0f}%)")
    print(f"Within 1.0s: {sum(1 for e in abs_errors if e <= 1.0)}/{len(abs_errors)} ({sum(1 for e in abs_errors if e <= 1.0)/len(abs_errors)*100:.0f}%)")
    print(f"Within 1.5s: {sum(1 for e in abs_errors if e <= 1.5)}/{len(abs_errors)} ({sum(1 for e in abs_errors if e <= 1.5)/len(abs_errors)*100:.0f}%)")
