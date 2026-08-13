"""
Fast, model-free unit tests — safe to run in CI (no GPU, no downloaded
model, no network). Covers the two things most likely to silently break:
normalize_arabic() drifting out of sync between preprocessing and
training/eval, and manifest-building not skipping bad rows correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from dataset import normalize_arabic as normalize_arabic_dataset  # noqa: E402
from preprocess import normalize_arabic as normalize_arabic_preprocess  # noqa: E402
from preprocess import detect_column  # noqa: E402


def test_normalize_arabic_strips_diacritics():
    assert normalize_arabic_preprocess("مَرْحَبًا") == "مرحبا"


def test_normalize_arabic_unifies_alef_variants():
    assert normalize_arabic_preprocess("إنسان") == normalize_arabic_preprocess("ان" + "سان")
    assert "أ" not in normalize_arabic_preprocess("أحمد")
    assert "إ" not in normalize_arabic_preprocess("إحمد")
    assert "آ" not in normalize_arabic_preprocess("آحمد")


def test_normalize_arabic_unifies_tah_marbuta():
    assert normalize_arabic_preprocess("مدرسة") == normalize_arabic_preprocess("مدرسه")


def test_normalize_arabic_strips_non_arabic():
    out = normalize_arabic_preprocess("hello مرحبا!!!")
    assert "hello" not in out
    assert "مرحبا" in out


def test_normalize_arabic_handles_empty_and_nan():
    assert normalize_arabic_preprocess("") == ""
    assert normalize_arabic_preprocess("nan") == ""
    assert normalize_arabic_preprocess(None) == ""


def test_normalize_arabic_matches_between_preprocess_and_dataset():
    # Critical: preprocess.py normalizes when building manifests, dataset.py
    # normalizes again in the collator. If these two implementations ever
    # drift apart, WER numbers become meaningless (notebook changelog flags
    # exactly this as a past bug).
    samples = ["مَرْحَبًا يا صاحبي!", "إزيك النهارده؟", "المدرسة الجديدة"]
    for s in samples:
        assert normalize_arabic_preprocess(s) == normalize_arabic_dataset(s)


def test_detect_column_finds_known_names():
    assert detect_column(["id", "audio_file", "text"], ["text", "sentence"]) == "text"
    assert detect_column(["id", "audio_file", "text"], ["audio_file", "audio"]) == "audio_file"


def test_detect_column_case_insensitive():
    assert detect_column(["ID", "Audio_File", "Text"], ["text"]) == "Text"


def test_detect_column_returns_none_when_missing():
    assert detect_column(["id", "foo", "bar"], ["text", "sentence"]) is None
