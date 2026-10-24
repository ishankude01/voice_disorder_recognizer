"""Tests for the audio vowel validator component."""
import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter
from components.validator import validate_audio, DEFAULT_THRESHOLDS, INVALID_AUDIO_MESSAGE

ROOT = Path(__file__).resolve().parents[1]


def make_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Encode numpy float array into WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def make_resonator(f: float, bw: float, fs: int):
    """Biquad acoustic resonator filter."""
    r = np.exp(-np.pi * bw / fs)
    theta = 2 * np.pi * f / fs
    b = [1 - 2 * r * np.cos(theta) + r**2]
    a = [1, -2 * r * np.cos(theta), r**2]
    return b, a


def synth_formant_vowel(f1: float, f2: float, f3: float, f0: float = 135.0, duration: float = 4.0, sr: int = 16000) -> np.ndarray:
    """Synthesize a vocal-tract filtered vowel with glottal pulse train."""
    t = np.arange(int(sr * duration)) / sr
    pulse = np.zeros_like(t)
    pulse[::int(sr / f0)] = 1.0

    b1, a1 = make_resonator(f1, 70, sr)
    b2, a2 = make_resonator(f2, 90, sr)
    b3, a3 = make_resonator(f3, 120, sr)
    sound = lfilter(b3, a3, lfilter(b2, a2, lfilter(b1, a1, pulse)))
    sound = lfilter([1.0], [1.0, -0.95], sound)  # -6 dB/octave glottal spectral tilt
    return 0.25 * sound / np.max(np.abs(sound))


def test_bundled_demo_samples_pass_validation():
    """All bundled samples in samples/ (healthy, pathological, laryngeal) must be accepted."""
    samples_dir = ROOT / 'samples'
    wav_files = list(samples_dir.glob('*.wav'))
    assert len(wav_files) >= 3, "Expected at least 3 demo samples in samples/"
    
    for path in wav_files:
        content = path.read_bytes()
        res = validate_audio(content, path.name)
        assert res.is_valid, f"{path.name} failed validation: {res.rejection_reason}"
        assert res.metrics['voiced_frame_ratio'] >= 0.70
        assert res.metrics['max_continuous_voiced_sec'] >= 1.5
        assert res.metrics['rms_energy'] > 0.01
        assert res.metrics['f1_median_hz'] >= 580.0


def test_all_bundled_examples_and_demos_pass():
    """All WAV files in demo_audio/ and examples/ must pass validation."""
    other_wavs = list((ROOT / 'demo_audio').glob('*.wav')) + list((ROOT / 'examples').glob('*.wav'))
    assert len(other_wavs) >= 3, "Expected audio files in demo_audio/ and examples/"
    
    for path in other_wavs:
        content = path.read_bytes()
        res = validate_audio(content, path.name)
        assert res.is_valid, f"{path.name} failed validation: {res.rejection_reason}"
        assert res.metrics['f1_median_hz'] >= 580.0


def test_accept_clean_sustained_vowel_aaaa():
    """A clean sustained /a/ vowel ('father' / 'aaaa') should pass validation."""
    sr = 16000
    vowel = synth_formant_vowel(f1=740, f2=1180, f3=2400, f0=135, duration=4.0, sr=sr)
    res = validate_audio(make_wav(vowel, sr), 'steady_aaaa.wav')
    assert res.is_valid
    assert res.rejection_reason is None
    assert res.metrics['max_continuous_voiced_sec'] >= 2.0
    assert res.metrics['f1_median_hz'] >= 580.0


def test_accept_sustained_vowel_with_tremor():
    """Sustained /a/ with vocal tremor (high pitch IQR but steady open /a/ formants) must pass."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    # Vocal tremor: 5 Hz modulation of pitch (115 Hz to 155 Hz, IQR ~ 28 Hz)
    f0_tremor = 135.0 + 20.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0_tremor) / sr
    crossings = np.where(np.diff(np.floor(phase / (2 * np.pi))))[0]
    pulse = np.zeros_like(t)
    pulse[crossings] = 1.0

    b1, a1 = make_resonator(730, 70, sr)
    b2, a2 = make_resonator(1150, 90, sr)
    tremor_sound = lfilter(b2, a2, lfilter(b1, a1, pulse))
    tremor_sound = 0.25 * tremor_sound / np.max(np.abs(tremor_sound))

    res = validate_audio(make_wav(tremor_sound, sr), 'tremor_aaaa.wav')
    # Must NOT be rejected based on pitch variation alone
    assert res.is_valid
    assert res.rejection_reason is None


def test_reject_normal_conversation():
    """Conversational speech with sentence prosody and syllabic pauses must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    speech_f0 = 170.0 + 55.0 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(speech_f0) / sr
    # Syllabic bursts and consonant stops (3 words per second)
    words = (np.sin(2 * np.pi * 3.0 * t) > 0.0).astype(float)
    speech = 0.25 * np.sin(phase) * words

    res = validate_audio(make_wav(speech, sr), 'speech.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_vowel_eeee():
    """High front vowel 'eeee' (/i/) must be rejected due to low F1 and high F2."""
    sr = 16000
    # /i/ formants: F1 ~ 270 Hz, F2 ~ 2290 Hz
    vowel_ee = synth_formant_vowel(f1=270, f2=2290, f3=3010, f0=135, duration=4.0, sr=sr)
    res = validate_audio(make_wav(vowel_ee, sr), 'vowel_ee.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_vowel_oooo():
    """Rounded back vowel 'oooo' (/u/) must be rejected due to low F1 and low F2."""
    sr = 16000
    # /u/ formants: F1 ~ 295 Hz, F2 ~ 870 Hz
    vowel_oo = synth_formant_vowel(f1=295, f2=870, f3=2240, f0=135, duration=4.0, sr=sr)
    res = validate_audio(make_wav(vowel_oo, sr), 'vowel_oo.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_vowel_ohhh():
    """Mid-rounded vowel 'ohhh' (/o/) must be rejected due to F1 < 580 Hz."""
    sr = 16000
    # /o/ formants: F1 ~ 500 Hz, F2 ~ 900 Hz
    vowel_oh = synth_formant_vowel(f1=500, f2=900, f3=2500, f0=135, duration=4.0, sr=sr)
    res = validate_audio(make_wav(vowel_oh, sr), 'vowel_oh.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_singing_scales():
    """Singing scales / melody across intervals with moving formants must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    # Singing melody / siren: wide pitch sweep 120 Hz to 240 Hz
    sing_f0 = 180.0 + 60.0 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(sing_f0) / sr
    crossings = np.where(np.diff(np.floor(phase / (2 * np.pi))))[0]
    pulse = np.zeros_like(t)
    pulse[crossings] = 1.0

    # Moving vocal tract formants (shifting between vowels)
    f1_contour = 450.0 + 250.0 * np.sin(2 * np.pi * 0.8 * t)
    # Filter with moving harmonics
    singing = 0.25 * np.sin(phase) + 0.18 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    res = validate_audio(make_wav(singing, sr), 'singing.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_music_instrument():
    """Harmonic music instrument (e.g. violin / chords) must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    chord = 0.1 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 277 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)
    res = validate_audio(make_wav(chord, sr), 'chord.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_coughing():
    """A sudden transient cough followed by silence must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    cough = np.zeros_like(t)
    burst_len = int(sr * 0.35)
    burst = np.random.normal(0, 0.4, burst_len) * np.hanning(burst_len)
    cough[int(sr * 1.0):int(sr * 1.0) + burst_len] = burst

    res = validate_audio(make_wav(cough, sr), 'cough.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_silence():
    """Digital silence or near-zero background hum must be rejected."""
    sr = 16000
    silence = np.zeros(int(sr * 4.0))
    res = validate_audio(make_wav(silence, sr), 'silent.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_white_noise():
    """Broadband white noise must be rejected."""
    sr = 16000
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.08, int(sr * 4.0))
    res = validate_audio(make_wav(noise, sr), 'noise.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_short_audio():
    """Audio under 3.0 seconds must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 2.0)) / sr
    short_tone = 0.2 * np.sin(2 * np.pi * 150 * t)
    res = validate_audio(make_wav(short_tone, sr), 'short.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_reject_high_whistle():
    """Whistling or pure tone above human vowel pitch range must be rejected."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    whistle = 0.2 * np.sin(2 * np.pi * 950 * t)
    res = validate_audio(make_wav(whistle, sr), 'whistle.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_cautious_formant_fallback_rejects_non_vowel():
    """When formant tracking fails on mechanical rumble, low vowel band energy causes rejection."""
    sr = 16000
    t = np.arange(int(sr * 4.0)) / sr
    rumble = 0.25 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t)
    res = validate_audio(make_wav(rumble, sr), 'rumble.wav')
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE


def test_custom_threshold_override():
    """Custom thresholds can override defaults."""
    sr = 16000
    t = np.arange(int(sr * 3.5)) / sr
    tone = 0.1 * np.sin(2 * np.pi * 150 * t)
    res = validate_audio(make_wav(tone, sr), 'tone.wav', thresholds={'min_duration_sec': 5.0})
    assert not res.is_valid
    assert res.rejection_reason == INVALID_AUDIO_MESSAGE
