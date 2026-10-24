"""Audio input validation for the Voice Disorder Recognizer.

Validates that audio recordings or uploads contain a sustained "aaaa" vowel
rather than conversational speech, sentences, different vowels ("eeee", "oooo"),
music, singing, coughing, silence, or unrelated noise.

Designed specifically to avoid rejecting genuine dysphonic / pathological voices:
- Evaluates combined evidence (formants, pitch stability, phonation continuity, and amplitude envelope)
  rather than relying on any single threshold (such as pitch IQR) in isolation.
- Accommodates vocal roughness, breathiness, and vocal tremor by preserving tolerance
  when steady open-vowel formants (F1, F2) and continuous phonation are confirmed.
- Employs a cautious spectral fallback if severe dysphonia disrupts Praat formant tracking.
"""
from dataclasses import dataclass, field
import io
import numpy as np
import parselmouth
import soundfile as sf
import librosa

# Standardized user-facing error message
INVALID_AUDIO_MESSAGE = "Invalid audio input. Please record a sustained ‘aaaa’ vowel for 3–5 seconds."

# Configurable validation thresholds with centrally documented defaults
DEFAULT_THRESHOLDS = {
    # 1. Duration and Level
    'min_duration_sec': 3.0,               # Minimum recording duration required
    'min_rms_energy': 0.003,               # Minimum RMS level (~ -50 dBFS; rejects silence / ambient hum)
    
    # 2. Voicing & Phonation Continuity (Rejects conversational speech, coughing, pauses)
    'min_voiced_ratio': 0.55,              # Minimum fraction of voiced frames (running speech is typically 35-50%)
    'min_continuous_voiced_sec': 1.4,      # Longest contiguous voiced run required (speech breaks every 0.2-0.5s)
    
    # 3. Fundamental Frequency (Human Range & Prosodic Stability)
    'pitch_floor_hz': 60.0,                # Praat autocorrelation pitch floor
    'pitch_ceiling_hz': 500.0,             # Praat autocorrelation pitch ceiling
    'min_human_f0_hz': 65.0,               # Minimum median pitch (adult male lower bound)
    'max_human_f0_hz': 450.0,              # Maximum median pitch (child/high female upper bound; rejects whistle)
    'max_f0_iqr_hz': 32.0,                 # Pitch IQR threshold for prosodic variation / speech intonation
    
    # 4. Vowel Formants (Identification of open /a/ vocal tract cavity)
    'min_f1_hz': 580.0,                    # Minimum F1 (open jaw / low vowel; rejects mid/closed vowels /o/, /u/, /i/)
    'max_f1_hz': 1100.0,                   # Maximum F1 for open vowel
    'min_f2_hz': 950.0,                    # Minimum F2 (central/back vowel; rejects /u/ where F2 < 900 Hz)
    'max_f2_hz': 1650.0,                   # Maximum F2 (rejects front vowels like /i/ and /e/ where F2 > 1750 Hz)
    'max_f2_f1_diff_hz': 1250.0,           # Maximum separation (F2 - F1)
    
    # 5. Formant Tracking Fallback (Cautious verification if severe dysphonia impedes Burg tracking)
    'min_vowel_band_energy_ratio': 0.45,   # Minimum energy fraction in 400-2200 Hz band if formants cannot be tracked
    
    # 6. Envelope Stability & Transients (Rejects coughing, impulsive noise)
    'max_rms_cv': 0.55,                    # Coefficient of variation of 50ms frame RMS energies
    'max_zcr_unvoiced': 0.35,              # Broadband static / friction zero-crossing rate threshold
}


@dataclass
class ValidationResult:
    """Structured result returned by audio validation."""
    is_valid: bool
    rejection_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | str] = field(default_factory=dict)


def validate_audio(
    content: bytes,
    filename: str = '',
    thresholds: dict | None = None
) -> ValidationResult:
    """Validate that audio content represents a usable sustained "aaaa" vowel attempt.
    
    Parameters
    ----------
    content : bytes
        Raw WAV or audio file bytes.
    filename : str, optional
        Name of the file for reporting.
    thresholds : dict, optional
        Override dictionary for any entries in DEFAULT_THRESHOLDS.
        
    Returns
    -------
    ValidationResult
        Object containing is_valid status, user-facing rejection reason (if invalid),
        advisory warnings (if applicable), and computed acoustic metrics.
    """
    cfg = DEFAULT_THRESHOLDS.copy()
    if thresholds:
        cfg.update(thresholds)

    # 1. Decode audio container
    try:
        with sf.SoundFile(io.BytesIO(content)) as source:
            sr = source.samplerate
            channels = source.channels
            audio = source.read(dtype='float64', always_2d=True)
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics={'diagnostic_detail': f"Could not decode audio file: {exc}"}
        )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    duration = len(audio) / max(1, sr)
    metrics: dict[str, float | str] = {
        'duration_sec': float(duration),
        'sample_rate_hz': float(sr),
        'channels': float(channels),
    }

    # 2. Duration Check
    if duration < cfg['min_duration_sec']:
        metrics['diagnostic_detail'] = f"Audio is too short ({duration:.1f}s < {cfg['min_duration_sec']:.1f}s)."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # 3. Energy / Silence Check
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    metrics['rms_energy'] = rms
    metrics['peak_amplitude'] = peak

    if rms < cfg['min_rms_energy']:
        metrics['diagnostic_detail'] = f"Audio is silent or too quiet (RMS: {rms:.5f})."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # 4. Zero-Crossing Rate & Frame-Level RMS Envelope
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=512)))
    metrics['zero_crossing_rate'] = zcr

    frame_len = max(1, int(0.05 * sr))  # 50 ms frames
    frame_rms = [np.sqrt(np.mean(audio[i:i+frame_len]**2)) for i in range(0, len(audio) - frame_len, frame_len)]
    if frame_rms:
        rms_arr = np.array(frame_rms)
        active_rms = rms_arr[rms_arr > (np.mean(rms_arr) * 0.2)]
        rms_cv = float(np.std(active_rms) / max(1e-8, np.mean(active_rms))) if len(active_rms) else 0.0
    else:
        rms_cv = 0.0
    metrics['rms_cv'] = rms_cv

    # 5. Pitch and Phonation Continuity via Praat Parselmouth
    try:
        sound = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = sound.to_pitch_ac(
            time_step=0.01,
            pitch_floor=cfg['pitch_floor_hz'],
            pitch_ceiling=cfg['pitch_ceiling_hz']
        )
        f0_array = pitch.selected_array['frequency']
        voiced_mask = (f0_array > 0) & np.isfinite(f0_array)
        voiced_f0 = f0_array[voiced_mask]
        voiced_ratio = float(len(voiced_f0) / max(1, len(f0_array)))
    except Exception:
        voiced_f0 = np.array([])
        voiced_mask = np.zeros(1, dtype=bool)
        voiced_ratio = 0.0

    metrics['voiced_frame_ratio'] = voiced_ratio

    # Calculate longest contiguous voiced segment
    runs: list[float] = []
    current_run = 0
    for is_voiced in voiced_mask:
        if is_voiced:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run * 0.01)
            current_run = 0
    if current_run > 0:
        runs.append(current_run * 0.01)
    max_continuous_voiced = max(runs) if runs else 0.0
    metrics['max_continuous_voiced_sec'] = max_continuous_voiced

    # Hard rejection for non-vocal broadband static/noise
    if voiced_ratio < 0.30 and zcr > cfg['max_zcr_unvoiced']:
        metrics['diagnostic_detail'] = "Static, hiss, or broadband background noise."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # Rejection for coughing / brief transient noise
    if max_continuous_voiced < 0.6 and (rms_cv > cfg['max_rms_cv'] or voiced_ratio < 0.35):
        metrics['diagnostic_detail'] = "Cough or transient acoustic impulse."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # If voicing was entirely undetected
    if len(voiced_f0) == 0:
        metrics['diagnostic_detail'] = "No vocal pitch/tone detected."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    f0_median = float(np.median(voiced_f0))
    f0_iqr = float(np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25))
    metrics['f0_median_hz'] = f0_median
    metrics['f0_iqr_hz'] = f0_iqr

    # Pitch outside human singing/speaking range (whistle, high harmonics, low mechanical hum)
    if f0_median < cfg['min_human_f0_hz'] or f0_median > cfg['max_human_f0_hz']:
        metrics['diagnostic_detail'] = f"Pitch ({f0_median:.1f} Hz) outside human range (65-450 Hz)."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # Discontinuous phonation (pauses, broken vowel attempt, or intermittent phonation)
    if max_continuous_voiced < cfg['min_continuous_voiced_sec']:
        metrics['diagnostic_detail'] = f"Interrupted phonation (longest continuous run {max_continuous_voiced:.1f}s < {cfg['min_continuous_voiced_sec']:.1f}s)."
        return ValidationResult(
            is_valid=False,
            rejection_reason=INVALID_AUDIO_MESSAGE,
            metrics=metrics
        )

    # 6. Vowel Formants (Identification of Open /a/ Vowel Cavity)
    warnings_list: list[str] = []
    max_formant = min(5500.0, sr * 0.48)  # Safeguard against Nyquist boundary for 8 kHz audio
    f1_list, f2_list = [], []
    try:
        formants = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=max_formant
        )
        # Sample formants at voiced points in the central region of the file
        t_start = max(0.2, duration * 0.15)
        t_end = min(duration - 0.2, duration * 0.85)
        times_to_sample = np.linspace(t_start, t_end, 30)
        
        for t in times_to_sample:
            f1 = formants.get_value_at_time(1, float(t))
            f2 = formants.get_value_at_time(2, float(t))
            if not np.isnan(f1) and f1 > 0:
                f1_list.append(f1)
            if not np.isnan(f2) and f2 > 0:
                f2_list.append(f2)
    except Exception:
        f1_list, f2_list = [], []

    # 7. Formant Evaluation & Combined Evidence Check
    has_formants = len(f1_list) >= 5 and len(f2_list) >= 5
    if has_formants:
        f1_med = float(np.median(f1_list))
        f2_med = float(np.median(f2_list))
        f2_f1_diff = f2_med - f1_med
        metrics['f1_median_hz'] = f1_med
        metrics['f2_median_hz'] = f2_med
        metrics['f2_f1_diff_hz'] = f2_f1_diff

        # Check for non-/a/ vowel cavities:
        # High/closed/mid vowels (e.g. /i/, /u/, /o/, /e/) have low F1 (< 580 Hz)
        if f1_med < cfg['min_f1_hz']:
            metrics['diagnostic_detail'] = f"Non-/a/ vowel: F1 ({f1_med:.0f} Hz) < {cfg['min_f1_hz']:.0f} Hz."
            return ValidationResult(
                is_valid=False,
                rejection_reason=INVALID_AUDIO_MESSAGE,
                metrics=metrics
            )
        if f1_med > cfg['max_f1_hz']:
            metrics['diagnostic_detail'] = f"Non-/a/ vowel: F1 ({f1_med:.0f} Hz) > {cfg['max_f1_hz']:.0f} Hz."
            return ValidationResult(
                is_valid=False,
                rejection_reason=INVALID_AUDIO_MESSAGE,
                metrics=metrics
            )

        # Front vowels (e.g. /i/ 'ee', /e/ 'eh') have high F2 (> 1650 Hz) or wide F2-F1 separation
        if f2_med > cfg['max_f2_hz'] or f2_f1_diff > cfg['max_f2_f1_diff_hz']:
            metrics['diagnostic_detail'] = f"Front vowel: F2 ({f2_med:.0f} Hz) > {cfg['max_f2_hz']:.0f} Hz."
            return ValidationResult(
                is_valid=False,
                rejection_reason=INVALID_AUDIO_MESSAGE,
                metrics=metrics
            )

        # Rounded back vowels (e.g. /u/ 'oo') have low F2 (< 950 Hz)
        if f2_med < cfg['min_f2_hz']:
            metrics['diagnostic_detail'] = f"Rounded back vowel: F2 ({f2_med:.0f} Hz) < {cfg['min_f2_hz']:.0f} Hz."
            return ValidationResult(
                is_valid=False,
                rejection_reason=INVALID_AUDIO_MESSAGE,
                metrics=metrics
            )

        # Combined Evidence: Pitch Variation & Formant Stability (Singing / Prosodic speech)
        # In conversational speech or singing melodies, F0 IQR is wide AND formants drift over time.
        # But in pure vocal tremor, formants remain stable on open /a/.
        f1_iqr = float(np.percentile(f1_list, 75) - np.percentile(f1_list, 25))
        metrics['f1_iqr_hz'] = f1_iqr
        if f0_iqr > cfg['max_f0_iqr_hz']:
            # If pitch fluctuates widely AND formants are moving (singing / speech), reject
            if f1_iqr > 60.0 or voiced_ratio < cfg['min_voiced_ratio'] or rms_cv > 0.35:
                metrics['diagnostic_detail'] = f"Singing or conversational speech (F0 IQR: {f0_iqr:.1f} Hz, F1 IQR: {f1_iqr:.1f} Hz)."
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=INVALID_AUDIO_MESSAGE,
                    metrics=metrics
                )
            else:
                # Vocal tremor with steady open /a/ vocal tract: accept with advisory notice
                warnings_list.append(
                    f"Pitch variation detected (F0 IQR: {f0_iqr:.1f} Hz). Sustained /a/ vowel verified; proceeding to analysis."
                )
    else:
        # Formant tracking failed (e.g. severe dysphonia or non-vocal audio).
        # Cautious Fallback: Measure spectral band energy concentration in the 400-2200 Hz open-vowel region.
        spec = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
        vowel_band_mask = (freqs >= 400.0) & (freqs <= 2200.0)
        total_energy = float(np.sum(spec ** 2))
        band_energy = float(np.sum(spec[vowel_band_mask] ** 2))
        vowel_band_ratio = band_energy / max(1e-12, total_energy)
        metrics['vowel_band_energy_ratio'] = vowel_band_ratio

        if vowel_band_ratio < cfg['min_vowel_band_energy_ratio']:
            metrics['diagnostic_detail'] = f"Lacks vowel resonance (energy in 400-2200 Hz: {vowel_band_ratio:.1%})."
            return ValidationResult(
                is_valid=False,
                rejection_reason=INVALID_AUDIO_MESSAGE,
                metrics=metrics
            )
        
        # Audio has continuous phonation and strong energy in vowel region, but glottal irregularity disrupted Burg tracking
        warnings_list.append(
            "Vocal formants could not be tracked cleanly due to acoustic roughness or breathiness, "
            "but continuous vocal energy was verified. Analysis will proceed."
        )

    # Supporting advisory notice for moderate pitch fluctuations (e.g. vocal tremor)
    if f0_iqr > 20.0 and not warnings_list:
        warnings_list.append(
            f"Mild pitch fluctuation detected (F0 IQR: {f0_iqr:.1f} Hz). "
            "For highest accuracy, sustain a steady note without varying your pitch."
        )

    return ValidationResult(
        is_valid=True,
        rejection_reason=None,
        warnings=warnings_list,
        metrics=metrics
    )
