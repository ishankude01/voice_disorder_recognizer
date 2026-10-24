"""Train GMM and MLP on local SVD WAVs; keep speakers disjoint throughout.

Run: python svd_training.py train --output artifacts_svd_local
Predict: python svd_training.py predict sample.wav --model artifacts_svd_local/svd_voice_model.joblib
Requires scikit-learn >= 1.7 for weighted MLP fitting. No Streamlit dependency.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import warnings
from pathlib import Path
from math import gcd

import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import scipy
from scipy.signal import resample_poly
from scipy.special import softmax
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
import soundfile as sf

DEFAULT_DATA = Path(r"C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset")
DEFAULT_MODEL = Path(__file__).resolve().parent / "artifacts_svd_local/svd_voice_model.joblib"
SPEC = {"sample_rate": 16000, "min_seconds": 0.5, "max_seconds": 5.0,
        "crop": "central", "normalization": "DC removal then peak normalization",
        "n_mfcc": 13, "n_fft": 512, "win_length": 400, "hop_length": 160,
        "n_mels": 40, "pitch_floor": 60, "pitch_ceiling": 500, "version": 1}
FEATURE_NAMES = ([f"mfcc_{i}_{stat}" for stat in ("mean", "std") for i in range(1, 14)]
                 + [f"{name}_{stat}" for name in ("centroid", "spread", "rolloff", "rms", "zcr")
                    for stat in ("mean", "std")]
                 + ["f0_median", "f0_std", "f0_iqr", "voiced_fraction", "hnr_db",
                    "frame_amplitude_variation_proxy"])


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_audio(audio_file, spec=SPEC):
    """Accept a path, bytes, BytesIO, or Streamlit UploadedFile. Bound decoding.

    Peak normalization makes energy relative; it removes absolute loudness information.
    No padding or tiling of short files. Reject files shorter than half a second.
    """
    if isinstance(audio_file, (bytes, bytearray)):
        audio_file = io.BytesIO(audio_file)
    elif hasattr(audio_file, "getvalue"):
        audio_file = io.BytesIO(audio_file.getvalue())
    try:
        with sf.SoundFile(audio_file) as source:
            if source.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError("Please provide WAV audio.")
            sr = source.samplerate
            if not 8000 <= sr <= 192000 or source.channels not in (1, 2):
                raise ValueError("Use mono/stereo WAV sampled between 8 and 192 kHz.")
            duration = source.frames / sr
            if duration < spec["min_seconds"]:
                raise ValueError(f"Record at least {spec['min_seconds']} seconds of a steady vowel.")
            count = min(source.frames, int(spec["max_seconds"] * sr))
            source.seek((source.frames - count) // 2)
            x = source.read(count, dtype="float64", always_2d=True).mean(axis=1)
    except (RuntimeError, OSError) as exc:
        raise ValueError("Cannot decode this WAV file.") from exc
    if not np.isfinite(x).all():
        raise ValueError("Audio contains nonfinite samples.")
    x -= x.mean()
    if np.sqrt(np.mean(x ** 2)) < 1e-7:
        raise ValueError("Audio is silent or too quiet.")
    target = spec["sample_rate"]
    if sr != target:
        common = gcd(sr, target)
        x = resample_poly(x, target // common, sr // common)
    x -= x.mean()
    x /= max(np.max(np.abs(x)), 1e-12)
    return x, {"original_sample_rate": sr, "original_seconds": duration,
               "analyzed_seconds": len(x)/target}


def extract_features(audio_file, spec=SPEC):
    """Same deterministic feature path in training and inference.

    HNR and F0 use Praat. The amplitude proxy is frame RMS change, NOT clinical
    cycle-to-cycle shimmer. Unvoiced/failed voice measurements remain NaN until
    the training-only imputer is applied.
    """
    x, meta = load_audio(audio_file, spec)
    sr = spec["sample_rate"]
    kwargs = dict(n_fft=spec["n_fft"], hop_length=spec["hop_length"],
                  win_length=spec["win_length"], center=False)
    magnitude = np.abs(librosa.stft(x, **kwargs))
    mel = librosa.feature.melspectrogram(S=magnitude**2, sr=sr, n_fft=spec["n_fft"],
                                        n_mels=spec["n_mels"], fmin=50, fmax=sr/2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=1.0), n_mfcc=spec["n_mfcc"])
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    spread = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr, centroid=centroid, p=2)
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=.85)
    rms = librosa.feature.rms(y=x, frame_length=spec["win_length"],
                              hop_length=spec["hop_length"], center=False)[0]
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=spec["win_length"],
                                            hop_length=spec["hop_length"], center=False)
    values = [*mfcc.mean(axis=1), *mfcc.std(axis=1)]
    for array in (centroid, spread, rolloff, rms, zcr):
        values.extend([float(np.mean(array)), float(np.std(array))])
    f0_stats, hnr = [np.nan] * 4, np.nan
    sound = parselmouth.Sound(x, sampling_frequency=sr)
    failures = []
    try:
        pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=spec["pitch_floor"],
                                  pitch_ceiling=spec["pitch_ceiling"])
        f0 = pitch.selected_array["frequency"]
        voiced = f0[(f0 > 0) & np.isfinite(f0)]
        f0_stats[3] = len(voiced) / max(1, len(f0))
        if len(voiced):
            f0_stats[:3] = [float(np.median(voiced)), float(np.std(voiced)),
                            float(np.percentile(voiced, 75)-np.percentile(voiced, 25))]
        else:
            failures.append("No voiced F0 frames")
    except parselmouth.PraatError as exc:
        failures.append(str(exc))
    try:
        harmonics = sound.to_harmonicity_cc(time_step=.01, minimum_pitch=spec["pitch_floor"])
        valid = harmonics.values[np.isfinite(harmonics.values) & (harmonics.values > -199)]
        if valid.size:
            hnr = float(valid.mean())
        else:
            failures.append("HNR undefined")
    except parselmouth.PraatError as exc:
        failures.append(str(exc))
    # Frame-amplitude variation proxy: successive RMS difference / mean RMS.
    proxy = float(np.mean(np.abs(np.diff(rms))) / max(float(rms.mean()), 1e-12))
    values.extend([*f0_stats, hnr, proxy])
    features = np.asarray(values, dtype=float)
    if len(features) != len(FEATURE_NAMES) or np.isinf(features).any():
        raise ValueError("Invalid acoustic feature vector.")
    return features, {**meta, "measurement_warnings": failures}


def assert_disjoint(frame, left, right):
    if set(frame.iloc[left].speaker_id) & set(frame.iloc[right].speaker_id):
        raise RuntimeError("Speaker leakage detected.")
    if set(frame.iloc[left].audio_hash) & set(frame.iloc[right].audio_hash):
        raise RuntimeError("Duplicate audio leakage detected.")


def grouped_partition(frame, y, folds, seed):
    # Fold 0 is fixed in advance, never selected using model performance.
    split = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    left, right = next(split.split(frame, y, groups=frame.speaker_id))
    assert_disjoint(frame, left, right)
    expected = set(np.unique(y))
    if set(y[left]) != expected or set(y[right]) != expected:
        raise ValueError("A partition lacks a class. Obtain more speakers or merge labels explicitly.")
    return left, right


def audit_and_extract(audio_dir, csv_path, output):
    frame = pd.read_csv(csv_path, dtype=str).rename(columns={
        "File": "file", "Speaker ID": "speaker_id", "Diagnosis": "diagnosis"})
    required = ["file", "speaker_id", "diagnosis"]
    if not set(required) <= set(frame.columns):
        raise ValueError("CSV must contain File, Speaker ID, and Diagnosis.")
    if frame[required].isna().any().any():
        raise ValueError("Missing filename, speaker ID, or diagnosis; fix the CSV first.")
    frame[required] = frame[required].apply(lambda col: col.str.strip())
    if (frame[required] == "").any().any() or frame.file.duplicated().any():
        raise ValueError("Empty labels or duplicate CSV filenames; resolve before training.")
    audit = frame.groupby("diagnosis").agg(recordings=("file", "size"), unique_speakers=("speaker_id", "nunique"))
    audit.to_csv(output / "dataset_audit.csv")
    print(audit.to_string(), flush=True)
    frame[frame.speaker_id.isin(frame.groupby("speaker_id").diagnosis.nunique().loc[lambda s:s>1].index)].to_csv(
        output / "speakers_with_multiple_diagnoses.csv", index=False)
    accepted, features, rejected = [], [], []
    seen = {}
    audio_dir = Path(audio_dir).resolve()
    cache_dir = output / "feature_cache"
    cache_dir.mkdir(exist_ok=True)
    signature = hashlib.sha256((Path(__file__).read_text(encoding="utf-8") + json.dumps(SPEC)).encode()).hexdigest()
    for i, row in enumerate(frame.to_dict("records")):
        try:
            path = (audio_dir / row["file"]).resolve()
            if not path.is_relative_to(audio_dir):
                raise ValueError("Audio path must stay inside the supplied audio directory")
            raw_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            cache = cache_dir / f"{raw_hash}_{signature}.npz"
            if cache.exists():
                with np.load(cache, allow_pickle=False) as z:
                    f, metadata, ahash = z["features"], json.loads(str(z["metadata"])), str(z["audio_hash"])
            else:
                f, metadata = extract_features(path)
                x, _ = load_audio(path)
                ahash = hashlib.sha256(np.round(x*1e6).astype("<i4").tobytes()).hexdigest()
                np.savez(cache, features=f, metadata=json.dumps(metadata), audio_hash=ahash)
            if ahash in seen:
                previous = seen[ahash]
                if previous["speaker_id"] != row["speaker_id"] or previous["diagnosis"] != row["diagnosis"]:
                    raise RuntimeError(f"Identical analyzed audio has conflicting identity/label: {previous['file']} / {row['file']}")
                raise ValueError("Repeated identical recording from same speaker; retained first only")
            seen[ahash] = row
            accepted.append({**row, "audio_hash": ahash, **metadata})
            features.append(f)
        except (ValueError, OSError, parselmouth.PraatError) as exc:
            rejected.append({**row, "reason": str(exc)})
        if (i+1) % 100 == 0:
            print(f"Features: {i+1}/{len(frame)}; accepted {len(accepted)}", flush=True)
    pd.DataFrame(rejected, columns=required+["reason"]).to_csv(output / "excluded_files.csv", index=False)
    if not accepted:
        raise ValueError("No usable audio recordings.")
    valid = pd.DataFrame(accepted)
    valid.to_csv(output / "usable_recordings.csv", index=False)
    valid.groupby("diagnosis").agg(recordings=("file", "size"), unique_speakers=("speaker_id", "nunique")).to_csv(output / "usable_class_counts.csv")
    if valid.groupby("diagnosis").speaker_id.nunique().min() < 10:
        raise ValueError("Fewer than 10 usable speakers in a class; merge labels or gather data first.")
    return valid, np.stack(features)


def fit_bundle(x, y, encoder, config, seed):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x))
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite model input after imputation.")
    k = len(encoder.classes_)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        if config["family"] == "GMM":
            models = []
            for label in range(k):
                model = GaussianMixture(n_components=config["components"], covariance_type="diag",
                                        reg_covar=.05, n_init=3, max_iter=300, random_state=seed)
                model.fit(z[y == label])
                if not model.converged_:
                    raise RuntimeError("GMM did not converge; this configuration is not accepted.")
                models.append(model)
            # Equal class priors handle class imbalance without duplicating test data.
            model = models
        else:
            model = MLPClassifier(hidden_layer_sizes=tuple(config["layers"]), activation="relu",
                                  solver="adam", alpha=.01, batch_size=min(32, len(y)),
                                  learning_rate_init=.001, max_iter=300, early_stopping=False,
                                  n_iter_no_change=25, random_state=seed)
            # Built-in early stopping is deliberately disabled: its random split is not grouped.
            model.fit(z, y, sample_weight=compute_sample_weight("balanced", y))
    return {"family": config["family"], "model": model, "scaler": scaler,
            "imputer": imputer, "label_encoder": encoder, "feature_spec": SPEC.copy(),
            "feature_names": FEATURE_NAMES, "config": config, "seed": seed,
            "warnings": [str(m.message) for m in messages], "probabilities_calibrated": False,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "scipy": scipy.__version__, "librosa": librosa.__version__}}


def probabilities(bundle, x):
    z = bundle["scaler"].transform(bundle["imputer"].transform(x))
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite features after saved preprocessing.")
    if bundle["family"] == "GMM":
        result = softmax(np.column_stack([m.score_samples(z) for m in bundle["model"]]), axis=1)
    else:
        result = bundle["model"].predict_proba(z)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction scores.")
    return result


def metrics(y, probability, names):
    predicted = np.argmax(probability, axis=1)
    return {"balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "per_class": classification_report(y, predicted, labels=np.arange(len(names)),
                                                target_names=list(names), output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted, labels=np.arange(len(names))).tolist()}


def speaker_bootstrap(y, predicted, speakers, repetitions=1000, seed=20261022):
    """Cluster bootstrap: sample speakers, include all recordings for each draw."""
    rng = np.random.default_rng(seed)
    unique = np.unique(speakers)
    groups = {s: np.flatnonzero(speakers == s) for s in unique}
    values = []
    for _ in range(repetitions):
        idx = np.concatenate([groups[s] for s in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) != len(np.unique(y)):
            continue
        values.append(balanced_accuracy_score(y[idx], predicted[idx]))
    return {"balanced_accuracy_95_ci": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
            "valid_bootstrap_replicates": len(values), "unit": "speaker",
            "note": "Conditional on this fitted model and split; metrics weight recordings."}


def train(audio_dir, labels, output, seed=20261022):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never repeatedly inspect the same test set while changing configurations.
    if (output / "test_evaluation_started.json").exists():
        raise RuntimeError("This run's test evaluation already started. Reuse its saved results; do not tune on them.")
    frame, x = audit_and_extract(audio_dir, labels, output)
    encoder = LabelEncoder().fit(frame.diagnosis)
    y = encoder.transform(frame.diagnosis)
    dev, test = grouped_partition(frame, y, folds=5, seed=seed)
    inner_train, inner_validation = grouped_partition(frame.iloc[dev].reset_index(drop=True), y[dev], folds=4, seed=seed+1)
    fit, validation = dev[inner_train], dev[inner_validation]
    split = {"seed": seed, "method": "StratifiedGroupKFold: fixed fold 0; outer 5 folds, inner 4 folds",
             "development": frame.iloc[dev].file.tolist(), "test": frame.iloc[test].file.tolist(),
             "inner_train": frame.iloc[fit].file.tolist(), "inner_validation": frame.iloc[validation].file.tolist()}
    split_path = output / "splits.json"
    if split_path.exists() and json.loads(split_path.read_text()) != split:
        raise RuntimeError("Frozen split does not match this dataset/seed. Do not overwrite it.")
    write_json(split_path, split)
    partitions = frame[["file", "speaker_id", "diagnosis", "audio_hash"]].copy()
    partitions["partition"] = "test"
    partitions.loc[dev, "partition"] = "development"
    partitions["inner_partition"] = "not used"
    partitions.loc[fit, "inner_partition"] = "train"
    partitions.loc[validation, "inner_partition"] = "validation"
    partitions.to_csv(output / "partition_manifest.csv", index=False)
    partitions.groupby(["partition", "diagnosis"]).agg(recordings=("file", "size"),
                                                        unique_speakers=("speaker_id", "nunique")).to_csv(output / "split_counts.csv")
    configs = [{"family": "GMM", "components": n} for n in (1, 2)] + [
        {"family": "DNN", "layers": layers} for layers in ([64, 32], [128, 64])]
    candidates = []
    with threadpool_limits(limits=2):
        for config in configs:
            print(f"Training on inner training speakers: {config}", flush=True)
            bundle = fit_bundle(x[fit], y[fit], encoder, config, seed)
            score = metrics(y[validation], probabilities(bundle, x[validation]), encoder.classes_)
            candidates.append({"config": config, "balanced_accuracy": score["balanced_accuracy"],
                               "macro_f1": score["macro_f1"], "warnings": bundle["warnings"]})
            print(f"Validation balanced accuracy: {score['balanced_accuracy']:.6f}", flush=True)
        write_json(output / "validation_results.json", candidates)
        chosen = {family: max((r for r in candidates if r["config"]["family"] == family),
                              key=lambda r: r["balanced_accuracy"]) for family in ("GMM", "DNN")}
        winner = max(chosen, key=lambda family: chosen[family]["balanced_accuracy"])
        write_json(output / "selection.json", {"selected_family": winner, "selection_criterion": "inner validation balanced accuracy",
                                               "selected_configurations": chosen})
        bundles = {}
        for family, candidate in chosen.items():
            print(f"Refitting {family} on {len(dev)} development recordings", flush=True)
            bundle = fit_bundle(x[dev], y[dev], encoder, candidate["config"], seed)
            bundle["training_speakers"] = sorted(frame.iloc[dev].speaker_id.unique().tolist())
            bundle["training_audio_hashes"] = frame.iloc[dev].audio_hash.tolist()
            bundle["training_files"] = frame.iloc[dev].file.tolist()
            bundle["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            joblib.dump(bundle, output / f"svd_{family.lower()}.joblib")
            restored = joblib.load(output / f"svd_{family.lower()}.joblib")
            np.testing.assert_allclose(probabilities(bundle, x[validation[:3]]), probabilities(restored, x[validation[:3]]))
            bundles[family] = bundle
        joblib.dump(bundles[winner], output / "svd_voice_model.joblib")
        write_json(output / "test_evaluation_started.json", {"selected_family": winner, "test_speakers": int(frame.iloc[test].speaker_id.nunique())})
        summary = []
        for family, bundle in bundles.items():
            probability = probabilities(bundle, x[test])
            report = metrics(y[test], probability, encoder.classes_)
            report.update(speaker_bootstrap(y[test], probability.argmax(axis=1), frame.iloc[test].speaker_id.to_numpy(), seed=seed))
            write_json(output / f"test_metrics_{family.lower()}.json", report)
            predictions = frame.iloc[test][["file", "speaker_id", "diagnosis"]].reset_index(drop=True)
            predictions["predicted_diagnosis"] = encoder.inverse_transform(probability.argmax(axis=1))
            for j, name in enumerate(encoder.classes_):
                predictions[f"score_{name}"] = probability[:, j]
            predictions.to_csv(output / f"test_predictions_{family.lower()}.csv", index=False)
            pd.DataFrame(report["per_class"]).T.to_csv(output / f"classification_report_{family.lower()}.csv")
            fig, ax = plt.subplots(figsize=(10, 8))
            ConfusionMatrixDisplay(np.array(report["confusion_matrix"]), display_labels=encoder.classes_).plot(
                ax=ax, xticks_rotation=35, colorbar=False, cmap="Blues")
            ax.set_title(f"{family}: reserved test speakers")
            fig.tight_layout()
            fig.savefig(output / f"confusion_matrix_{family.lower()}.png", dpi=160)
            plt.close(fig)
            summary.append({"model": family, "test_balanced_accuracy": report["balanced_accuracy"],
                            "test_macro_f1": report["macro_f1"], "selected_before_test": family == winner})
        pd.DataFrame(summary).to_csv(output / "model_comparison.csv", index=False)
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return output / "svd_voice_model.joblib"


def predict_voice(audio_file, model_path=DEFAULT_MODEL, *, bundle=None):
    """Return diagnosis and uncalibrated percentage scores for an uploaded WAV.

    In Streamlit load joblib once with st.cache_resource, then pass bundle=...
    Only load trusted joblib files: deserialization can execute Python code.
    """
    if bundle is None:
        bundle = joblib.load(model_path)
    if bundle["feature_spec"] != SPEC or bundle["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature specification mismatch: use the training script version supplied with this model.")
    x, metadata = extract_features(audio_file, bundle["feature_spec"])
    p = probabilities(bundle, x[None])[0]
    labels = bundle["label_encoder"].classes_
    order = np.argsort(-p)
    diagnosis = str(labels[order[0]])
    return {"diagnosis": diagnosis, "binary_result": "Healthy" if diagnosis == "Healthy" else "Pathological",
            "model": bundle["family"], "confidence_percent": float(100*p[order[0]]),
            "probabilities_percent": {str(labels[i]): float(100*p[i]) for i in order},
            "uncertain": bool(p[order[0]] < .6), "probabilities_calibrated": False,
            "notice": "Uncalibrated model scores; 60% uncertainty cutoff is a heuristic, not a validated reliability threshold.",
            "audio": metadata}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--audio-dir", type=Path, default=DEFAULT_DATA / "audio_wav")
    training.add_argument("--labels", type=Path, default=DEFAULT_DATA / "recordings.csv")
    training.add_argument("--output", type=Path, default=DEFAULT_MODEL.parent)
    training.add_argument("--seed", type=int, default=20261022)
    prediction = sub.add_parser("predict")
    prediction.add_argument("audio", type=Path)
    prediction.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.command == "train":
        print(train(args.audio_dir, args.labels, args.output, args.seed))
    else:
        print(json.dumps(predict_voice(args.audio, args.model), indent=2))
