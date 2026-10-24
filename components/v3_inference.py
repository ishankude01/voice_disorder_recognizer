"""V3 Multi-Source Model Service.

Loads the trained MLP_medium_8000 and GMM_4_8000 models from artifacts_v3
(trained on 1,420 participant recordings from SVD + VOICED).
Provides inference matching the feature extraction and preprocessing
contract established in svd_v3.py.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
import joblib
import numpy as np
import soundfile as sf

from svd_v3 import LABELS, extract, probability, spec_for
from svd_training import load_audio


class V3ModelService:
    """Service wrapping the v3 multi-source models for runtime inference."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / 'artifacts_v3'
        self.bundle_path = self.artifacts_dir / 'binary.joblib'
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"V3 model bundle not found at {self.bundle_path}")

        self.bundle = joblib.load(self.bundle_path)
        self.models = self.bundle['models']
        self.selection = self.bundle.get('selection', {})
        self.winner_id = self.selection.get('winner', {}).get('id', 'MLP_medium_8000')
        self.best_gmm_id = self.selection.get('best_gmm', 'GMM_4_8000')
        self.primary_model = self.models.get(self.winner_id)
        self.secondary_model = self.models.get(self.best_gmm_id)

        if self.primary_model is None:
            raise KeyError(f"Primary model {self.winner_id} not found in bundle")

    def analyze(self, content: bytes, name: str = 'voice.wav') -> dict:
        """Run deterministic feature extraction and inference on audio bytes.
        
        Parameters
        ----------
        content : bytes
            Raw audio file bytes (WAV or supported soundfile format).
        name : str
            Source filename for metadata tracking.
            
        Returns
        -------
        dict
            Inference results compatible with Streamlit app and DoctorRecommendation.
        """
        start = time.perf_counter()
        spec8k = spec_for(8000)

        # 1. Load central 3-second audio segment at 8 kHz for playback and waveform
        audio_segment, meta = load_audio(content, spec8k)

        # 2. Extract deterministic 97-dimensional acoustic features
        rich, fixed_mel, fingerprint, extract_meta = extract(content, 8000)

        # 3. Model inference: Primary MLP (winning model)
        p_mlp = probability(self.primary_model, rich[None, :], fixed_mel[None, :, :])[0]
        p_mlp = np.asarray(p_mlp, dtype=float)
        p_mlp = p_mlp / p_mlp.sum()
        mlp_idx = int(np.argmax(p_mlp))
        mlp_label = LABELS[mlp_idx]
        mlp_score = float(p_mlp[mlp_idx])

        # 4. Model inference: Secondary GMM (comparison model)
        if self.secondary_model is not None:
            p_gmm = probability(self.secondary_model, rich[None, :], fixed_mel[None, :, :])[0]
            p_gmm = np.asarray(p_gmm, dtype=float)
            p_gmm = p_gmm / p_gmm.sum()
            gmm_idx = int(np.argmax(p_gmm))
            gmm_label = LABELS[gmm_idx]
            gmm_score = float(p_gmm[gmm_idx])
        else:
            gmm_label = mlp_label
            gmm_score = mlp_score

        # Extract quality features (CPPS, Jitter, Shimmer) from rich vector
        cpps = float(rich[94]) if len(rich) > 94 and np.isfinite(rich[94]) else None
        jitter = float(rich[95]) if len(rich) > 95 and np.isfinite(rich[95]) else None
        shimmer = float(rich[96]) if len(rich) > 96 and np.isfinite(rich[96]) else None

        elapsed = time.perf_counter() - start

        return {
            'models': {
                'MLP': {
                    'result': mlp_label,
                    'raw_score': float(np.log(np.clip(mlp_score, 1e-9, 1.0))),
                    'model_score': mlp_score,
                    'subtype': 'Multi-Source MLP (1,420 participants)',
                    'three_class_result': mlp_label,
                    'model_name': self.winner_id,
                    'calibrated': True,
                    'pathological_probability': float(p_mlp[1]),
                },
                'GMM': {
                    'result': gmm_label,
                    'raw_score': float(np.log(np.clip(gmm_score, 1e-9, 1.0))),
                    'model_score': gmm_score,
                    'subtype': 'Comparison GMM (1,420 participants)',
                    'three_class_result': gmm_label,
                    'model_name': self.best_gmm_id,
                    'calibrated': True,
                    'pathological_probability': float(p_gmm[1]) if self.secondary_model is not None else float(p_mlp[1]),
                }
            },
            'primary_model': 'MLP',
            'is_v3': True,
            'audio': audio_segment,
            'quality': {
                'missing_quality_features': bool(np.isnan(rich).any()),
                'cpps_db': cpps,
                'jitter_local': jitter,
                'shimmer_local': shimmer,
            },
            'input': {
                'trimmed': meta.get('trimmed', True),
                'original_seconds': meta.get('original_duration', len(audio_segment) / 8000),
                'filename': name,
                'fingerprint': fingerprint,
            },
            'context': (
                'SVD v3 Multi-source model evaluated across 1,420 participant recordings '
                '(1,212 Saarbruecken Voice Database + 208 VOICED). '
                'Measured balanced accuracy: 68.6% (68.7% overall accuracy, 60.7% sensitivity, 76.4% specificity).'
            ),
            'elapsed_seconds': elapsed,
        }
