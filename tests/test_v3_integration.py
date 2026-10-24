"""Integration tests for the SVD v3 Multi-Source model and pipeline."""
from pathlib import Path
import pytest
from components.v3_inference import V3ModelService
from components.doctor_recommendation import get_doctor_recommendation
from components.validator import validate_audio

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def v3_service():
    return V3ModelService(ROOT)


def test_v3_service_initialization(v3_service):
    """Ensure v3 model bundle loads properly with required models."""
    assert v3_service.bundle is not None
    assert "MLP_medium_8000" in v3_service.models
    assert v3_service.winner_id == "MLP_medium_8000"
    assert v3_service.primary_model is not None


def test_v3_healthy_sample_inference(v3_service):
    """Test v3 inference and doctor recommendation on healthy sample."""
    sample_path = ROOT / "samples" / "healthy_voice002.wav"
    audio_bytes = sample_path.read_bytes()

    # Pre-validation check
    val = validate_audio(audio_bytes, "healthy_voice002.wav")
    assert val.is_valid, f"Healthy sample failed validator: {val.rejection_reason}"

    # Model inference
    result = v3_service.analyze(audio_bytes, "healthy_voice002.wav")
    assert result["is_v3"] is True
    assert "models" in result
    assert "MLP" in result["models"]
    assert "GMM" in result["models"]

    mlp = result["models"]["MLP"]
    assert mlp["result"] in ["Healthy", "Pathological"]
    assert 0.0 <= mlp["model_score"] <= 1.0
    assert mlp["calibrated"] is True

    # Doctor recommendation
    rec = get_doctor_recommendation(result)
    assert rec.specialist is not None
    assert rec.explanation is not None
    assert rec.disclaimer is not None


def test_v3_pathological_sample_inference(v3_service):
    """Test v3 inference on pathological sample."""
    sample_path = ROOT / "samples" / "pathological_voice003.wav"
    audio_bytes = sample_path.read_bytes()

    # Pre-validation check
    val = validate_audio(audio_bytes, "pathological_voice003.wav")
    assert val.is_valid, f"Pathological sample failed validator: {val.rejection_reason}"

    result = v3_service.analyze(audio_bytes, "pathological_voice003.wav")
    assert result["is_v3"] is True
    assert len(result["audio"]) == 24000  # 3 seconds at 8 kHz

    rec = get_doctor_recommendation(result)
    assert rec.specialist is not None


def test_v3_contract_matches_app_expectations(v3_service):
    """Verify dictionary keys required by Streamlit UI are all present."""
    sample_path = ROOT / "samples" / "healthy_voice002.wav"
    result = v3_service.analyze(sample_path.read_bytes(), "test.wav")

    required_keys = ["models", "audio", "quality", "input", "context", "elapsed_seconds"]
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"

    # Verify quality metrics
    quality = result["quality"]
    assert "missing_quality_features" in quality
    assert "cpps_db" in quality
    assert "jitter_local" in quality
    assert "shimmer_local" in quality
