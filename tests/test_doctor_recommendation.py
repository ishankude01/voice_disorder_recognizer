"""Tests for the Doctor Recommendation feature."""
from pathlib import Path
import pytest
from components.doctor_recommendation import get_doctor_recommendation, DoctorRecommendation
from components.validator import validate_audio

ROOT = Path(__file__).resolve().parents[1]


def test_pathological_laryngeal_recommendation():
    """Pathological prediction with laryngeal indication recommends ENT or Laryngologist."""
    mock_result = {
        'models': {
            'GMM': {
                'result': 'Pathological',
                'subtype': 'LARYNGEAL — laryngitis-labelled group',
                'three_class_result': 'Laryngeal'
            },
            'DNN': {
                'result': 'Pathological',
                'subtype': 'LARYNGEAL — laryngitis-labelled group',
                'three_class_result': 'Laryngeal'
            }
        }
    }
    rec = get_doctor_recommendation(mock_result)
    assert isinstance(rec, DoctorRecommendation)
    assert 'ENT' in rec.specialist or 'Laryngologist' in rec.specialist
    assert 'laryngeal' in rec.explanation.lower() or 'larynx' in rec.explanation.lower()
    assert 'laryngoscopy' in rec.next_step.lower() or 'videostroboscopy' in rec.next_step.lower() or 'examination' in rec.next_step.lower()
    assert 'experimental' in rec.disclaimer.lower()


def test_pathological_general_recommendation():
    """General pathological prediction without specific laryngeal indication recommends ENT specialist."""
    mock_result = {
        'models': {
            'GMM': {
                'result': 'Pathological',
                'subtype': 'Other pathology — specific disorder unresolved',
                'three_class_result': 'Other'
            },
            'DNN': {
                'result': 'Pathological',
                'subtype': 'Other pathology — specific disorder unresolved',
                'three_class_result': 'Other'
            }
        }
    }
    rec = get_doctor_recommendation(mock_result)
    assert 'ENT' in rec.specialist or 'Otolaryngologist' in rec.specialist
    assert 'pathological' in rec.explanation.lower()
    assert 'initial clinical assessment' in rec.next_step.lower() or 'consult' in rec.next_step.lower()
    assert 'diagnosis' in rec.disclaimer.lower()


def test_healthy_recommendation():
    """Healthy prediction advises monitoring and consulting if symptoms persist."""
    mock_result = {
        'models': {
            'GMM': {
                'result': 'Healthy',
                'subtype': 'Not assigned by the binary result',
                'three_class_result': 'Healthy'
            },
            'DNN': {
                'result': 'Healthy',
                'subtype': 'Not assigned by the binary result',
                'three_class_result': 'Healthy'
            }
        }
    }
    rec = get_doctor_recommendation(mock_result)
    assert 'did not detect' in rec.explanation.lower()
    assert 'persistent' in rec.next_step.lower() or 'symptoms' in rec.next_step.lower()
    assert 'experimental' in rec.disclaimer.lower()


def test_disagreeing_models_recommendation():
    """When GMM and DNN disagree, a cautious recommendation for professional assessment is shown."""
    mock_result = {
        'models': {
            'GMM': {
                'result': 'Healthy',
                'subtype': 'Not assigned by the binary result',
                'three_class_result': 'Healthy'
            },
            'DNN': {
                'result': 'Pathological',
                'subtype': 'LARYNGEAL — laryngitis-labelled group',
                'three_class_result': 'Laryngeal'
            }
        }
    }
    rec = get_doctor_recommendation(mock_result)
    assert 'conflicting' in rec.explanation.lower() or 'disagree' in rec.explanation.lower()
    assert 'clinical examination' in rec.explanation.lower() or 'reliable' in rec.explanation.lower()
    assert 'schedule an in-person assessment' in rec.next_step.lower() or 'consult' in rec.next_step.lower()
    assert 'experimental' in rec.disclaimer.lower()


def test_recommendation_with_real_demo_samples():
    """Verify recommendation generation on real model inference outputs from samples/."""
    from voice_review.web_inference import ModelService
    service = ModelService(ROOT)

    # 1. Healthy sample
    healthy_path = ROOT / 'samples' / 'healthy_voice002.wav'
    res_healthy = service.analyze(healthy_path.read_bytes(), healthy_path.name)
    rec_healthy = get_doctor_recommendation(res_healthy)
    assert 'General Healthcare' in rec_healthy.specialist

    # 2. Pathological sample (both agree on pathological + laryngeal)
    patho_path = ROOT / 'samples' / 'pathological_voice003.wav'
    res_patho = service.analyze(patho_path.read_bytes(), patho_path.name)
    rec_patho = get_doctor_recommendation(res_patho)
    assert 'ENT Specialist' in rec_patho.specialist or 'Laryngologist' in rec_patho.specialist

    # 3. Disagreeing sample (GMM healthy, DNN pathological)
    laryngeal_path = ROOT / 'samples' / 'laryngeal_voice008.wav'
    res_disagree = service.analyze(laryngeal_path.read_bytes(), laryngeal_path.name)
    rec_disagree = get_doctor_recommendation(res_disagree)
    assert 'conflicting' in rec_disagree.explanation.lower() or 'disagree' in rec_disagree.explanation.lower()


def test_invalid_audio_does_not_produce_recommendation():
    """Invalid audio fails validation so model analysis and doctor recommendation are not triggered."""
    # Short audio
    short_audio = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    val = validate_audio(short_audio, 'invalid.wav')
    assert not val.is_valid
    # In app.py, if not val.is_valid, st.stop() halts before get_doctor_recommendation is called
