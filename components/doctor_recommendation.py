"""Doctor recommendation logic based on voice disorder recognition results.

Provides general healthcare and specialist guidance derived strictly from
the existing model predictions, without diagnosing medical conditions.
"""
from dataclasses import dataclass


@dataclass
class DoctorRecommendation:
    """Structured healthcare guidance based on model inference."""
    specialist: str
    explanation: str
    next_step: str
    disclaimer: str


def get_doctor_recommendation(analysis_result: dict) -> DoctorRecommendation:
    """Generate doctor recommendation from existing analysis results.
    
    Parameters
    ----------
    analysis_result : dict
        The result dictionary returned by ModelService.analyze(), containing
        'models' with 'GMM' and 'DNN' prediction entries.
        
    Returns
    -------
    DoctorRecommendation
        Structured recommendation with specialist, rationale, next step,
        and clinical disclaimer.
    """
    disclaimer = (
        "Medical Disclaimer: This application is an experimental machine learning research tool "
        "and does not provide a medical diagnosis, clinical opinion, or treatment plan. "
        "Only a licensed physician can diagnose voice and laryngeal disorders."
    )

    models = analysis_result.get('models', {})
    primary_name = analysis_result.get('primary_model')

    if primary_name and primary_name in models:
        primary = models[primary_name]
        other_keys = [k for k in models if k != primary_name]
        secondary_name = other_keys[0] if other_keys else None
        secondary = models[secondary_name] if secondary_name else {}
    elif 'DNN' in models:
        primary_name = 'DNN'
        secondary_name = 'GMM' if 'GMM' in models else None
        primary = models['DNN']
        secondary = models.get('GMM', {})
    elif 'GMM' in models:
        primary_name = 'GMM'
        secondary_name = 'DNN' if 'DNN' in models else None
        primary = models['GMM']
        secondary = models.get('DNN', {})
    elif 'MLP' in models:
        primary_name = 'MLP'
        secondary_name = 'GMM' if 'GMM' in models else None
        primary = models['MLP']
        secondary = models.get('GMM', {})
    else:
        primary_name = list(models.keys())[0] if models else 'Model'
        secondary_name = None
        primary = models.get(primary_name, {})
        secondary = {}

    primary_result = primary.get('result', '')
    secondary_result = secondary.get('result', '')
    primary_subtype = primary.get('subtype', '') or secondary.get('subtype', '')

    # 1. Models Disagree
    if secondary_result and primary_result != secondary_result:
        return DoctorRecommendation(
            specialist="General Practitioner (GP) or ENT Specialist (Otolaryngologist)",
            explanation=(
                f"The primary model ({primary_name}) and comparison model ({secondary_name}) produced conflicting classifications "
                f"({primary_name}: {primary_result}; {secondary_name}: {secondary_result}). Because the algorithmic predictions disagree, "
                "no definitive acoustic conclusion can be drawn. A clinical examination is necessary "
                "to reliably evaluate your vocal health."
            ),
            next_step=(
                "If you are experiencing hoarseness, vocal fatigue, pain, or changes in voice quality, "
                "schedule an in-person assessment with a primary care doctor or ENT specialist rather "
                "than relying on computational predictions."
            ),
            disclaimer=disclaimer
        )

    # 2. Both Agree on Pathological
    if primary_result == 'Pathological':
        is_laryngeal = (
            'LARYNGEAL' in primary_subtype.upper() or
            primary.get('three_class_result') == 'Laryngeal' or
            secondary.get('three_class_result') == 'Laryngeal'
        )
        if is_laryngeal:
            return DoctorRecommendation(
                specialist="ENT Specialist (Otolaryngologist) or Laryngologist",
                explanation=(
                    f"Your recording was classified by the primary model ({primary_name}) as showing a pathological voice pattern, "
                    "with features associated with laryngeal or vocal-fold involvement. An ear, nose, and throat (ENT) "
                    "specialist or subspecialized laryngologist has expertise in visual and functional "
                    "examination of the larynx and vocal cords."
                ),
                next_step=(
                    "Consult an ENT specialist or laryngologist for an in-person examination, such as "
                    "laryngoscopy or videostroboscopy, to determine whether an underlying vocal condition exists."
                ),
                disclaimer=disclaimer
            )
        else:
            return DoctorRecommendation(
                specialist="ENT Specialist (Otolaryngologist)",
                explanation=(
                    f"Your recording was classified by the primary model ({primary_name}) as showing a pathological voice pattern. "
                    "An otolaryngologist (ENT specialist) is the medical specialist trained to evaluate "
                    "the vocal cords, larynx, and underlying acoustic abnormalities."
                ),
                next_step=(
                    "Consult an ENT specialist for an initial clinical assessment to evaluate whether further "
                    "diagnostic testing or voice therapy is appropriate."
                ),
                disclaimer=disclaimer
            )

    # 3. Both Agree on Healthy
    return DoctorRecommendation(
        specialist="General Healthcare Professional (or ENT Specialist if symptoms persist)",
        explanation=(
            f"The primary model ({primary_name}) did not detect a pathological acoustic pattern in this sample. "
            "However, computer models evaluate only brief sustained phonation and cannot rule out "
            "intermittent voice changes, vocal fatigue, discomfort, or non-acoustic vocal pathology."
        ),
        next_step=(
            "If you are experiencing persistent hoarseness, voice strain, difficulty speaking, or pain "
            "lasting longer than two weeks, seek an evaluation from a doctor regardless of this screening result."
        ),
        disclaimer=disclaimer
    )
