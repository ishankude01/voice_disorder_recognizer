"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import base64
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from voice_review.web_inference import ModelService
from components.validator import validate_audio
from components.doctor_recommendation import get_doctor_recommendation
from components.v3_inference import V3ModelService

ROOT = Path(__file__).resolve().parent
_raw_audio_recorder = components.declare_component(
    'raw_audio_recorder',
    path=str(ROOT / 'components' / 'raw_audio_recorder')
)


class RecordedAudio:
    """Minimal wrapper around raw WAV bytes to match UploadedFile interface."""
    def __init__(self, data: bytes, name: str = 'recorded_voice.wav'):
        self._data = data
        self.name = name

    def getvalue(self) -> bytes:
        return self._data
st.set_page_config(page_title='Voice Disorder Recognizer',page_icon=':material/graphic_eq:',layout='centered')
st.title('Voice Disorder Recognizer')
st.write('Upload a sustained “aa” vowel to evaluate voice recordings with DNN and comparison models.')
st.caption('Academic project, not a medical diagnosis.')


@st.cache_resource(max_entries=1,show_spinner='Loading trained models…')
def load_baseline_models():
    return ModelService(ROOT)


@st.cache_resource(max_entries=1,show_spinner='Loading v3 multi-source model…')
def load_v3_models():
    return V3ModelService(ROOT)


uploaded = st.file_uploader('Upload your voice recording',type=['wav','mp3'],
                            max_upload_size=20,help='At least 3 seconds. The central 3 seconds are analyzed.',key='voice_file')
with st.expander('Or record audio'):
    rec_val = _raw_audio_recorder(key='voice_mic')
    recorded = None
    if isinstance(rec_val, dict) and rec_val.get('wav_base64'):
        try:
            wav_bytes = base64.b64decode(rec_val['wav_base64'])
            if wav_bytes.startswith(b'RIFF') and wav_bytes[8:12] == b'WAVE':
                recorded = RecordedAudio(wav_bytes, 'recorded_voice.wav')
        except Exception:
            logging.exception('Failed to decode recorded audio')

source = uploaded if uploaded is not None else recorded
both = uploaded is not None and recorded is not None
content = source.getvalue() if source is not None else None
name = source.name if source is not None else ''
fingerprint = hashlib.sha256(content).hexdigest() if content else None
st.sidebar.header('Model Selection')
model_choice = st.sidebar.selectbox(
    'Inference Engine',
    options=[
        'Original DNN Model (Default)',
        'Original GMM Model',
        'SVD v3 MLP Model (1,420 samples)',
    ],
    index=0,
    help='Select which trained model evaluates the voice recording.'
)
use_v3 = 'v3' in model_choice

if st.session_state.get('input_hash') != fingerprint or both or st.session_state.get('current_model_choice') != model_choice:
    st.session_state.pop('analysis',None)
    st.session_state['input_hash'] = fingerprint
    st.session_state['current_model_choice'] = model_choice
if both:
    st.info('Remove the uploaded file or clear the microphone recording, then analyze one recording.')

try:
    if use_v3:
        service = load_v3_models()
    else:
        service = load_baseline_models()
except Exception:
    logging.exception('Model initialization failed')
    st.error('Models could not be loaded. Launch the app from the complete project folder with its supplied environment.')
    st.stop()

if st.button('Analyze Voice',type='primary',width='stretch',disabled=source is None or both):
    st.session_state.pop('analysis',None)
    st.session_state.pop('validation_warnings',None)
    
    validation = validate_audio(content, name)
    if not validation.is_valid:
        st.error(validation.rejection_reason)
        st.stop()
    else:
        if validation.warnings:
            st.session_state['validation_warnings'] = validation.warnings
        try:
            with st.spinner('Analyzing your voice…'):
                analysis_res = service.analyze(content, name)
                if not use_v3:
                    analysis_res['primary_model'] = 'DNN' if 'DNN' in model_choice else 'GMM'
                st.session_state['analysis'] = analysis_res
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except Exception:
            logging.exception('Voice analysis failed')
            st.error('This recording could not be analyzed. Try a clear WAV recording of a steady vowel.')
            st.stop()

result = st.session_state.get('analysis')
if result is not None:
    if st.session_state.get('validation_warnings'):
        for warning_msg in st.session_state['validation_warnings']:
            st.warning(f"Audio Quality Notice: {warning_msg}")
    is_v3 = result.get('is_v3', False)
    primary_key = result.get('primary_model', 'DNN')
    if primary_key not in result['models']:
        primary_key = 'DNN' if 'DNN' in result['models'] else list(result['models'].keys())[0]

    main = result['models'][primary_key]
    st.subheader(f'Result: {main["result"]}')

    if primary_key == 'DNN':
        st.write('**Primary model: DNN (Deep Neural Network)** · trained on baseline VOICED clinical recordings.')
        st.write(f'**Disorder group:** {main.get("subtype", "Not assigned")}')
        st.progress(main['model_score'], text=f'Model score: {main["model_score"]:.1%} — uncalibrated')
        st.caption('Validated confidence is unavailable. This score is not a medical probability and can be high for an incorrect prediction.')
    elif primary_key == 'GMM':
        st.write('**Primary model: GMM (Gaussian Mixture Model)** · baseline clinical model.')
        st.write(f'**Disorder group:** {main.get("subtype", "Not assigned")}')
        st.caption('Subtype suggestions are experimental and have frequent false positives. “Laryngeal” covers laryngitis-labelled recordings only; other pathology may also involve the larynx.')
        st.progress(main['model_score'], text=f'Model score: {main["model_score"]:.1%} — uncalibrated')
        st.caption('Validated confidence is unavailable. This score is not a medical probability and can be high for an incorrect prediction.')
    elif primary_key == 'MLP':
        st.write('**Primary model: MLP (Multi-Layer Perceptron)** · trained on 1,420 multi-source participants (SVD + VOICED).')
        st.write(f'**Architecture:** {main.get("model_name", "MLP_medium_8000")}')
        st.progress(main['model_score'], text=f'Model probability: {main["model_score"]:.1%} (calibrated)')
        st.caption('Calibrated probability using temperature scaling on cross-validated multi-source clinical data.')

    columns = st.columns(2)
    other_keys = [k for k in result['models'].keys() if k != primary_key]
    ordered_keys = [primary_key] + other_keys

    for column, family in zip(columns, ordered_keys):
        prediction = result['models'][family]
        with column:
            role = 'Primary' if family == primary_key else 'Comparison'
            st.metric(f'{family} ({role})', prediction['result'])
            cal_label = 'calibrated' if prediction.get('calibrated') else 'uncalibrated'
            st.caption(f'Score: {prediction["model_score"]:.1%} · {cal_label}')
            st.write(prediction.get('subtype', ''))

    if len(ordered_keys) >= 2:
        m1, m2 = ordered_keys[0], ordered_keys[1]
        if result['models'][m1]['result'] != result['models'][m2]['result']:
            st.warning(f'The two models ({m1} and {m2}) disagree. Neither result establishes a diagnosis.')
    st.caption(result['context'])
    if result['input']['trimmed']:
        st.caption(f'Original duration: {result["input"]["original_seconds"]:.1f} s. Analyzed the central 3 seconds at 8 kHz.')
    if result['quality']['missing_quality_features']:
        st.warning('Some acoustic measurements were undefined. The saved training-time imputation was applied.')
    if is_v3:
        q = result.get('quality', {})
        metrics_str = []
        if q.get('cpps_db') is not None:
            metrics_str.append(f"CPPS: {q['cpps_db']:.2f} dB")
        if q.get('jitter_local') is not None:
            metrics_str.append(f"Jitter: {q['jitter_local'] * 100:.2f}%")
        if q.get('shimmer_local') is not None:
            metrics_str.append(f"Shimmer: {q['shimmer_local'] * 100:.2f}%")
        if metrics_str:
            st.info("Acoustic Quality Indicators (v3): " + " · ".join(metrics_str))
    st.caption('Playback: the three-second segment analyzed from your upload (converted for browser playback only).')
    st.audio(result['audio'],sample_rate=8000,alt='Analyzed segment of uploaded voice recording')
    waveform = pd.DataFrame({'Time (s)':np.arange(len(result['audio']))[::8]/8000,
                             'Amplitude':result['audio'][::8]})
    st.line_chart(waveform,x='Time (s)',y='Amplitude',height=180,color='#0f766e',
                  alt='Waveform of the central three seconds used for prediction')
    st.caption(f'Analysis completed in {result["elapsed_seconds"]:.2f} seconds. Audio stays in this local session and is not written to disk.')

    st.subheader('Doctor Recommendation')
    rec = get_doctor_recommendation(result)
    st.markdown(f"**Recommended Healthcare Professional:** {rec.specialist}")
    st.write(f"**Why this specialist:** {rec.explanation}")
    st.info(f"**Suggested Next Step:** {rec.next_step}")
    st.caption(f"ℹ️ {rec.disclaimer}")

with st.expander('About the Models'):
    st.markdown(
        "- **DNN (Deep Neural Network · Default):** Trained on baseline clinical recordings (VOICED dataset) "
        "using 13 MFCC acoustic features. Evaluated via 3-fold cross-validation on the development set "
        "(**Balanced Accuracy: 57.5%**; uncalibrated score).\n"
        "- **GMM (Gaussian Mixture Model):** Density-estimation model trained on baseline clinical recordings "
        "(VOICED dataset). Evaluated via 3-fold cross-validation on the development set "
        "(**Balanced Accuracy: 61.6%**; uncalibrated score).\n"
        "- **MLP (Multi-Layer Perceptron · SVD v3):** Feedforward neural network trained on 97 acoustic features "
        "(MFCCs, CPPS, HNR, jitter, shimmer) across 1,420 participants (Saarbruecken Voice Database + VOICED). "
        "Evaluated via 5-fold participant-disjoint nested cross-validation "
        "(**Balanced Accuracy: 68.6%**, **Overall Accuracy: 68.7%**; calibrated probability)."
    )
