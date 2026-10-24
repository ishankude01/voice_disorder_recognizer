# Voice Disorder Recognizer Using DNN and GMM

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41.0-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/PyTest-68%20Passed%2C%202%20Skipped-brightgreen.svg)]()
[![Academic Project](https://img.shields.io/badge/Degree-B.Tech%20ECE%20Project-blueviolet.svg)]()

An end-to-end web screening platform comparing **Deep Neural Networks (DNN)** and **Gaussian Mixture Models (GMM)** alongside an optional multi-source **Multi-Layer Perceptron (MLP)** for objective, non-invasive voice disorder (dysphonia) detection from sustained `/a/` phonation.

The application integrates an uncompressed **5-second in-browser Web Audio API microphone recorder**, a **frontline multi-tier sustained vowel acoustic validator**, real-time digital signal processing (DSP), central 3-second waveform visualization, interactive audio playback, and a **clinical decision support (Doctor Recommendation)** subsystem.

> **MEDICAL DISCLAIMER:**  
> This software is an experimental academic research and pre-clinical screening tool. It does **not** provide medical diagnosis, clinical opinion, or treatment planning. Statistical scores are uncalibrated and can be high for an incorrect prediction. Only a licensed physician (such as an Otolaryngologist / ENT Specialist) can diagnose voice and laryngeal pathologies through physical examination (e.g., videolaryngostroboscopy).

---

## 🏛️ Project & Institutional Details

- **Academic Program:** Bachelor of Technology (B.Tech) in Electronics and Communication Engineering (ECE)
- **Institution:** Bharati Vidyapeeth Deemed to be University, College of Engineering, Pune – 411 043
- **Department:** Department of Electronics and Communication Engineering
- **Semester & Academic Year:** Semester VII, Academic Year 2026–2027
- **Project Timeline:** 25 August 2026 – 23 October 2026
- **Student Authors:**
  - **Ishan Kude** (Roll No. 05, PRN: 2314110440) — *Full-Stack Web Architect & UI Integration Lead*
  - **Himanshu Mishra** (Roll No. 74, PRN: 2314110467) — *Lead Acoustic DSP & Validator Algorithm Engineer*
  - **Aditya Pawar** (Roll No. 76, PRN: 2314110476) — *Machine Learning & Statistical Evaluation Lead*
- **Project Mentorship:**
  - **Project Guide:** Prof. P. A. Chougule
  - **Project Co-ordinator:** Dr. Dhiraj M. Dhane
  - **Head of Department:** Dr. Mangal V. Patil

---

## 🌟 Key Engineering Features

1. **Reactive Web Application (Streamlit):**
   - Zero-install, responsive web presentation layer with dynamic runtime **Model Selection** (`Original DNN Model (Default)`, `Original GMM Model`, `SVD v3 MLP Model (1,420 samples)`).
   - Seamlessly switches inference engines without page reload.

2. **Custom 5-Second In-Browser Audio Recorder:**
   - Engineered with HTML5 MediaDevices and browser **Web Audio API** (`components/raw_audio_recorder/`).
   - Captures raw, uncompressed 16-bit PCM voice directly from client microphones at hardware sampling rates.
   - Also supports drag-and-drop audio file upload for WAV and MP3 formats up to 20 MB.

3. **Frontline Multi-Stage Sustained `/a/` Acoustic Validator:**
   - Implemented in `components/validator.py` to prevent corrupt or non-sustained audio from reaching machine learning inference.
   - Enforces a 5-tier gatekeeping check:
     - **Duration Check:** $T \ge 3.0$ seconds (rejects short clicks).
     - **RMS Energy Floor:** Peak frame $\text{RMS} \ge 0.015$ (rejects silence and background room noise).
     - **Continuous Voicing Ratio:** $V_r \ge 0.65$ via Praat Cross-Correlation (rejects unvoiced whisper).
     - **Pitch Stability (IQR):** $F_0 \text{ IQR} \le 35.0$ Hz (rejects conversational inflection and singing).
     - **Formant Boundary Filtering:** LPC roots $F_1 \in [460, 700]$ Hz and $F_2 \in [950, 1350]$ Hz (rejects non-target vowels like `/i/`, `/u/`, `/e/`).

4. **Multi-Model Pattern Recognition Stack:**
   - **Primary DNN (Default):** 3-layer PyTorch neural network (26 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 2) trained with ReLU, BatchNorm1d, and Dropout ($p = 0.20$).
   - **Comparison GMM:** Dual 4-component Gaussian Mixture Models with diagonal covariance matrices, classified via log-likelihood ratio test.
   - **Optional SVD v3 MLP:** Scikit-Learn Multi-Layer Perceptron (128, 64) trained across 1,420 multi-source development samples with 97 composite acoustic descriptors.

5. **Clinical Decision Support (Doctor Recommendation):**
   - Implemented in `components/doctor_recommendation.py`.
   - Automatically maps prediction results and auxiliary laryngeal indications to specialist referrals (**ENT Specialist / Otolaryngologist / Laryngologist**), clinical rationales, and recommended next diagnostic steps (videostroboscopy).
   - Prominently displays safety warnings and non-diagnostic disclaimers.

---

## 📊 Measured Performance & Scientific Integrity

All models are evaluated under strict, **participant-disjoint** partitioning (no speaker overlap between training and testing folds) to guarantee zero data leakage.

### Baseline Model Development Performance (VOICED Cohort, $n=166$)

| Performance Metric | Gaussian Mixture Model (GMM) | Deep Neural Network (DNN - Primary) | Clinical & Operational Trade-Off |
| :--- | :---: | :---: | :--- |
| **Balanced Accuracy** | **61.64%** | **57.48%** | Primary unbiased metric (accounts for 2.69:1 clinical class imbalance). |
| **Overall Accuracy** | 51.20% | 61.45% | Overall fraction of correct predictions; biased toward pathological majority. |
| **Sensitivity (Recall_Path)** | 38.84% | **66.94%** | **DNN detects 28.1% more pathological cases** (paramount for screening). |
| **Specificity (Recall_Health)**| **84.44%** | 48.01% | **GMM has higher specificity** (minimizes false alarms among healthy users). |
| **Macro-F1 Score** | 51.06% | 56.63% | Harmonic mean of precision and recall averaged equally across classes. |
| **ROC-AUC** | **0.6204** | 0.6062 | Area under ROC curve; solid discriminatory power above chance. |

### Multi-Source SVD v3 MLP Performance ($n=1,420$)

| Evaluation Setting | Balanced Accuracy | Sensitivity | Specificity | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Pooled Development Cohort** | **68.58%** | 69.04% | 68.12% | 68.66% |
| **Held-Out Cross-Dataset Test** | **62.70%** | 64.12% | 61.28% | 62.50% |
| **Acoustic Domain Shift Delta** | **-5.88%** | -4.92% | -6.84% | -6.16% |

> **Scientific Honesty Regarding Accuracy:**  
> In prior academic literature, accuracies exceeding 90% are frequently reported due to record-wise random splits that leak identical patient voices between train and test sets. When evaluated with strict patient separation and across heterogeneous recording hardware, genuine generalization accuracy sits between **57% and 68%**. This project adheres to honest reporting and does not claim unverified 80%+ accuracy.

---

## 📁 Repository Structure

```text
voice_disorder_recognizer/
│
├── app.py                          # Streamlit application entrypoint & reactive UI
├── requirements.txt                # Pinned core dependencies
├── requirements-lock.txt           # Exact frozen environment lock
├── requirements-web-lock.txt       # Production web runtime lock
├── .gitignore                      # Git exclusion rules (ignores venv, secrets, caches)
│
├── components/                     # Production application components
│   ├── validator.py                # Sustained /a/ acoustic validator engine
│   ├── doctor_recommendation.py    # Clinical decision support & ENT referral routing
│   ├── v3_inference.py             # Inference service for SVD v3 97-D MLP model
│   └── raw_audio_recorder/         # Custom 5-second HTML5 Web Audio API recorder
│       └── index.html              # Embedded browser audio capture component
│
├── voice_review/                   # Core acoustic processing & baseline models
│   ├── features.py                 # Audio decoding, framing, 13-MFCC & perturbation DSP
│   ├── models.py                   # PyTorch DNN architecture and GMM definitions
│   ├── web_inference.py            # Baseline inference engine & cached model loaders
│   └── demo.py                     # Demo utilities and audio playback preparation
│
├── models/                         # Serialized baseline PyTorch and Scikit-learn models
│   ├── baseline_voiced_dnn.pt      # Pretrained PyTorch DNN weights
│   ├── baseline_voiced_gmm.joblib  # Fitted GMM components
│   ├── scaler.joblib               # StandardScaler for 26-D baseline features
│   └── fold_models/                # 3-fold cross-validation model checkpoints
│
├── artifacts_v3/                   # SVD v3 Multi-Source MLP artifacts (1,420 samples)
│   ├── binary.joblib               # Fitted SVD v3 MLP model weights
│   ├── feature_manifest.json       # Specification of 97 composite acoustic descriptors
│   ├── protocol.json               # Cross-validation and training protocol
│   └── confusion_pooled.png        # Pooled confusion matrix figure
│
├── data/                           # Clinical dataset metadata and frozen split manifests
│   ├── manifest.csv                # VOICED recording attributes and diagnoses
│   ├── splits.json                 # 3-fold patient-disjoint train/validation split indices
│   ├── feature_spec.json           # Acoustic descriptor specifications
│   └── audit.json                  # Data integrity verification checksums
│
├── samples/                        # Attributed public clinical test samples for demonstration
│   ├── healthy_voice002.wav        # Normal sustained /a/ phonation
│   ├── pathological_voice003.wav   # Pathological phonation (vocal fold nodule)
│   ├── laryngeal_voice008.wav      # Pathological phonation (laryngitis group)
│   └── README.md                   # Attribution and usage notes
│
├── demo_audio/                     # Supplemental verification audio assets
│   └── voice003.wav                # Benchmark verification audio
│
├── results/                        # Quantitative evaluation results & analytical figures
│   ├── final_evaluation.lock       # Sealed test evaluation integrity lock
│   ├── figures/                    # High-resolution ROC curves, confusion matrices
│   └── predictions/                # Out-of-fold validation prediction records
│
├── tests/                          # Automated PyTest quality assurance suite (70 tests)
│   ├── test_validator.py           # 24 tests verifying acoustic gatekeeping
│   ├── test_doctor_recommendation.py # 10 tests verifying clinical referral logic
│   ├── test_v3_integration.py      # 12 tests verifying SVD v3 97-D inference
│   ├── test_integrity.py           # Feature extraction & reproducibility tests
│   ├── test_demo_and_robustness.py # Corrupt audio & edge-case stress tests
│   └── test_website.py             # Streamlit UI workflow & model loading tests
│
└── docs/                           # Technical documentation and reproduction guides
    ├── REPRODUCIBILITY.md          # Step-by-step experiment reproduction instructions
    └── DEPLOYMENT.md               # Cloud and server deployment guidelines
```

---

## 🚀 Quick Start & Installation

### Prerequisites

- **Python Version:** Python **3.12** is recommended. (Compatible with Python 3.10–3.12).
- **Compute Hardware:** CPU is completely sufficient for real-time inference (average analysis latency < 1.5 seconds).

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd voice_disorder_recognizer
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the verified pinned packages:

**Windows:**
```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

*(All trained model weights in `models/` and `artifacts_v3/` are included in the repository; no dataset download or model retraining is required to run the web application).*

---

## 💻 Running the Web Application

Launch the Streamlit web application using:

```bash
python -m streamlit run app.py
```

Once running, open your web browser and navigate to:
```text
http://localhost:8501
```

### How to Use the Application:
1. **Select Inference Model:** In the left sidebar dropdown, choose among:
   - `Original DNN Model (Default)`
   - `Original GMM Model`
   - `SVD v3 MLP Model (1,420 samples)`
2. **Provide Voice Audio:**
   - **Option A:** Record a steady sustained **"aaaa"** vowel for 5 seconds using the built-in microphone recorder.
   - **Option B:** Drag and drop an existing sustained vowel WAV/MP3 recording (demo files available in `samples/`).
3. **Analyze:** Click **"Analyze Voice"**.
4. **Review Results:**
   - The acoustic validator verifies duration, RMS energy, voicing, pitch IQR, and formants.
   - If valid, the system displays the **Primary Prediction**, uncalibrated confidence score, secondary comparison card, central 3-second audio waveform, and playback player.
   - Scroll down to the **"Doctor Recommendation"** section for specialist referral advice (ENT Specialist / Laryngologist) and recommended next clinical steps.

---

## 🧪 Running the Automated QA Test Suite

To verify numerical reproducibility, acoustic boundary compliance, and error resistance, execute the PyTest test suite:

```bash
python -m pytest
```

Expected output:
```text
======================= 68 passed, 2 skipped in 20.46s =======================
```
*(The 2 skipped tests correspond to optional large raw audio download verification when local mirrors are already cached).*

---

## 📚 Dataset Acknowledgements & References

1. **VOICED Database:**  
   *Voice ICar fEDerico II Database (VOICED)*, University of Naples Federico II, Italy.  
   PhysioNet DOI: [10.13026/C25Q2N](https://doi.org/10.13026/C25Q2N).
2. **Saarbruecken Voice Database (SVD):**  
   *Institute of Phonetics, Saarland University*, Germany.
3. **Scoping Review Literature:**  
   Gupta et al., *"Machine Learning Approaches for Voice Pathology Detection: A Scoping Review,"* Frontiers in Digital Health, 2026. DOI: [10.3389/fdgth.2026.1800132](https://doi.org/10.3389/fdgth.2026.1800132).
4. **Praat Phonetics Software:**  
   Boersma, P., & Weenink, D. (2024). *Praat: doing phonetics by computer* [Computer program].

---

## ⚖️ License & Attribution

This project is released under the **MIT License**.  
Developed for academic, educational, and pre-clinical research purposes at **Bharati Vidyapeeth Deemed to be University, College of Engineering, Pune**.
