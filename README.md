# HGU AI-Based University Phishing Detection System

Local synthetic-data demonstration for classifying university emails as **Legitimate** or **Phishing**. It combines TF-IDF text features with sender, URL, HTML, attachment, authentication-header, and language indicators. It includes training/evaluation, explanations, reports, feedback, Streamlit, and FastAPI.

## University customization

Configured for Dr. Harisingh Gour Vishwavidyalaya, also called Dr. Hari Singh Gour University, Sagar University, or University of Saugar, in Sagar, Madhya Pradesh, India. The trusted domain set is configurable. Verify official university domains before operational use. No university systems are contacted.

## Features and structure

Detection covers sender and impersonation, links and anchor mismatches, email headers, attachments, HTML, SPF/DKIM/DMARC header indicators, and urgent/fear/reward/credential-seeking language including Hindi terms. Four scikit-learn models are compared by F1; the best model reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, false positive/negative rates, and 3-fold cross-validation F1.

```text
hgu-phishing-detection-enhanced/
├── requirements.txt  README.md  enhanced_phishing_detector.py  streamlit_app.py  api.py
├── tests/ (__init__.py, test_detector.py)
├── data/.gitkeep  models/.gitkeep  logs/.gitkeep
```

## Installation and commands

Python 3.10+: `python -m venv .venv`, activate it (`.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on macOS/Linux), then `pip install -r requirements.txt`.

- Train: `python enhanced_phishing_detector.py` (writes `models/phishing_artifact.joblib`).
- Streamlit: `streamlit run streamlit_app.py`.
- API: `uvicorn api:app --reload`.
- Tests: `pytest -q`.

Demo credentials: `admin / admin123`, `user / user123`. They are demonstration-only; production requires secure authentication and secret management.

## Metrics and limitations

The generator creates paraphrases, varied greetings/signatures, harder legitimate security notices, and polite spoofed emails. Where generated template groups are available, whole groups are held out for testing. A separate validation portion of the training data selects the threshold; 0.5 is used as the documented tie/default fallback. Three-fold stratified cross-validation is run on the training partition. Each run logs its actual per-model accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, error rates, threshold, and selected model's cross-validation F1. The values vary by run/configuration and are recorded in the console and saved artifact rather than presented here as fixed benchmark scores.

## Known limitations

Metrics are computed on synthetic data and do not establish real-world detection performance. Generated message families remain a simplified proxy for real mail; template-group separation reduces direct text leakage but cannot model the full variety of university communications or adversarial adaptation. Heuristic rules can be wrong. Authentication is parsed only from headers, without DNS checks. The threat-intelligence helper is a local heuristic placeholder; possible future optional providers include VirusTotal, Google Safe Browsing, URLhaus, OpenPhish, and AbuseIPDB. SHAP is optional.

## Security and privacy

Synthetic/demo data only. Do not submit real student/staff mail, passwords, credentials, or private data. Prediction and feedback records are saved locally under `logs/`; set appropriate retention and access controls. No external API or mail server is contacted. Human review is required before action.
