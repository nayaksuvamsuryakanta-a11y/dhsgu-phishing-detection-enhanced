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

## Evaluation protocol

The training workflow uses training, validation, and test partitions. When generated `template_group` metadata is available and each class has enough distinct groups, complete message families are assigned to one partition so paraphrases do not cross partitions. Custom data without enough groups uses stratified row splits. Candidate models and the phishing threshold are selected using validation F1. The selected model is refitted on combined training and validation data; the held-out test partition is used only once for its final accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion counts, and false-positive/negative rates. Three-fold cross-validation F1 is also recorded on development data. The built-in dataset is synthetic, so these metrics do not establish real-world performance.

Hindi keyword support uses UTF-8 source encoding and Unicode-safe Devanagari terms.

If predictions seem inconsistent after detector code changes, delete `models/phishing_artifact.joblib` and run `python enhanced_phishing_detector.py` to retrain and save a fresh artifact. Review the printed metrics and demo predictions after retraining.

## Known limitations

- The built-in training dataset is synthetic; no real university dataset is included.
- SPF/DKIM/DMARC are parsed from message headers, with no live DNS or cryptographic validation.
- No real university mail server integration is implemented.
- No real threat-intelligence API is enabled; its score is a local heuristic placeholder.
- Demo authentication is hard-coded and is not production-safe.
- False positives and false negatives remain possible; synthetic metrics do not prove real-world effectiveness.
- Human review is required before blocking messages or taking production action.
- Prediction and feedback records are stored locally under `logs/`; apply suitable access and retention controls.

## Security and privacy

Synthetic/demo data only. Do not submit real student/staff mail, passwords, credentials, or private data. Prediction and feedback records are saved locally under `logs/`; set appropriate retention and access controls. No external API or mail server is contacted. Human review is required before action.
