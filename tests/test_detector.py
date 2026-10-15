import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from api import app
from enhanced_phishing_detector import CANONICAL_LEGITIMATE_SAMPLE, FEATURE_NAMES, extract_single_numeric_features, generate_sample_dataset, normalize_record, predict_record, train_models

def test_dataset_nonempty_and_balanced_labels():
    df=generate_sample_dataset(8,8)
    assert isinstance(df,pd.DataFrame) and not df.empty
    assert set(df.label)=={0,1}

def test_normalization_extracts_html_body_and_links():
    record=normalize_record({"subject":"Notice","html":'<a href="https://example.org/a">Open</a>'})
    assert "Open" in record["body"] and "https://example.org/a" in record["links"]

def test_all_numeric_features_present():
    feats=extract_single_numeric_features({"sender":"office@dhgsu.ac.in","body":"notice"})
    assert list(feats)==FEATURE_NAMES
    assert all(isinstance(value,(float,int)) for value in feats.values())

def test_hindi_phishing_terms_trigger_hindi_feature():
    feats=extract_single_numeric_features({"subject":"खाता तुरंत सत्यापित करें","body":"पासवर्ड दर्ज करें या खाता निलंबित होगा; चेतावनी"})
    assert feats["hindi_urgent_words"] >= 5
    assert feats["asks_password"] == 1

def test_prediction_result_fields_for_both_examples():
    artifact=train_models(generate_sample_dataset(40,40),save=False)
    required={"id","timestamp","university","subject","sender","classification","risk_score_percent","probability_phishing","risk_level","threshold_used","reasons","recommendations","features","links","attachments","model_name"}
    legitimate=predict_record({"subject":"Exam notice","body":"Examination timetable published","sender":"exam@dhgsu.ac.in","links":["https://dhgsu.ac.in/exam"],"spf_result":"pass","dkim_result":"pass","dmarc_result":"pass"},artifact)
    phishing=predict_record({"subject":"URGENT account suspended","body":"Verify your password and OTP immediately","sender":"support@dhgsu-portal.xyz","links":["http://dhgsu-portal.xyz/login"],"attachments":["notice.pdf.exe"],"spf_result":"fail","dkim_result":"fail","dmarc_result":"fail"},artifact)
    assert required <= legitimate.keys() and required <= phishing.keys()
    assert legitimate["classification"] in {"Legitimate","Phishing"} and phishing["classification"] in {"Legitimate","Phishing"}
    assert 0 <= legitimate["risk_score_percent"] <= 100 and 0 <= phishing["risk_score_percent"] <= 100
    fallback="No strong phishing indicators were detected by the available checks."
    if phishing["classification"] == "Phishing":
        assert any(reason != fallback for reason in phishing["reasons"])
    canonical=predict_record(CANONICAL_LEGITIMATE_SAMPLE,artifact)
    assert canonical["classification"] == "Legitimate"
    minimal=predict_record({"subject":"Campus notice"},artifact,threshold=0.0)
    assert minimal["classification"] == "Phishing"
    assert any("Limited evidence" in reason and all(term in reason for term in ("sender domain", "links", "SPF/DKIM/DMARC")) for reason in minimal["reasons"])

def test_normalization_handles_null_empty_and_nan_values():
    record = normalize_record({"subject": None, "body": np.nan, "html": "", "sender": None,
                               "links": None, "attachments": np.nan})
    assert record["subject"] == record["body"] == record["sender"] == ""
    assert record["links"] == record["attachments"] == []
    features = extract_single_numeric_features(None)
    assert list(features) == FEATURE_NAMES
    assert all(np.isfinite(value) for value in features.values())

def test_malformed_nested_html_and_unusual_urls_are_safe():
    html = "<div>" * 500 + '<a href="http://127.0.0.1/路径">例え.テスト</a>' + "</span>" * 500
    record = normalize_record({"html": html, "links": ["example.com/path", "http://127.0.0.1/a", "https://例え.テスト/道"]})
    assert record["html_link_count"] == 1
    features = extract_single_numeric_features(record)
    assert features["ip_url"] == 2
    assert features["num_links"] >= 3
    assert all(np.isfinite(value) for value in features.values())

def test_predict_json_integration_valid_missing_optional_and_long_input():
    with TestClient(app) as client:
        complete = client.post("/predict-json", json={"subject":"Notice", "body":"Campus update", "sender":"office@dhgsu.ac.in", "html":"<p>Update</p>", "attachments":[], "links":[]})
        assert complete.status_code == 200
        assert complete.json()["classification"] in {"Legitimate", "Phishing"}
        minimal = client.post("/predict-json", json={"subject":"Hello", "body":"No optional fields"})
        assert minimal.status_code == 200
        assert minimal.json()["features"]["num_attachments"] == 0
        long_text = client.post("/predict-json", json={"subject":"Long", "body":"x" * 100_000})
        assert long_text.status_code == 200
        assert 0 <= long_text.json()["risk_score_percent"] <= 100

def test_predict_eml_integration_valid_empty_and_non_email_uploads():
    valid = b"From: office@dhgsu.ac.in\r\nTo: student@example.com\r\nSubject: Campus notice\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nPlease review the campus update."
    with TestClient(app) as client:
        response = client.post("/predict-eml", files={"file": ("notice.eml", valid, "message/rfc822")})
        assert response.status_code == 200
        assert response.json()["subject"] == "Campus notice"
        empty = client.post("/predict-eml", files={"file": ("empty.eml", b"", "message/rfc822")})
        assert empty.status_code == 400
        disguised = client.post("/predict-eml", files={"file": ("notes.eml", b"This is ordinary text, not an email message.", "text/plain")})
        assert disguised.status_code == 400
        assert "parsable email" in disguised.json()["detail"]

def test_predict_eml_rejects_upload_over_size_limit():
    payload=b"From: office@dhgsu.ac.in\r\nSubject: Large message\r\n\r\n" + b"x" * (10 * 1024 * 1024)
    with TestClient(app) as client:
        response=client.post("/predict-eml",files={"file":("large.eml",payload,"message/rfc822")})
    assert response.status_code == 413
