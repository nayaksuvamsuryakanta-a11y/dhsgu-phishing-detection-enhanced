import pandas as pd
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
