"""Streamlit dashboard for the local HGU phishing detector."""
import pandas as pd
import streamlit as st
from enhanced_phishing_detector import LOG_DIR, generate_report, load_artifact, log_prediction, normalize_record, parse_email_bytes, predict_record, record_feedback, scan_directory

st.set_page_config(page_title="HGU Phishing Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ HGU University Phishing Detection")
st.caption("AI-assisted email screening for Dr. Harisingh Gour Vishwavidyalaya")
st.warning("Demonstration login only. Production deployments must use secure authentication.")
USERS = {"admin": ("admin123", "admin"), "user": ("user123", "user")}
if "role" not in st.session_state:
    with st.form("login"):
        username = st.text_input("Username"); password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if username in USERS and USERS[username][0] == password:
                st.session_state.role = USERS[username][1]; st.rerun()
            st.error("Invalid demo credentials")
    st.stop()
if st.sidebar.button("Log out"):
    del st.session_state.role; st.rerun()

@st.cache_resource
def get_artifact():
    return load_artifact()

artifact = get_artifact(); role = st.session_state.role
tab_names = ["Analyze Email"] + (["Scan EML Folder", "Prediction Logs", "Feedback Logs"] if role == "admin" else [])
tabs = st.tabs(tab_names)
with tabs[0]:
    uploaded = st.file_uploader("Upload an email (.eml)", type=["eml"])
    with st.form("analyze"):
        subject = st.text_input("Subject"); sender = st.text_input("Sender", placeholder="office@dhgsu.ac.in")
        reply_to = st.text_input("Reply-To"); return_path = st.text_input("Return-Path")
        c1,c2,c3 = st.columns(3)
        spf=c1.selectbox("SPF result",["unknown","pass","fail","softfail","none"]); dkim=c2.selectbox("DKIM result",["unknown","pass","fail","none"]); dmarc=c3.selectbox("DMARC result",["unknown","pass","fail","none"])
        links=st.text_area("Links (one per line)"); attachments=st.text_input("Attachment names (comma separated)")
        body=st.text_area("Body",height=180); html_body=st.text_area("HTML body",height=120)
        submitted=st.form_submit_button("Analyze email")
    if submitted:
        rec=parse_email_bytes(uploaded.getvalue()) if uploaded else normalize_record({"subject":subject,"sender":sender,"reply_to":reply_to,"return_path":return_path,"spf_result":spf,"dkim_result":dkim,"dmarc_result":dmarc,"links":links,"attachments":attachments,"body":body,"html":html_body})
        result=predict_record(rec,artifact); log_prediction(result); st.session_state.last_result=result
    result=st.session_state.get("last_result")
    if result:
        icon="🔴" if result["classification"]=="Phishing" else "🟢"
        st.subheader(f"{icon} {result['classification']} — {result['risk_score_percent']}% ({result['risk_level']})")
        st.write(f"Threshold: {result['threshold_used']:.2f} · Model: {result['model_name']}")
        left,right=st.columns(2)
        with left:
            st.markdown("**Explanation**")
            for item in result["reasons"]: st.write("• "+item)
        with right:
            st.markdown("**Recommended actions**")
            for item in result["recommendations"]: st.write("• "+item)
        f=result["features"]
        st.markdown("**Indicator summary**")
        st.dataframe(pd.DataFrame([{ "Indicator":"Sender domain","Value":result["sender"].split("@")[-1]}, {"Indicator":"Sender lookalike score","Value":f["sender_domain_similarity"]}, {"Indicator":"SPF / DKIM / DMARC failure","Value":f"{f['spf_fail']} / {f['dkim_fail']} / {f['dmarc_fail']}"}, {"Indicator":"Untrusted links","Value":f["untrusted_link"]}, {"Indicator":"Suspicious TLD","Value":f["suspicious_tld"]}, {"Indicator":"Anchor mismatch","Value":f["anchor_mismatch_count"]}, {"Indicator":"Suspicious attachment","Value":f["suspicious_attachment"]}, {"Indicator":"Urgent language","Value":f["urgent_words"]+f["fear_words"]}, {"Indicator":"Hindi suspicious terms","Value":f["hindi_urgent_words"]}]),hide_index=True,use_container_width=True)
        st.download_button("Download report",generate_report(result),file_name="hgu_phishing_report.txt")
        a,b=st.columns(2)
        if a.button("Prediction is Correct"): record_feedback(result["id"],"correct"); st.success("Feedback recorded")
        if b.button("Prediction is Incorrect"): record_feedback(result["id"],"incorrect"); st.success("Feedback recorded")
if role == "admin":
    with tabs[1]:
        folder=st.text_input("Folder path to scan recursively")
        if st.button("Scan .eml files"):
            st.session_state.scan_results=scan_directory(folder,artifact)
        if st.session_state.get("scan_results"):
            st.dataframe(pd.DataFrame([{k:r[k] for k in ("subject","sender","classification","risk_score_percent","risk_level")} for r in st.session_state.scan_results]),use_container_width=True)
    for index,filename in ((2,"predictions.csv"),(3,"feedback.csv")):
        with tabs[index]:
            st.subheader(filename); path=LOG_DIR/filename
            if path.exists(): st.dataframe(pd.read_csv(path),use_container_width=True)
            else: st.info("No records yet.")
