"""Synthetic-data university email phishing detector."""
from __future__ import annotations
import logging, re, uuid
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import urlparse
from difflib import SequenceMatcher
import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, LOG_DIR = BASE_DIR / "data", BASE_DIR / "models", BASE_DIR / "logs"
for folder in (DATA_DIR, MODEL_DIR, LOG_DIR): folder.mkdir(parents=True, exist_ok=True)
UNIVERSITY_NAME = "Dr. Harisingh Gour Vishwavidyalaya (Sagar University)"
# Official university domains should be verified and updated before operational use.
TRUSTED_DOMAINS = {"dhgsu.ac.in", "sagaruniversity.ac.in", "universityofsagar.ac.in"}
URGENT_WORDS = {"urgent", "immediately", "today", "deadline", "asap", "\u0924\u0941\u0930\u0902\u0924", "\u0936\u0940\u0918\u094d\u0930"}
FEAR_WORDS = {"suspended", "blocked", "terminated", "warning", "locked", "\u0928\u093f\u0932\u0902\u092c\u093f\u0924", "\u091a\u0947\u0924\u093e\u0935\u0928\u0940"}
REWARD_WORDS = {"scholarship", "refund", "prize", "reward", "selected", "grant"}
CREDENTIAL_WORDS = {"password", "verify", "login", "otp", "credential", "\u092a\u093e\u0938\u0935\u0930\u094d\u0921", "\u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924", "\u0916\u093e\u0924\u093e"}
MONEY_WORDS = {"fee payment", "payment", "bank", "refund", "transfer", "\u092b\u0940\u0938", "\u092d\u0941\u0917\u0924\u093e\u0928"}
AUTHORITY_WORDS = {"ugc", "aicte", "registrar", "examination cell", "ict centre"}
HINDI_URGENT_WORDS = {"\u0924\u0941\u0930\u0902\u0924", "\u092a\u093e\u0938\u0935\u0930\u094d\u0921", "\u0938\u0924\u094d\u092f\u093e\u092a\u093f\u0924", "\u0916\u093e\u0924\u093e", "\u0928\u093f\u0932\u0902\u092c\u093f\u0924", "\u091a\u0947\u0924\u093e\u0935\u0928\u0940"}
FEATURE_NAMES = "trusted_sender sender_domain_length sender_domain_similarity sender_lookalike reply_to_mismatch return_path_mismatch spf_fail dkim_fail dmarc_fail auth_failure num_links trusted_link untrusted_link suspicious_tld ip_url url_shortener http_only at_in_url login_keyword_url url_length_avg subdomain_count_max hyphen_domain url_domain_similarity anchor_mismatch_count anchor_mismatch_ratio urgent_words fear_words reward_words credential_words money_words authority_words hindi_urgent_words asks_password generic_greeting exclamation_count uppercase_ratio_subject university_name_but_untrusted_sender num_attachments suspicious_attachment double_extension macro_attachment archive_attachment script_attachment html_link_count html_button_count html_image_count has_hidden_input external_threat_score".split()
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

def _domain(value):
    value = str(value or "").strip()
    if "@" in value and "://" not in value: value = value.rsplit("@", 1)[-1].strip(" >")
    return (urlparse(value if "://" in value else "//" + value).hostname or "").lower().strip(".")

def _similarity(domain):
    if not domain: return 0.0
    if domain in TRUSTED_DOMAINS: return 1.0
    if any(t in domain for t in TRUSTED_DOMAINS): return .35
    return max((SequenceMatcher(None, domain.replace("-", ""), t.split(".")[0].replace("-", "")).ratio() for t in TRUSTED_DOMAINS), default=0.0)

def analyze_html(html):
    # HTML is untrusted input. Coerce odd values and keep parser failures from
    # taking down a whole prediction request.
    try:
        soup = BeautifulSoup(html if isinstance(html, str) else "", "html.parser")
    except Exception:
        soup = BeautifulSoup("", "html.parser")
    anchors = [{"href": a.get("href", ""), "text": a.get_text(" ", strip=True)} for a in soup.find_all("a")]
    hidden = any(i.get("type", "").lower() == "hidden" or "display:none" in str(i.get("style", "")).replace(" ", "").lower() for i in soup.find_all(["input", "div", "span"]))
    return {"anchors": anchors, "html_link_count": len(anchors), "html_button_count": len(soup.find_all("button")), "html_image_count": len(soup.find_all("img")), "has_hidden_input": int(hidden)}

def normalize_record(record):
    try:
        r = dict(record) if record is not None else {}
    except (TypeError, ValueError):
        r = {}
    def safe(value):
        if value is None: return ""
        try:
            if pd.isna(value): return ""
        except (TypeError, ValueError):
            pass
        return value
    r = {k: safe(v) for k, v in r.items()}
    out = {k: r.get(k, "") for k in ("subject", "body", "html", "sender", "reply_to", "return_path", "spf_result", "dkim_result", "dmarc_result", "text")}
    for key in ("links", "anchors", "attachments"):
        v = r.get(key, [])
        out[key] = [x.strip() for x in re.split(r"[,\n]", v) if x.strip()] if isinstance(v, str) else list(v or [])
    info = analyze_html(str(out["html"]))
    if not out["body"] and out["html"]: out["body"] = BeautifulSoup(str(out["html"]), "html.parser").get_text(" ", strip=True)
    if not out["body"]: out["body"] = out["text"] or ""
    if not out["anchors"]: out["anchors"] = info["anchors"]
    for k in ("html_link_count", "html_button_count", "html_image_count", "has_hidden_input"): out[k] = r.get(k) or info[k]
    links = list(out["links"]) + [str(a.get("href", "")) for a in out["anchors"] if isinstance(a, dict) and a.get("href")] + URL_RE.findall(str(out["body"]) + " " + str(out["html"]))
    out["links"] = list(dict.fromkeys(x.rstrip(".,;:") for x in links if x))
    out["text"] = (str(out["subject"]) + " " + str(out["body"])).strip()
    return out

def parse_email_bytes(raw_bytes):
    if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
        raise ValueError("The uploaded email is empty or invalid.")
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes); plain, html_parts, attachments = [], [], []
    for part in msg.walk():
        if part.is_multipart(): continue
        if part.get_filename(): attachments.append(part.get_filename()); continue
        content = part.get_content()
        if part.get_content_type() == "text/html": html_parts.append(str(content))
        elif part.get_content_type() == "text/plain": plain.append(str(content))
    meaningful_headers = ("subject", "from", "to", "date", "message-id", "reply-to", "return-path")
    if not any(str(msg.get(name, "")).strip() for name in meaningful_headers):
        raise ValueError("Uploaded file does not appear to contain a parsable email.")
    headers = " ".join(str(msg.get(h, "")) for h in ("Authentication-Results", "Received-SPF", "DMARC-Results", "X-DMARC-Results", "X-Spam-Status")).lower()
    def result(kind):
        m = re.search(r"\b" + kind + r"\s*=\s*(pass|fail|softfail|none|neutral|temperror|permerror)\b", headers)
        if not m and kind == "spf": m = re.search(r"received-spf:\s*(pass|fail|softfail|none)\b", headers)
        return m.group(1) if m else "unknown"
    return normalize_record({"subject": str(msg.get("subject", "")), "sender": str(msg.get("from", "")), "reply_to": str(msg.get("reply-to", "")), "return_path": str(msg.get("return-path", "")), "body": "\n".join(plain), "html": "\n".join(html_parts), "attachments": attachments, "spf_result": result("spf"), "dkim_result": result("dkim"), "dmarc_result": result("dmarc")})

def external_threat_intelligence_score(urls):
    """Offline score placeholder; future adapters may use VirusTotal, Google Safe Browsing, URLhaus, OpenPhish, or AbuseIPDB."""
    return float(min(1, sum(bool(re.search(r"\.(?:xyz|top|club|click|work|zip)$", _domain(u))) or "@" in u for u in urls) / max(1, len(urls))))

def extract_single_numeric_features(record):
    r = normalize_record(record); subject, body = str(r["subject"]), str(r["body"]); combined = (subject + " " + body).lower()
    sd = _domain(r["sender"]); sim = _similarity(sd); trusted = int(sd in TRUSTED_DOMAINS); urls = r["links"]; domains = [_domain(u) for u in urls]
    trusted_link = lambda d: any(d == t or d.endswith("." + t) for t in TRUSTED_DOMAINS)
    attachments = [str(a).lower() for a in r["attachments"]]; anchors = r["anchors"]
    mismatch = sum(bool(a.get("href") and a.get("text")) and "." in str(a["text"]) and _domain(a["text"]) != _domain(a["href"]) for a in anchors if isinstance(a, dict))
    count = lambda words: sum(len(re.findall(r"(?<!\w)" + re.escape(w) + r"(?!\w)", combined)) for w in words)
    bad = lambda k: int(str(r[k]).lower() in {"fail", "softfail", "none"})
    features = {"trusted_sender":trusted,"sender_domain_length":len(sd),"sender_domain_similarity":sim,"sender_lookalike":int(sim>=.68 and not trusted),"reply_to_mismatch":int(bool(r["reply_to"]) and _domain(r["reply_to"])!=sd),"return_path_mismatch":int(bool(r["return_path"]) and _domain(r["return_path"])!=sd),"spf_fail":bad("spf_result"),"dkim_fail":bad("dkim_result"),"dmarc_fail":bad("dmarc_result"),"auth_failure":int(any(bad(k) for k in ("spf_result","dkim_result","dmarc_result"))),"num_links":len(urls),"trusted_link":int(any(trusted_link(d) for d in domains)),"untrusted_link":sum(bool(d) and not trusted_link(d) for d in domains),"suspicious_tld":sum(bool(re.search(r"\.(?:xyz|top|club|click|work|zip|ru|tk)$",d)) for d in domains),"ip_url":sum(bool(re.match(r"\d{1,3}(?:\.\d{1,3}){3}$",d)) for d in domains),"url_shortener":sum(any(x in d for x in ("bit.ly","tinyurl","t.co","goo.gl")) for d in domains),"http_only":sum(u.lower().startswith("http://") for u in urls),"at_in_url":sum("@" in u for u in urls),"login_keyword_url":sum(bool(re.search(r"login|verify|account|password|otp",u,re.I)) for u in urls),"url_length_avg":float(np.mean([len(u) for u in urls])) if urls else 0,"subdomain_count_max":max((max(0,d.count(".")-1) for d in domains),default=0),"hyphen_domain":sum("-" in d for d in domains),"url_domain_similarity":max((_similarity(d) for d in domains),default=0),"anchor_mismatch_count":mismatch,"anchor_mismatch_ratio":mismatch/max(1,len(anchors)),"urgent_words":count(URGENT_WORDS),"fear_words":count(FEAR_WORDS),"reward_words":count(REWARD_WORDS),"credential_words":count(CREDENTIAL_WORDS),"money_words":count(MONEY_WORDS),"authority_words":count(AUTHORITY_WORDS),"hindi_urgent_words":count(HINDI_URGENT_WORDS),"asks_password":int(bool(re.search(r"(send|share|enter|submit|provide).{0,30}(password|otp|credential)",combined)) or any(word in combined for word in HINDI_URGENT_WORDS)),"generic_greeting":int(bool(re.search(r"dear (user|student|customer)|dear valued|प्रिय उपयोगकर्ता",combined))),"exclamation_count":(subject+body).count("!"),"uppercase_ratio_subject":sum(c.isupper() for c in subject)/max(1,sum(c.isalpha() for c in subject)),"university_name_but_untrusted_sender":int(any(n in combined for n in ("dhgsu","sagar university","university of saugar","harisingh gour")) and not trusted),"num_attachments":len(attachments),"suspicious_attachment":sum(bool(re.search(r"\.(?:exe|scr|js|vbs|bat|cmd|html?)$",a)) or ".pdf." in a or ".doc." in a for a in attachments),"double_extension":sum(bool(re.search(r"\.(?:pdf|docx?|xlsx?)\.(?:exe|js|scr|bat)$",a)) for a in attachments),"macro_attachment":sum(bool(re.search(r"\.docm?$|\.xlsm$",a)) for a in attachments),"archive_attachment":sum(bool(re.search(r"\.(?:zip|rar|7z)$",a)) for a in attachments),"script_attachment":sum(bool(re.search(r"\.(?:js|vbs|bat|cmd|ps1)$",a)) for a in attachments),"html_link_count":r["html_link_count"],"html_button_count":r["html_button_count"],"html_image_count":r["html_image_count"],"has_hidden_input":int(r["has_hidden_input"]),"external_threat_score":external_threat_intelligence_score(urls)}
    return {k:float(features.get(k,0)) for k in FEATURE_NAMES}

def extract_numeric_features(df):
    return pd.DataFrame([extract_single_numeric_features(row) for _,row in df.iterrows()],columns=FEATURE_NAMES).fillna(0)

_LEGIT_TEMPLATES = [
    ("exam_notice", ["Examination timetable published", "Examination Cell schedule for the semester", "Notice: upcoming university examinations"], ["The Examination Cell has published the semester timetable on the official student portal.", "Please consult the examination schedule at the university portal. Contact the Registrar's office with questions."]),
    ("results", ["Semester results available", "Result declaration notice", "University examination results"], ["The university has posted semester results in the student portal. Sign in through the address you normally use.", "Results are now available on the official portal. The Examination Cell will publish any corrections separately."]),
    ("fees", ["Fee payment schedule", "Reminder: semester fee information", "University fee notice"], ["This is a routine fee payment reminder. Review the amount and due date on the official university portal; do not reply with bank details.", "The Finance Office has published the semester fee schedule. Type the university portal address manually to review payment options."]),
    ("scholarship", ["Scholarship verification process", "Student scholarship status update", "Scholarship documentation notice"], ["Students can verify scholarship documentation through the official portal. Never send an OTP or password by email.", "The scholarship desk has updated its verification instructions. Visit the portal directly or contact the office during working hours."]),
    ("ict_security", ["ICT Centre security advisory", "Account safety guidance from ICT Centre", "Portal access: security reminder"], ["The ICT Centre advises users to type the university portal address manually instead of following links in email. Staff will never ask for your password.", "For account security, open the university site from your saved bookmark. This notice is informational; no credentials are requested."]),
    ("library_placement", ["Library return date reminder", "Placement Cell orientation notice", "Campus service announcement"], ["The Library asks members to check their account for book return dates. This routine reminder requires no payment.", "The Placement Cell has scheduled an orientation for registered students. Details are posted on the campus portal."]),
]
_PHISH_TEMPLATES = [
    ("suspension", ["Account access review", "Notice regarding student account", "Action requested for portal access"], ["Please verify your password to restore access; failure to respond may result in account suspension.", "Our records indicate that your account is blocked. Sign in now and confirm your credentials to avoid termination."]),
    ("scholarship_prize", ["Scholarship award confirmation", "Student grant notification", "Scholarship refund eligibility"], ["You have been selected for a scholarship refund. Submit your login and OTP to confirm the award.", "A grant has been reserved for your account. Verify your details and bank information using the link below."]),
    ("fake_results", ["Examination result access", "Result portal account notice", "Semester result availability"], ["Your result is ready. Use this sign-in page to view the marksheet and retain access to your account.", "The Examination Cell has placed a result document in your account. Confirm your password to open it."]),
    ("fake_fee", ["Outstanding fee payment", "Fee account review notice", "Payment record confirmation"], ["A payment remains pending on your student account. Complete the transfer through the linked page to avoid a late charge.", "Please confirm your fee payment details and bank information through this page for account reconciliation."]),
    ("polite_registrar", ["Registrar's office: document review", "Examination Cell document access", "Student record confirmation"], ["Dear Student, please review the online record at your convenience. The Registrar's office appreciates your cooperation.", "Good day. The Examination Cell has prepared a document for your review. Kindly use the secure university access page when convenient."]),
    ("polite_offer", ["Campus opportunity information", "Student placement correspondence", "University programme invitation"], ["Dear Student, we are pleased to invite you to review an opportunity. Kindly confirm your account details on the linked page at your convenience.", "Good afternoon. The selection committee has prepared your programme details. Please review them through the student sign-in page."]),
]
_SYNONYMS = {"published": ["posted", "made available", "released"], "official": ["university", "designated", "approved"], "please": ["kindly", "students are requested to"], "review": ["check", "consult", "read"], "portal": ["student site", "online portal", "website"], "notice": ["advisory", "announcement", "message"], "confirm": ["validate", "verify", "review"]}
_GREETINGS = ["", "Dear Student, ", "Good day. ", "Greetings from the university. "]
_SIGNATURES = ["", " Regards, University Office", " Sincerely, Student Services", " — ICT Centre"]

def _vary_text(subject, body, rng):
    for source, alternatives in _SYNONYMS.items():
        if source in body.lower() and rng.random() < .55:
            body = re.sub(r"\b" + source + r"\b", str(rng.choice(alternatives)), body, count=1, flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    if len(sentences) > 1 and rng.random() < .5: rng.shuffle(sentences)
    body = str(rng.choice(_GREETINGS)) + " ".join(sentences) + str(rng.choice(_SIGNATURES))
    if rng.random() < .06:
        body = re.sub(r"\b(the|your|please)\b", lambda m: m.group(0)[:-1] if len(m.group(0)) > 3 else m.group(0), body, count=1, flags=re.I)
    return subject, body

def generate_sample_dataset(n_legit=400,n_phish=400,random_state=42):
    rng=np.random.default_rng(random_state); rows=[]
    for label,n in ((0,n_legit),(1,n_phish)):
        templates = _LEGIT_TEMPLATES if label == 0 else _PHISH_TEMPLATES
        for i in range(n):
            family, subjects, bodies = templates[int(rng.integers(len(templates)))]
            subject,body=_vary_text(str(rng.choice(subjects)),str(rng.choice(bodies)),rng)
            if label==0:
                domain=str(rng.choice(sorted(TRUSTED_DOMAINS))); sender=f"{str(rng.choice(['ICT Centre','Examination Cell','Registrar Office','Student Services']))} <office{rng.integers(1,99)}@{domain}>"; link=f"https://{domain}/student/notice"; auth=("pass","pass","pass"); attach=[]; html=f'<p>{body}</p><a href="{link}">University portal notice</a>'
            else:
                formal=family in {"polite_registrar","polite_offer"}
                domains=["dhgsu-portal.xyz","sagar-university-login.club","mail-alert.top","secure-student.click","account-verify.work","dhgsu-portal.ac.in.verify-now.xyz","sagaruniversity-login.com"]
                domain=str(rng.choice(domains)); link=f"https://{domain}/student/record" if formal else f"http://{domain}/login/verify"
                display=str(rng.choice(["Examination Cell","Registrar Office","Student Services","ICT Centre"]))
                sender=f"{display} <office{rng.integers(1,99)}@{domain}>"
                auth=("pass","pass","pass") if rng.random()<.12 else tuple(str(rng.choice(x)) for x in (("fail","softfail","pass"),("fail","none","pass"),("fail","none","pass")))
                attach=[] if formal else ([str(rng.choice(["notice.pdf.exe","scholarship.zip","result.html","offer.docm"]))] if rng.random()<.28 else [])
                html=f'<p>{body}</p><a href="{link}">University record</a>' if rng.random()<.35 else f"<p>{body}</p>"
            rows.append({"subject":subject,"body":body,"html":html,"sender":sender,"reply_to":sender if label==0 else f"help@{domain}","return_path":sender if label==0 else f"bounce@{domain}","links":[link],"anchors":[],"attachments":attach,"spf_result":auth[0],"dkim_result":auth[1],"dmarc_result":auth[2],"template_group":f"{label}:{family}","label":label})
    return pd.DataFrame(rows).sample(frac=1,random_state=random_state).reset_index(drop=True)

def _matrix(df,vectorizer=None,fit=False):
    records=[normalize_record(r) for _,r in df.iterrows()]; texts=[r["text"]+" "+str(r["html"]) for r in records]
    vectorizer=vectorizer or TfidfVectorizer(ngram_range=(1,2),max_features=3000,sublinear_tf=True)
    xt=vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return hstack([xt,csr_matrix(extract_numeric_features(df).to_numpy(float))]).tocsr(),vectorizer

def train_models(df=None,test_size=.2,random_state=42,save=True):
    df=df if df is not None else generate_sample_dataset()
    if df.empty or df.label.nunique()<2: raise ValueError("Training data must contain both labels")
    # Keep complete template families in only one partition when enough groups exist.
    grouped="template_group" in df.columns and all(df.loc[df.label==label,"template_group"].nunique()>=3 for label in df.label.unique())
    if grouped:
        rng=np.random.default_rng(random_state); train_groups=[]; val_groups=[]; test_groups=[]
        for label in sorted(df.label.unique()):
            groups=np.array(sorted(df.loc[df.label==label,"template_group"].unique()),dtype=object); rng.shuffle(groups)
            n_test=max(1,min(len(groups)-2,int(np.ceil(len(groups)*test_size))))
            remain=groups[n_test:]; n_val=max(1,min(len(remain)-1,int(np.ceil(len(groups)*.2))))
            test_groups.extend(groups[:n_test]); val_groups.extend(remain[:n_val]); train_groups.extend(remain[n_val:])
        train=df[df.template_group.isin(train_groups)].copy(); val=df[df.template_group.isin(val_groups)].copy(); test=df[df.template_group.isin(test_groups)].copy()
    else:
        # Stratified three-way fallback for custom data without adequate groups.
        train_val,test=train_test_split(df,test_size=test_size,random_state=random_state,stratify=df.label)
        train,val=train_test_split(train_val,test_size=.2,random_state=random_state,stratify=train_val.label)
    Xtrain,vec=_matrix(train,fit=True); Xval,_=_matrix(val,vec); ytrain=train.label.astype(int); yval=val.label.astype(int)
    candidates={"Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced",random_state=random_state),"Random Forest":RandomForestClassifier(n_estimators=100,max_depth=18,class_weight="balanced",random_state=random_state,n_jobs=-1),"Gradient Boosting":GradientBoostingClassifier(n_estimators=80,max_depth=2,random_state=random_state),"MLPClassifier":MLPClassifier(hidden_layer_sizes=(48,),max_iter=180,early_stopping=True,random_state=random_state)}
    validation={}; thresholds={}
    for name,model in candidates.items():
        a,b=(Xtrain.toarray(),Xval.toarray()) if name=="Gradient Boosting" else (Xtrain,Xval)
        model.fit(a,ytrain); probs=model.predict_proba(b)[:,1]
        choices=np.unique(np.r_[np.linspace(.2,.8,25),.5])
        threshold=max(choices,key=lambda t:(f1_score(yval,probs>=t,zero_division=0),-abs(float(t)-.5)))
        validation[name]=float(f1_score(yval,probs>=threshold,zero_division=0)); thresholds[name]=float(threshold)
        LOGGER.info("%s validation F1: %.4f (threshold %.2f)",name,validation[name],threshold)
    bestname=max(validation,key=validation.get); development=pd.concat([train,val],ignore_index=True)
    Xdev,_=_matrix(development,vec); ydev=development.label.astype(int); model=candidates[bestname]
    model.fit(Xdev.toarray() if bestname=="Gradient Boosting" else Xdev,ydev)
    Xtest,_=_matrix(test,vec); test_input=Xtest.toarray() if bestname=="Gradient Boosting" else Xtest
    probs=model.predict_proba(test_input)[:,1]; pred=probs>=thresholds[bestname]; ytest=test.label.astype(int)
    tn,fp,fn,tp=confusion_matrix(ytest,pred,labels=[0,1]).ravel()
    metrics={"validation_f1":validation[bestname],"accuracy":float(accuracy_score(ytest,pred)),"precision":float(precision_score(ytest,pred,zero_division=0)),"recall":float(recall_score(ytest,pred,zero_division=0)),"f1":float(f1_score(ytest,pred,zero_division=0)),"roc_auc":float(roc_auc_score(ytest,probs)),"pr_auc":float(average_precision_score(ytest,probs)),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"false_positive_rate":float(fp/max(1,fp+tn)),"false_negative_rate":float(fn/max(1,fn+tp)),"best_threshold":thresholds[bestname]}
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=random_state); cv_X=Xdev.toarray() if bestname=="Gradient Boosting" else Xdev
    cvf=cross_val_score(clone(candidates[bestname]),cv_X,ydev,cv=cv,scoring="f1"); metrics["cross_validation_f1_mean"]=float(cvf.mean())
    LOGGER.info("Selected %s by validation F1 %.4f; independent test metrics: %s",bestname,metrics["validation_f1"],metrics)
    artifact={"model":model,"model_name":bestname,"input_dense":bestname=="Gradient Boosting","vectorizer":vec,"feature_names":FEATURE_NAMES.copy(),"threshold":thresholds[bestname],"threshold_selection":"validation partition; candidates selected by validation F1; test partition used only for final selected-model metrics","validation_f1":metrics["validation_f1"],"metrics":metrics,"university":UNIVERSITY_NAME,"trained_at":datetime.now(timezone.utc).isoformat()}
    if save: joblib.dump(artifact,MODEL_DIR/"phishing_artifact.joblib")
    return artifact

def load_artifact(path=None):
    path=Path(path) if path else MODEL_DIR/"phishing_artifact.joblib"
    return joblib.load(path) if path.exists() else train_models()

def rule_based_reasons(record,feats=None):
    r=normalize_record(record); f=feats or extract_single_numeric_features(r); reasons=[]
    checks=[(not f["trusted_sender"],"Sender domain is not trusted"),(f["reply_to_mismatch"],"Reply-To domain does not match From domain"),(f["return_path_mismatch"],"Return-Path domain does not match From domain"),(f["spf_fail"],"SPF authentication failed"),(f["dkim_fail"],"DKIM authentication failed"),(f["dmarc_fail"],"DMARC authentication failed"),(f["untrusted_link"],"Links point to non-university domains"),(f["suspicious_tld"],"Suspicious top-level domain detected"),(f["anchor_mismatch_count"],"Anchor text does not match actual URL"),(f["urgent_words"]+f["fear_words"],"Urgent or threatening language detected"),(f["credential_words"]+f["asks_password"],"Credential request detected"),(f["suspicious_attachment"],"Suspicious attachment detected"),(f["hindi_urgent_words"],"Hindi suspicious terms detected"),(f["sender_lookalike"],"Sender domain resembles a university domain")]
    reasons=[text for yes,text in checks if yes]
    return reasons or ["No strong phishing indicators were detected by the available checks."]

def top_model_reasons(feature_vector,model,feature_names,limit=8):
    try:
        coefs=getattr(model,"feature_importances_",None)
        signed = False
        if coefs is None and hasattr(model,"coef_"):
            coefs=model.coef_.toarray()[0] if hasattr(model.coef_,"toarray") else model.coef_[0]; signed=True
        if coefs is None and hasattr(model,"coefs_"): coefs=np.abs(model.coefs_[0]).mean(axis=1)
        if coefs is None:return []
        values=feature_vector.toarray().ravel() if hasattr(feature_vector,"toarray") else np.asarray(feature_vector).ravel()
        off=max(0,len(values)-len(feature_names)); pairs=[(feature_names[i],float(values[off+i]*coefs[off+i])) for i in range(min(len(feature_names),len(coefs)-off)) if values[off+i]>0]
        if signed: pairs=[pair for pair in pairs if pair[1]>0]
        return [x.replace("_"," ").title() for x,_ in sorted(pairs,key=lambda x:x[1],reverse=True)[:limit]]
    except Exception:return []

def shap_top_reasons(model,feature_vector,feature_names,limit=8):
    try:
        import shap
        values=shap.TreeExplainer(model).shap_values(feature_vector); arr=np.asarray(values[-1] if isinstance(values,list) else values).ravel()
        return [feature_names[i] for i in np.argsort(abs(arr))[-limit:][::-1] if i<len(feature_names)]
    except Exception:return []

def predict_record(record,artifact=None,threshold=None):
    r=normalize_record(record); artifact=artifact or load_artifact(); X,_=_matrix(pd.DataFrame([r]),artifact["vectorizer"]); model_input=X.toarray() if artifact.get("input_dense") else X; prob=float(artifact["model"].predict_proba(model_input)[0,1]); used=float(threshold if threshold is not None else artifact["threshold"]); feats=extract_single_numeric_features(r); phishing=prob>=used; score=round(prob*100); level="Low Risk" if score<=20 else "Medium Risk" if score<=50 else "High Risk" if score<=80 else "Critical Risk"
    reasons=rule_based_reasons(r,feats); fallback="No strong phishing indicators were detected by the available checks."
    auth_missing=any(str(r.get(k, "")).strip().lower() in {"", "unknown"} for k in ("spf_result", "dkim_result", "dmarc_result"))
    missing_evidence=[]
    if not _domain(r["sender"]): missing_evidence.append("sender domain")
    if not r["links"]: missing_evidence.append("links")
    if auth_missing: missing_evidence.append("SPF/DKIM/DMARC results")
    mr=top_model_reasons(X[0],artifact["model"],artifact["feature_names"])
    if not phishing and level == "Low Risk":
        # Low-risk legitimate messages should be explained with validating
        # evidence, not raw keyword matches that ignore their context.
        reasons=[]
        if feats["trusted_sender"]:
            reasons.append("Sender domain matches the configured trusted university domains.")
        if feats["trusted_link"]:
            reasons.append("Links point to configured official university portals.")
        if all(str(r.get(k, "")).lower() == "pass" for k in ("spf_result", "dkim_result", "dmarc_result")):
            reasons.append("SPF, DKIM, and DMARC indicators pass in the supplied headers.")
        has_security_terms=bool(feats["credential_words"] or feats["asks_password"] or feats["urgent_words"] or feats["fear_words"])
        if has_security_terms:
            reasons.append("Contains routine security terminology, but sender and available link/authentication indicators are consistent with a trusted notice.")
        if not reasons:
            reasons.append("The message has a low model risk score and no material sender, link, or authentication concerns were detected.")
    elif phishing and (not reasons or reasons==[fallback]):
        if mr: reasons=["Model evidence contributing to this phishing classification: "+", ".join(mr[:5])]
        else: reasons=[f"The model assigned phishing probability {prob:.0%}, above its decision threshold {used:.0%}."]
    if phishing and missing_evidence:
        reasons.insert(0,"Limited evidence: missing "+", ".join(missing_evidence)+"; this prediction relies mainly on available message text and indicators. Verify through a known university channel.")
    elif not phishing and level != "Low Risk":
        # Model feature weights are not threat verdicts; label them as context
        # and avoid presenting keyword indicators as confirmed malicious acts.
        reasons=[reason for reason in reasons if reason != fallback]
        if feats["trusted_sender"] and feats["credential_words"]:
            reasons=["Routine security terminology appears in a message from a configured trusted sender; review context if unexpected."]
        if not reasons: reasons=["No specific rule indicator was triggered; the model classified this message as legitimate."]
    elif mr and reasons != [fallback]: reasons.append("Supporting model features: "+", ".join(mr[:5]))
    recs=["Do not click links or open attachments.","Do not share passwords, OTPs, or payment information.","Verify via a known official university channel.","Report to university IT/security."] if phishing else ["Verify unexpected requests through a known university channel.","Never share passwords or OTPs by email."]
    return {"id":str(uuid.uuid4()),"timestamp":datetime.now(timezone.utc).isoformat(),"university":UNIVERSITY_NAME,"subject":str(r["subject"]),"sender":str(r["sender"]),"classification":"Phishing" if phishing else "Legitimate","risk_score_percent":int(score),"probability_phishing":prob,"risk_level":level,"threshold_used":used,"reasons":reasons,"recommendations":recs,"features":feats,"links":r["links"],"attachments":r["attachments"],"model_name":artifact["model_name"]}

def log_prediction(result):
    path=LOG_DIR/"predictions.csv"; row={k:result.get(k) for k in ("id","timestamp","university","subject","sender","classification","risk_score_percent","risk_level","model_name")}; pd.DataFrame([row]).to_csv(path,mode="a",header=not path.exists(),index=False)

def record_feedback(result_id,feedback,comment=""):
    path=LOG_DIR/"feedback.csv"; pd.DataFrame([{"result_id":result_id,"timestamp":datetime.now(timezone.utc).isoformat(),"feedback":str(feedback),"comment":comment}]).to_csv(path,mode="a",header=not path.exists(),index=False)

def generate_report(result):
    lines=[f"University: {result.get('university',UNIVERSITY_NAME)}",f"Report ID: {result.get('id','')}",f"Timestamp: {result.get('timestamp','')}",f"Subject: {result.get('subject','')}",f"Sender: {result.get('sender','')}",f"Classification: {result.get('classification','')}",f"Risk score: {result.get('risk_score_percent',0)}%",f"Risk level: {result.get('risk_level','')}","Detected links: "+(", ".join(result.get("links",[])) or "None"),"Attachments: "+(", ".join(result.get("attachments",[])) or "None"),"Reasons:"]
    lines += ["- "+x for x in result.get("reasons",[])]+["Recommendations:"]+["- "+x for x in result.get("recommendations",[])]+["Disclaimer: Human review is required."]
    return "\n".join(lines)

def scan_directory(directory,artifact=None):
    results=[]; path=Path(directory)
    if path.is_dir():
        for eml in path.rglob("*.eml"):
            try:
                result=predict_record(parse_email_bytes(eml.read_bytes()),artifact); log_prediction(result); results.append(result)
            except Exception as exc: LOGGER.warning("Could not process %s: %s",eml,exc)
    return results

CANONICAL_LEGITIMATE_SAMPLE={"subject":"ICT Centre portal security advisory","body":"The ICT Centre advises users to type the university portal address manually instead of following links in email. Staff will never ask for your password.","sender":"ICT Centre <security@dhgsu.ac.in>","links":["https://dhgsu.ac.in/ict/security-advisory"],"spf_result":"pass","dkim_result":"pass","dmarc_result":"pass"}

if __name__=="__main__":
    artifact=train_models()
    LOGGER.info("Training complete; selected model: %s",artifact["model_name"])
    # Explicitly demonstrate both classes using synthetic records only.
    demos = [
        CANONICAL_LEGITIMATE_SAMPLE,
        {"subject":"URGENT account suspended", "body":"Verify your password and OTP immediately or your account will be blocked!", "sender":"support@dhgsu-portal.xyz", "links":["http://dhgsu-portal.xyz/login"], "attachments":["notice.pdf.exe"], "spf_result":"fail", "dkim_result":"fail", "dmarc_result":"fail"},
    ]
    for demo in demos:
        result = predict_record(demo, artifact)
        LOGGER.info("Top contributing numeric features: %s", ", ".join(top_model_reasons(_matrix(pd.DataFrame([normalize_record(demo)]), artifact["vectorizer"])[0][0], artifact["model"], artifact["feature_names"])) or "not available for this model")
        LOGGER.info("Demo prediction: classification=%s risk_score=%d%% risk_level=%s", result["classification"], result["risk_score_percent"], result["risk_level"])
        LOGGER.info("Reasons: %s", "; ".join(result["reasons"]))
        LOGGER.info("Recommendations: %s", "; ".join(result["recommendations"]))
        if demo is CANONICAL_LEGITIMATE_SAMPLE:
            assert result["classification"] == "Legitimate" and result["risk_level"] in {"Low Risk", "Medium Risk"}, "Canonical legitimate demo should be low or medium risk."
