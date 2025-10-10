import re, time, phonenumbers
import numpy as np

def timer():
    start = time.time()
    def done():
        return time.time() - start
    return done

# some simple patterns for PII
EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#PHONE_RE  = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3}[\s-]?\d{2,4}[\s-]?\d{2,4})")
#PHONE_RE  = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)"
#                       r"(?:\s*(?:ext\.?|x)\s*\d{1,6})?",
#                       re.IGNORECASE)
POLICY_RE = re.compile(r"\b([A-Z0-9]{2,4}-?[A-Z0-9]{2,4}-?[A-Z0-9]{2,4})\b")  # my synthetic policy id

def strip_placeholders(text: str) -> str:
    # remove anz [TOKEN] placeholders so detectors never see them
    return re.sub(r"\[[A-Z_]+\]", " ", text)

def contains_pii(text: str) -> bool:
    if any(tag in text for tag in ["[EMAIL]", "[PHONE]", "[POLICY_ID]"]):
        return False
    text = strip_placeholders(text)

    if EMAIL_RE.search(text): return True
    if POLICY_RE.search(text): return True

    # use the same detector as masking for phones
    # treat any match as PII
    for m in phonenumbers.PhoneNumberMatcher(text, region=None):
        return True # ignore tiny matches
    return False

    # TO BE REMOVED:
    # i require phone no. to be at least 10 digits to count as PII
    #for m in PHONE_RE.findall(text):
    #    digits = re.sub(r"\D", "", m)
    #    if len(digits) >= 10:
    #        return True
    return False
