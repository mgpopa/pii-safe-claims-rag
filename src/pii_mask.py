import re, spacy
from utils import EMAIL_RE, PHONE_RE, POLICY_RE

# Load a lightweight English Named Entity Recognition (NER), e.g. under plceholders
try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    raise SystemExit(
        "spaCy model not found. Run:\n  python -m spacy download en_core_web_sm"
    ) from e

PLACEHOLDERS = {
    "PERSON": "[PERSON]", "ORG": "[ORG]", "GPE": "[GPE]", "LOC": "[LOC]",
    "NORP": "[GROUP]", "DATE": "[DATE]", "TIME": "[TIME]", "MONEY": "[MONEY]",
    "CARDINAL": "[NUM]"
}

def mask_text(text: str) -> str:
    # i first do a regex-based replacement (it's cheap and exact)
    text = EMAIL_RE.sub("[EMAIL]", text)

    def phone_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        return "[PHONE]" if len(digits) >= 9 else m.group(0)
    text = PHONE_RE.sub(phone_repl, text)

    text = POLICY_RE.sub("[POLICY_ID]", text)

    # then NER-based masking for names, places etc.
    doc = nlp(text)
    if not doc.ents:
        return text

    # build non-overlapping spans, so i don't mangle text
    spans = [(ent.start_char, ent.end_char, PLACEHOLDERS.get(ent.label_, f"[{ent.label_}]"))
             for ent in doc.ents]
    spans.sort()
    out = []
    last = 0
    for a, b, ph in spans:
        out.append(text[last:a])
        out.append(ph)
        last = b
    out.append(text[last:])
    return "".join(out)
