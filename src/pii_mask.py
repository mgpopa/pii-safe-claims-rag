import spacy
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
