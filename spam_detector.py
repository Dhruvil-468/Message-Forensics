"""
Hybrid Spam Detector
=====================
A spam classifier that combines:
  1. A standard TF-IDF + ML text classifier (the "what words are used" signal)
  2. A hand-built "Manipulation Tactics" scorer (the "how is it trying to
     manipulate you" signal) — urgency, scarcity, financial bait, fake
     authority, excessive punctuation/caps, suspicious links, etc.

The unique part: instead of a single opaque probability, every prediction
comes with a human-readable EXPLANATION showing which tactics were detected
and how much each one (and the text model) contributed to the verdict.
This makes the model's reasoning auditable — useful for trust, debugging,
and for teaching people to recognize manipulative messages themselves.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from narrative import generate_narrative

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. MANIPULATION TACTIC FEATURES  (the "unique" layer)
# ---------------------------------------------------------------------------
# Each detector below returns a 0-1ish score for one social-engineering
# tactic commonly used in spam/scam messages. These are combined into a
# "Tactic Score" that is both (a) fed to the ML model as extra features and
# (b) shown directly to the user as an explanation.

URGENCY_WORDS = [
    "urgent", "immediately", "act now", "expires", "expiring", "hurry",
    "last chance", "final notice", "limited time", "right away",
    "today only", "before it's too late", "act fast", "don't wait",
    "respond now", "24 hours", "deadline",
]

SCARCITY_WORDS = [
    "limited", "only a few", "exclusive", "while supplies last",
    "selected", "chosen", "one time", "won't last", "almost gone",
    "few left", "rare opportunity",
]

MONEY_BAIT_WORDS = [
    "free", "win", "winner", "won", "cash", "prize", "reward", "$",
    "£", "claim", "guaranteed", "no cost", "risk free", "bonus",
    "million", "lottery", "jackpot", "refund", "credit",
]

AUTHORITY_WORDS = [
    "bank", "irs", "government", "official", "security", "account",
    "verify", "suspended", "police", "legal action", "court", "amazon",
    "paypal", "microsoft", "apple support",
]

PRESSURE_WORDS = [
    "click here", "click below", "call now", "text back", "reply now",
    "confirm now", "verify now", "don't tell anyone", "keep this private",
]

TACTIC_LEXICONS = {
    "urgency": URGENCY_WORDS,
    "scarcity": SCARCITY_WORDS,
    "money_bait": MONEY_BAIT_WORDS,
    "false_authority": AUTHORITY_WORDS,
    "pressure_to_act": PRESSURE_WORDS,
}

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(?:com|net|biz|info|xyz|ru)\b)", re.I)
PHONE_RE = re.compile(r"\b(\+?\d[\d\-\s]{7,}\d)\b")


def _lexicon_hits(text_lower, words):
    return sum(1 for w in words if w in text_lower)


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)


def punctuation_intensity(text):
    """Excess exclamation marks / question marks — classic spam tell."""
    exclaims = text.count("!")
    questions = text.count("?")
    return min((exclaims + questions) / 5.0, 1.0)  # cap at 1.0


def has_url(text):
    return 1 if URL_RE.search(text) else 0


def has_phone(text):
    return 1 if PHONE_RE.search(text) else 0


def tactic_scores(text):
    """Return a dict of individual tactic scores (0-1 each) for one message."""
    t_lower = text.lower()
    scores = {}
    for tactic, words in TACTIC_LEXICONS.items():
        hits = _lexicon_hits(t_lower, words)
        scores[tactic] = min(hits / 3.0, 1.0)  # 3+ hits saturates the score
    scores["excess_caps"] = min(caps_ratio(text) * 2, 1.0) if len(text) > 15 else 0.0
    scores["excess_punct"] = punctuation_intensity(text)
    scores["has_url"] = float(has_url(text))
    scores["has_phone"] = float(has_phone(text))
    return scores


TACTIC_LABELS = {
    "urgency": "Urgency",
    "scarcity": "Scarcity",
    "money_bait": "Money bait",
    "false_authority": "False authority",
    "pressure_to_act": "Pressure to act",
}


def tactic_matches(text):
    """
    Return a list of {start, end, tactic, label} spans for every lexicon
    word/phrase found in the text (case-insensitive), plus synthetic spans
    for detected URLs and phone numbers. Used to highlight the exact
    evidence in the original message, not just report a score.
    """
    spans = []
    t_lower = text.lower()
    for tactic, words in TACTIC_LEXICONS.items():
        for w in words:
            start = 0
            while True:
                idx = t_lower.find(w, start)
                if idx == -1:
                    break
                spans.append({
                    "start": idx, "end": idx + len(w),
                    "tactic": tactic, "label": TACTIC_LABELS[tactic],
                })
                start = idx + len(w)
    for m in URL_RE.finditer(text):
        spans.append({"start": m.start(), "end": m.end(), "tactic": "has_url", "label": "Link"})
    for m in PHONE_RE.finditer(text):
        spans.append({"start": m.start(), "end": m.end(), "tactic": "has_phone", "label": "Phone number"})

    # remove spans fully contained within a longer span (avoid double highlight)
    spans.sort(key=lambda s: (s["start"], -(s["end"] - s["start"])))
    filtered = []
    last_end = -1
    for s in spans:
        if s["start"] >= last_end:
            filtered.append(s)
            last_end = s["end"]
    return filtered


def tactic_feature_matrix(texts):
    """Vectorized tactic-score feature matrix for a list/Series of texts."""
    rows = [tactic_scores(t) for t in texts]
    df = pd.DataFrame(rows)
    return df


TACTIC_COLUMNS = [
    "urgency", "scarcity", "money_bait", "false_authority", "pressure_to_act",
    "excess_caps", "excess_punct", "has_url", "has_phone",
]


# ---------------------------------------------------------------------------
# 2. HYBRID MODEL
# ---------------------------------------------------------------------------

class HybridSpamDetector:
    """
    TF-IDF + Naive Bayes text model, blended with the hand-built tactic
    score via a small logistic-regression combiner. This gives the
    accuracy of a trained text model plus the interpretability of an
    explicit rule layer, without giving up either.
    """

    def __init__(self, tactic_weight=0.35):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000, min_df=2
        )
        self.text_model = MultinomialNB()
        self.tactic_weight = tactic_weight  # how much the tactic score influences the final verdict

    def fit(self, texts, labels):
        X_text = self.vectorizer.fit_transform(texts)
        self.text_model.fit(X_text, labels)
        return self

    def _text_proba(self, texts):
        X_text = self.vectorizer.transform(texts)
        return self.text_model.predict_proba(X_text)[:, 1]  # P(spam)

    def _tactic_proba(self, texts):
        tf = tactic_feature_matrix(texts)
        # simple weighted-sum heuristic scaled to look like a probability
        raw = tf[TACTIC_COLUMNS].mean(axis=1)
        return raw.values

    def predict_proba(self, texts):
        texts = list(texts)
        p_text = self._text_proba(texts)
        p_tactic = self._tactic_proba(texts)
        blended = (1 - self.tactic_weight) * p_text + self.tactic_weight * p_tactic
        return blended, p_text, p_tactic

    def predict(self, texts, threshold=0.5):
        blended, _, _ = self.predict_proba(texts)
        return (blended >= threshold).astype(int)

    def verdict_tier(self, text, spam_threshold=0.5, suspicious_tactic_threshold=0.75):
        """
        Three-tier verdict: SPAM / SUSPICIOUS / HAM.
        A message lands in SUSPICIOUS if the text model alone isn't confident
        enough to call it spam, but a single manipulation tactic (e.g.
        impersonating a bank, or a classic phishing pressure line) is firing
        very strongly. This catches phishing-style messages that don't look
        like "typical" bag-of-words spam but use textbook social-engineering
        tactics — the kind of message a pure text classifier often misses.
        """
        blended, p_text, p_tactic = self.predict_proba([text])
        blended = blended[0]
        scores = tactic_scores(text)
        max_tactic = max(scores.values()) if scores else 0.0

        if blended >= spam_threshold:
            return "SPAM", blended
        if max_tactic >= suspicious_tactic_threshold:
            return "SUSPICIOUS", blended
        return "HAM", blended

    def analyze(self, text, spam_threshold=0.5, suspicious_tactic_threshold=0.75):
        """Structured, JSON-ready analysis of a single message (for the API)."""
        blended, p_text, p_tactic = self.predict_proba([text])
        blended, p_text, p_tactic = float(blended[0]), float(p_text[0]), float(p_tactic[0])
        scores = tactic_scores(text)
        tier, _ = self.verdict_tier(text, spam_threshold, suspicious_tactic_threshold)
        matches = tactic_matches(text)
        result = {
            "text": text,
            "verdict": tier,
            "confidence": round(blended, 4),
            "text_model_score": round(p_text, 4),
            "tactic_score": round(p_tactic, 4),
            "tactics": {k: round(v, 4) for k, v in scores.items() if v > 0},
            "matches": matches,
        }
        result["narrative"] = generate_narrative(result)
        return result

    def explain(self, text, threshold=0.5):
        """Human-readable explanation for a single message."""
        blended, p_text, p_tactic = self.predict_proba([text])
        blended, p_text, p_tactic = blended[0], p_text[0], p_tactic[0]
        scores = tactic_scores(text)
        active_tactics = {k: v for k, v in scores.items() if v > 0}
        tier, _ = self.verdict_tier(text, threshold)
        verdict = {"SPAM": "SPAM", "SUSPICIOUS": "SUSPICIOUS (worth a second look)",
                   "HAM": "HAM (not spam)"}[tier]

        lines = []
        lines.append(f'Message: "{text}"')
        lines.append(f"Verdict: {verdict}  (confidence: {blended:.0%})")
        lines.append(f"  - Text model (word patterns) score:    {p_text:.0%}")
        lines.append(f"  - Manipulation-tactic score:            {p_tactic:.0%}")
        if active_tactics:
            lines.append("  - Tactics detected:")
            labels_map = {
                "urgency": "Urgency language ('act now', 'immediately'...)",
                "scarcity": "Scarcity language ('limited', 'exclusive'...)",
                "money_bait": "Money/prize bait ('free', 'win', '$'...)",
                "false_authority": "Impersonating authority (bank, IRS, brand...)",
                "pressure_to_act": "Pressure to act now ('click here', 'call now'...)",
                "excess_caps": "Unusual amount of ALL CAPS",
                "excess_punct": "Excessive punctuation (!!! / ???)",
                "has_url": "Contains a link",
                "has_phone": "Contains a phone number",
            }
            for tactic, score in sorted(active_tactics.items(), key=lambda x: -x[1]):
                lines.append(f"      • {labels_map.get(tactic, tactic)}  (strength {score:.0%})")
        else:
            lines.append("  - No manipulation tactics detected.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. TRAIN / EVALUATE (runs when executed directly)
# ---------------------------------------------------------------------------

def load_data(path="spam_raw.csv"):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]
    df["y"] = (df["label"] == "spam").astype(int)
    return df


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["y"]
    )

    model = HybridSpamDetector(tactic_weight=0.35)
    model.fit(X_train, y_train)

    blended, p_text, p_tactic = model.predict_proba(X_test)
    y_pred = (blended >= 0.5).astype(int)

    print("=" * 60)
    print("HYBRID MODEL PERFORMANCE (TF-IDF + Naive Bayes + Tactic Score)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, blended):.4f}")
    print()
    print("Confusion Matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Compare: text-only baseline vs hybrid, to show the tactic layer's value
    y_pred_text_only = (p_text >= 0.5).astype(int)
    print("-" * 60)
    print("Baseline (text-only, no tactic layer) for comparison:")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_text_only):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred_text_only):.4f}")
    print("-" * 60)

    # Example explanations
    print("\n" + "=" * 60)
    print("EXAMPLE EXPLANATIONS")
    print("=" * 60)
    examples = [
        "URGENT! You have WON a $1000 Walmart gift card. Claim now at bit.ly/xyz before it expires!!!",
        "Hey, are we still on for lunch tomorrow at 1pm?",
        "Your bank account has been suspended. Verify your identity immediately by calling 1-800-555-0199.",
        "Don't forget to pick up milk on your way home.",
    ]
    for ex in examples:
        print()
        print(model.explain(ex))

    return model, df


if __name__ == "__main__":
    main()
