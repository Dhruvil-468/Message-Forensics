# Hybrid Spam Detector

A spam/ham classifier trained on the classic **SMS Spam Collection**
dataset (5,572 real text messages, ~13% spam), with one unique twist:

## The unique part: a "Manipulation Tactics" layer

Most spam classifiers are a black box — TF-IDF + Naive Bayes/Logistic
Regression that spits out a probability with no explanation. This one adds
a second, transparent scoring layer that detects specific **social
engineering tactics** independent of the word-frequency model:

| Tactic | Examples detected |
|---|---|
| Urgency | "act now", "immediately", "expires", "last chance" |
| Scarcity | "limited", "exclusive", "while supplies last" |
| Money bait | "free", "win", "$", "prize", "guaranteed" |
| False authority | "bank", "IRS", "PayPal", "verify", "suspended" |
| Pressure to act | "click here", "call now", "confirm now" |
| Excess CAPS / punctuation | ALL-CAPS ratio, `!!!`/`???` density |
| Contains link / phone number | regex detection |

These scores are:
1. **Blended into the model's prediction** (weighted alongside the TF-IDF +
   Naive Bayes text score), and
2. **Shown to the user as a plain-English explanation** for every
   prediction — which tactics fired, and how strongly.

### Three-tier verdict: SPAM / SUSPICIOUS / HAM
A pure text classifier trained on SMS spam sometimes misses phishing-style
messages that don't "look like" typical spam (e.g. *"Your bank account has
been suspended. Verify your identity immediately by calling..."*). The
tactic layer catches these: if a single tactic (like authority
impersonation) fires very strongly even when the text model is unsure, the
message is flagged **SUSPICIOUS** rather than silently passed as ham.

## Files
- `spam_detector.py` — data loading, tactic-scoring functions (including
  exact text-span matching for highlighting), the `HybridSpamDetector`
  class, training, and evaluation.
- `demo.py` — interactive CLI: type any message and get a live
  classification + explanation.
- `app.py` — FastAPI backend. Trains the real model on startup, handles
  login/signup, and serves analysis + evidence-log endpoints.
- `db.py` — SQLite persistence for user accounts and each user's analysis
  history ("evidence log"). Creates `spam_forensics.db` on first run.
- `auth.py` — signed-cookie session auth (no third-party auth service —
  passwords are salted + hashed with PBKDF2, sessions are a signed cookie).
- `narrative.py` — rule-based natural-language explanation generator:
  turns scores and detected tactics into plain-English paragraphs.
- `pdf_report.py` — renders a single case (evidence, verdict, stats,
  tactic log, narrative) into a downloadable, styled PDF report.
- `static/index.html` — the web frontend ("Message Forensics" case-file
  design): sign in, paste a message, watch the exact trigger words get
  highlighted inline by tactic, and a verdict get stamped (SPAM /
  SUSPICIOUS / CLEARED), with a written explanation underneath. Includes
  an Evidence Log tab to review, print, or delete past analyses, download
  a per-case PDF report, and export the whole log as JSON.
- `spam_raw.csv` — the training dataset (SMS Spam Collection).
- `requirements.txt` — all dependencies.

## Run the CLI

```bash
pip install -r requirements.txt
python3 spam_detector.py     # trains, evaluates, prints example explanations
python3 demo.py              # interactive: classify your own messages
```

## Run the web app

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000**. This is a real backend — the model
trains on startup from `spam_raw.csv`, and every analysis in the UI is a
live call to the actual `HybridSpamDetector`.

**Accounts & evidence log:** the first time you open the app you'll be
asked to sign in. Click "Open a case file" to create an account (a
username + passphrase, stored locally in `spam_forensics.db` — nothing
leaves your machine). Every message you analyze while signed in is saved
to your personal **Evidence Log** tab, where you can:
- reopen any past case and see the same annotated highlighting again
- read a **plain-English narrative explanation** of why the model reached
  that verdict — not just scores, actual written-out reasoning
- **download a one-click PDF evidence report** for any single case (also
  available right after analyzing, before you even leave the Analyze tab)
- delete individual entries
- print a single case as a clean record
- export your entire log as a timestamped `.json` file

Sessions are a signed cookie (7-day expiry) — no data is stored in the
browser itself, everything lives server-side in the SQLite database next
to `app.py`.

### The narrative explanation

Every analysis includes a `narrative` field: several paragraphs of
deterministic, rule-based prose (see `narrative.py`) that explain the
verdict the way a person would — what the word-pattern model saw, which
manipulation tactics fired and on which exact words, why a SUSPICIOUS
verdict overrode a low text-model score when relevant, and a plain
recommendation. It's not an LLM call — it's template logic driven by the
same tactic scores already computed — so it's instant, free, and
reproducible.

### The PDF evidence report

`pdf_report.py` renders a single case — highlighted message, verdict
badge, full statistics, tactic log, and the narrative explanation — into
a downloadable PDF styled to match the web UI. Available via:
- the **Download evidence report (PDF)** button right after analyzing
- the same button on any entry in the Evidence Log detail view
- directly at `GET /api/history/{id}/report`

## Results (held-out 20% test set)

| Metric | Hybrid model | Text-only baseline |
|---|---|---|
| Accuracy | 96.3% | 97.2% |
| F1 (spam class) | 0.84 | 0.88 |
| ROC AUC | 0.993 | — |

The hybrid model trades a little raw accuracy for **explainability and
better recall on manipulation-style messages** the pure text model would
otherwise miss (see the SUSPICIOUS tier above) — a worthwhile trade for a
tool meant to help people understand *why* something looks like a scam,
not just flag it.

## Extending it
- Add more languages to the tactic lexicons.
- Swap `MultinomialNB` for `LogisticRegression` or a small transformer for
  higher raw accuracy.
- Tune `tactic_weight` and `suspicious_tactic_threshold` in
  `HybridSpamDetector` to trade precision vs. recall.
