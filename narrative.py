"""
Narrative explanation generator for the Hybrid Spam Detector.

Turns a structured analysis (verdict, scores, detected tactics, matched
text spans) into plain-English paragraphs — the "why", written out in
sentences, not just numbers. Fully deterministic and rule-based: no LLM
call, so it's fast, free, and reproducible for every report.
"""

TACTIC_PHRASES = {
    "urgency": "creates a false sense of urgency, pushing you to act before you've had time to think",
    "scarcity": "leans on manufactured scarcity — implying a limited, one-time, or expiring opportunity",
    "money_bait": "dangles money, a prize, or an unearned reward to get your attention",
    "false_authority": "impersonates a trusted authority, such as a bank, government body, or well-known brand",
    "pressure_to_act": "pressures you toward an immediate action, like clicking a link or calling a number right away",
    "excess_caps": "leans on excessive CAPITAL LETTERS, a common attention-grabbing trick in spam",
    "excess_punct": "uses exaggerated punctuation (multiple '!' or '?'), typical of low-effort spam and scam blasts",
    "has_url": "includes a link, which is worth checking carefully before clicking — a common phishing vector",
    "has_phone": "includes a phone number, which scam messages often use to move the conversation somewhere less traceable",
}

RECOMMENDATIONS = {
    "SPAM": (
        "Recommendation: treat this message as spam. Don't click any links, call any numbers, or reply "
        "to it. If it's in your inbox, report or delete it."
    ),
    "SUSPICIOUS": (
        "Recommendation: proceed with caution. Don't act on this message directly — if it claims to be "
        "from an organization (a bank, a delivery service, etc.), verify independently by contacting them "
        "through a phone number or website you already know is real, not one provided in the message."
    ),
    "HAM": (
        "Recommendation: no action needed. This message doesn't show the manipulation patterns associated "
        "with spam or scams."
    ),
}


def _confidence_phrase(pct):
    if pct >= 90:
        return "very high confidence"
    if pct >= 70:
        return "high confidence"
    if pct >= 50:
        return "moderate confidence"
    if pct >= 25:
        return "low confidence"
    return "very low confidence"


def _example_words(matches, tactic, limit=3):
    seen = []
    for m in matches:
        if m["tactic"] == tactic and m.get("text"):
            w = m["text"]
            if w.lower() not in [s.lower() for s in seen]:
                seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def generate_narrative(analysis: dict) -> list:
    """
    Build a list of plain-English paragraphs explaining an analysis.
    `analysis` is expected to have: text, verdict, confidence,
    text_model_score, tactic_score, tactics (dict), matches (list of
    {start,end,tactic,label} — 'text' will be derived here if missing).
    """
    text = analysis["text"]
    verdict = analysis["verdict"]
    confidence = analysis["confidence"]
    text_model_score = analysis["text_model_score"]
    tactic_score = analysis["tactic_score"]
    tactics = analysis.get("tactics", {})
    matches = list(analysis.get("matches", []))

    # ensure matches carry their literal text (for quoting examples)
    for m in matches:
        if "text" not in m:
            m["text"] = text[m["start"]:m["end"]]

    conf_pct = round(confidence * 100)
    text_pct = round(text_model_score * 100)
    tactic_pct = round(tactic_score * 100)

    paragraphs = []

    # --- Paragraph 1: headline verdict sentence ---
    if verdict == "SPAM":
        paragraphs.append(
            f"This message was flagged as SPAM with {_confidence_phrase(conf_pct)} ({conf_pct}%). "
            f"Both the language-pattern model and the manipulation-tactic scan agree this message is "
            f"very unlikely to be a genuine, wanted message."
        )
    elif verdict == "SUSPICIOUS":
        paragraphs.append(
            f"This message was marked SUSPICIOUS. On its own, the language-pattern model wasn't fully "
            f"convinced — it estimated only a {text_pct}% chance of being spam, which is below the "
            f"threshold for an outright SPAM verdict. But one or more manipulation tactics fired strongly "
            f"enough that the message is still worth a second look rather than being cleared automatically."
        )
    else:
        paragraphs.append(
            f"This message was cleared as ordinary correspondence (HAM), with only a {conf_pct}% overall "
            f"spam score — {_confidence_phrase(100 - conf_pct)} that it is NOT spam. It reads like a "
            f"normal, human message rather than a mass-sent or manipulative one."
        )

    # --- Paragraph 2: what the word-pattern model saw ---
    if text_pct >= 70:
        model_sentence = (
            f"The trained language model, which learned word and phrase patterns from thousands of real "
            f"spam and non-spam messages, rated the wording of this message as {text_pct}% likely to be "
            f"spam — a strong signal on its own."
        )
    elif text_pct >= 30:
        model_sentence = (
            f"The trained language model rated the wording of this message as {text_pct}% likely to be "
            f"spam — a mixed signal, neither clearly normal nor clearly spam-like by vocabulary alone."
        )
    else:
        model_sentence = (
            f"The trained language model rated the wording of this message as only {text_pct}% likely to "
            f"be spam — its vocabulary and phrasing read as ordinary."
        )
    paragraphs.append(model_sentence)

    # --- Paragraph 3: the tactics, explained in prose ---
    # Only narrate tactics with meaningful strength — trivial noise (e.g. one
    # stray question mark) still shows in the full stats table but shouldn't
    # get inflated into a sentence of its own. Structural signals (caps,
    # punctuation) are noisier at low levels than lexicon hits, so they need
    # a higher bar before they're worth a sentence.
    STRUCTURAL_TACTICS = {"excess_caps", "excess_punct"}
    def _worth_narrating(tactic, score):
        threshold = 0.4 if tactic in STRUCTURAL_TACTICS else 0.15
        return score >= threshold

    active = [(k, v) for k, v in tactics.items() if k in TACTIC_PHRASES and _worth_narrating(k, v)]
    active.sort(key=lambda kv: -kv[1])

    if active:
        clauses = []
        for tactic, score in active:
            phrase = TACTIC_PHRASES[tactic]
            examples = _example_words(matches, tactic)
            if examples:
                quoted = ", ".join(f'\u201c{w}\u201d' for w in examples)
                clauses.append(f"{phrase} (for example: {quoted})")
            else:
                clauses.append(phrase)

        if len(clauses) == 1:
            tactic_sentence = f"Beyond the wording alone, this message {clauses[0]}."
        else:
            tactic_sentence = (
                "Beyond the wording alone, this message " + "; it also ".join(clauses[:-1])
                + f"; and it {clauses[-1]}."
            )
        paragraphs.append(tactic_sentence)
    else:
        paragraphs.append(
            "No significant manipulation tactics — urgency, false authority, money bait, pressure to act, "
            "or similar — were detected in the text."
        )

    # --- Paragraph 4: explain the SUSPICIOUS override logic specifically, when relevant ---
    if verdict == "SUSPICIOUS" and active:
        top_tactic, top_score = active[0]
        top_label = TACTIC_PHRASES.get(top_tactic, top_tactic)
        paragraphs.append(
            f"This is exactly the kind of message the tactic layer exists to catch: the strongest single "
            f"signal here — that it {top_label} — reached {round(top_score*100)}% strength, which is high "
            f"enough by itself to raise a flag even though the overall word-pattern score stayed under the "
            f"50% SPAM threshold. Phishing messages often don't 'read' like typical bulk spam, but they "
            f"still rely on the same handful of pressure tactics, which is why they're worth flagging "
            f"separately."
        )

    # --- Paragraph 5: recommendation ---
    paragraphs.append(RECOMMENDATIONS.get(verdict, RECOMMENDATIONS["HAM"]))

    return paragraphs
