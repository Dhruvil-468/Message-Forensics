"""
Interactive demo for the Hybrid Spam Detector.

Usage:
    python3 demo.py
    (then type/paste messages, or press Enter on an empty line to quit)
"""

from spam_detector import HybridSpamDetector, load_data


def main():
    print("Training hybrid spam detector on SMS Spam Collection dataset...")
    df = load_data()
    model = HybridSpamDetector(tactic_weight=0.35)
    model.fit(df["text"], df["y"])
    print("Ready. Type a message to classify it (empty line to quit).\n")

    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        print()
        print(model.explain(text))
        print()


if __name__ == "__main__":
    main()
