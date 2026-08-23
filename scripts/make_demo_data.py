from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "account": [
        "I cannot log into my account",
        "Please reset my password",
        "My account is locked",
        "I need to change my email address",
        "How do I close my account",
        "Two factor authentication is not working",
        "Someone accessed my profile",
        "I forgot my username",
        "Please update my phone number",
        "My verification code never arrived",
        "I want to reactivate my account",
        "The sign in page keeps rejecting me",
    ],
    "billing": [
        "I was charged twice for my subscription",
        "My invoice has the wrong amount",
        "Why did my monthly price increase",
        "Please update my payment method",
        "My card was declined",
        "I do not recognize this charge",
        "Where can I download my invoice",
        "The coupon was not applied",
        "My billing address is incorrect",
        "The payment is still pending",
        "I need a receipt for the purchase",
        "Why was I charged after cancellation",
    ],
    "delivery": [
        "Where is my package",
        "My order is late",
        "The tracking link does not work",
        "The parcel arrived damaged",
        "My order was delivered to the wrong address",
        "Can I change the delivery date",
        "Only part of my order arrived",
        "The courier says delivery failed",
        "I need express shipping",
        "My package is missing",
        "When will the item be dispatched",
        "The tracking status has not changed",
    ],
    "refund": [
        "I want a refund",
        "My refund has not arrived",
        "How do I return this item",
        "Please cancel and refund my order",
        "The product does not match the description",
        "I received the wrong item and want my money back",
        "What is the return policy",
        "My return was rejected",
        "I changed my mind about the purchase",
        "Send me a return label",
        "The refund amount is incorrect",
        "Can I exchange instead of returning",
    ],
    "technical": [
        "The app crashes when I open it",
        "The website shows an error",
        "The page will not load",
        "Notifications stopped working",
        "The latest update broke the app",
        "I cannot upload a file",
        "The screen is blank",
        "Search returns no results",
        "The mobile app is very slow",
        "I found a bug in checkout",
        "The download keeps failing",
        "The button does nothing when clicked",
    ],
}
PREFIXES = ["", "Hello, ", "Hi support, ", "Please help: ", "Urgent: "]
SUFFIXES = ["", ".", " please", " as soon as possible", " Can you help?"]


def build_rows(seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for label, messages in EXAMPLES.items():
        for index, message in enumerate(messages):
            rows.append({"text": message, "label": label})
            rows.append(
                {
                    "text": (
                        f"{rng.choice(PREFIXES)}{message.lower()}"
                        f"{rng.choice(SUFFIXES)} #{index + 1}"
                    ),
                    "label": label,
                }
            )
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic synthetic support tickets")
    parser.add_argument("--output", default="data/raw/support_tickets.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.output)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.seed)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
