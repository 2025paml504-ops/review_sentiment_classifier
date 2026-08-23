from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request

NORMAL = [
    "I cannot sign in to my account",
    "Where is my delayed package",
    "Please refund my order",
    "The app crashes during checkout",
    "I was charged twice",
    "My tracking link is broken",
]
DRIFTED = [
    "yo this checkout is straight up cooked no cap",
    "the parcel status is giving ghost mode fr",
    "refund taking forever this is sus af",
    "app went full yeet after the update",
    "why did you double-dip my wallet fam",
    "login is bricked rn pls fix",
]


def send(url: str, text: str) -> dict:
    body = json.dumps({"text": text}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Send normal and drifted traffic to the API")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--normal", type=int, default=100)
    parser.add_argument("--drifted", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sent = 0
    try:
        for pool, count in ((NORMAL, args.normal), (DRIFTED, args.drifted)):
            for _ in range(count):
                send(args.url, rng.choice(pool))
                sent += 1
    except urllib.error.URLError as exc:
        raise SystemExit(f"API request failed after {sent} events: {exc}") from exc
    print(f"Sent {sent} events")


if __name__ == "__main__":
    main()
