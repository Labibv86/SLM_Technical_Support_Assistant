"""
Stack Overflow Python Q&A scraper + cleaner.

Run from project root:
    python data/prepare_data.py               # scrape + clean
    python data/prepare_data.py --clean-only  # clean existing raw JSON only

Outputs:
    data/raw/raw_qa.json       (raw scraped records)
    data/processed/train.jsonl (Alpaca format)
    data/processed/val.jsonl   (Alpaca format)
"""

import os
import sys
import json
import time
import random
import re
import argparse
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# Allow `python data/prepare_data.py` to import config.py from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    API_KEY, SITE, TAG, MIN_SCORE, PAGESIZE, TARGET_COUNT,
    SLEEP_BETWEEN_REQUESTS, RAW_DATA_PATH, TRAIN_PATH, VAL_PATH,
    MIN_ANSWER_CHARS, MIN_QUESTION_CHARS,
    INCLUDE_QUESTION_BODY_THRESHOLD, VAL_SPLIT_RATIO, SEED,
)

API_BASE = "https://api.stackexchange.com/2.3"


# ============================================================
# Part 1: Scraping
# ============================================================

def _get(url, params):
    """GET with backoff + error handling."""
    params["key"] = API_KEY
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "backoff" in data:
        print(f"  [backoff] sleeping {data['backoff']}s")
        time.sleep(data["backoff"])
    if "error_message" in data:
        raise RuntimeError(f"API error: {data['error_message']}")

    return data


def create_custom_filter():
    """Create a filter that includes question.body, answers list, and answer.body."""
    url = f"{API_BASE}/filters/create"
    params = {
        "include": "question.body;question.answers;answer.body",
        "key": API_KEY,
    }
    resp = requests.post(url, params=params, timeout=30)
    resp.raise_for_status()
    filter_id = resp.json()["items"][0]["filter"]
    print(f"Created custom filter: {filter_id}")
    return filter_id


def fetch_questions(page, filter_id):
    url = f"{API_BASE}/questions"
    params = {
        "site": SITE,
        "tagged": TAG,
        "sort": "votes",
        "order": "desc",
        "min": MIN_SCORE,
        "pagesize": PAGESIZE,
        "page": page,
        "filter": filter_id,
    }
    return _get(url, params)


def scrape():
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    filter_id = create_custom_filter()

    seen_ids = set()
    records = []
    page = 1

    while len(records) < TARGET_COUNT:
        print(f"Fetching page {page} ...")
        data = fetch_questions(page, filter_id)
        questions = data.get("items", [])
        if not questions:
            print("No more items returned.")
            break

        new_on_this_page = 0
        for q in questions:
            qid = q["question_id"]
            if qid in seen_ids:
                continue

            accepted = None
            for a in q.get("answers", []):
                if a.get("is_accepted"):
                    accepted = a
                    break
            if accepted is None:
                continue

            records.append({
                "question_id": qid,
                "title": q["title"],
                "question_body": q["body"],
                "answer_body": accepted["body"],
                "score": q["score"],
                "answer_score": accepted["score"],
                "tags": q["tags"],
                "link": q["link"],
            })
            seen_ids.add(qid)
            new_on_this_page += 1

        print(f"  +{new_on_this_page} new  |  total = {len(records)}")

        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        if len(records) >= TARGET_COUNT:
            break
        if not data.get("has_more", False):
            print("No more pages available.")
            break

        page += 1
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"\nScraping done. Saved {len(records)} records to {RAW_DATA_PATH}")


# ============================================================
# Part 2: HTML cleaning + Alpaca formatting
# ============================================================

def _html_to_markdown(node):
    """Convert a BeautifulSoup node to a Markdown-ish string."""
    def walk(n):
        out = []
        for child in n.children:
            if isinstance(child, NavigableString):
                out.append(str(child))
                continue
            if not isinstance(child, Tag):
                continue

            name = child.name
            if name == "pre":
                code_tag = child.find("code")
                code = (code_tag or child).get_text()
                lang = ""
                if code_tag and code_tag.get("class"):
                    for cls in code_tag["class"]:
                        if cls.startswith("language-"):
                            lang = cls.replace("language-", "")
                            break
                out.append(f"\n```{lang}\n{code.rstrip()}\n```\n")
            elif name == "code":
                out.append(f"`{child.get_text()}`")
            elif name == "a":
                text = child.get_text()
                href = child.get("href", "")
                out.append(f"[{text}]({href})" if href else text)
            elif name in ("b", "strong"):
                out.append(f"**{walk(child)}**")
            elif name in ("i", "em"):
                out.append(f"*{walk(child)}*")
            elif name in ("p", "div"):
                out.append(walk(child) + "\n\n")
            elif name in ("ul", "ol"):
                out.append(walk(child) + "\n")
            elif name == "li":
                out.append("- " + walk(child).strip() + "\n")
            elif name == "br":
                out.append("\n")
            elif name in ("h1", "h2", "h3", "h4"):
                out.append(f"\n{'#' * int(name[1])} {walk(child)}\n")
            elif name == "blockquote":
                inner = walk(child).strip().replace("\n", "\n> ")
                out.append(f"> {inner}\n")
            else:
                out.append(walk(child))
        return "".join(out)

    text = walk(node)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _decode_entities(s):
    return BeautifulSoup(s, "html.parser").get_text() if "&" in s else s


def clean_record(rec):
    """Return an Alpaca-format dict or None if too low quality."""
    title = _decode_entities(rec["title"]).strip()
    if len(title) < MIN_QUESTION_CHARS:
        return None

    q_md = _html_to_markdown(BeautifulSoup(rec["question_body"], "html.parser"))
    a_md = _html_to_markdown(BeautifulSoup(rec["answer_body"], "html.parser"))
    if len(a_md) < MIN_ANSWER_CHARS:
        return None

    return {
        "instruction": title,
        "input": q_md if len(title) < INCLUDE_QUESTION_BODY_THRESHOLD else "",
        "output": a_md,
        "meta": {
            "question_id": rec["question_id"],
            "score": rec["score"],
            "answer_score": rec["answer_score"],
            "tags": rec["tags"],
            "link": rec["link"],
        },
    }


def build_alpaca_dataset():
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} raw records")

    cleaned, seen_titles = [], set()
    for rec in raw:
        out = clean_record(rec)
        if out is None:
            continue
        key = out["instruction"].lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        cleaned.append(out)

    print(f"Kept {len(cleaned)} records after cleaning + dedup")

    random.seed(SEED)
    random.shuffle(cleaned)
    n_val = max(1, int(len(cleaned) * VAL_SPLIT_RATIO))
    val, train = cleaned[:n_val], cleaned[n_val:]

    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    for path, data in [(TRAIN_PATH, train), (VAL_PATH, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} rows -> {path}")

    if train:
        print("\n=== Sample cleaned record ===")
        s = train[0].copy()
        s["output"] = s["output"][:300] + "..."
        print(json.dumps(s, indent=2, ensure_ascii=False))


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-only", action="store_true",
                    help="Skip scraping; only clean existing raw data")
    args = ap.parse_args()

    if not args.clean_only:
        scrape()
    build_alpaca_dataset()