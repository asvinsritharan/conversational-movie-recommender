import json, random
import duckdb
import pandas as pd

CATEGORY = "Movies_and_TV"
N_ITEMS = 2000
REVIEWS_PER_ITEM = 6
VALID_FRAC = 0.1
random.seed(0)

SYSTEM = ("You are a movie/tv show expert that has watched every single movie and tv show in the planet with full understanding."
          "In 2-3 sentences, explain why someone might like the "
          "movie or tv show, using ONLY the reviews provided. Do not invent details.")

inter = pd.read_parquet("data/interactions.parquet")
top_items = inter["item_id"].value_counts().head(N_ITEMS).index.tolist()
titles = pd.read_parquet("data/items.parquet").set_index("item_id")["title"].to_dict()

con = duckdb.connect()
con.execute("PRAGMA memory_limit='4GB'")
item_list = ",".join("'" + i.replace("'", "''") + "'" for i in top_items)
rows = con.execute(f"""
    SELECT asin AS item_id, reviewText
    FROM read_json_auto('data/raw/{CATEGORY}.json',
                        format='newline_delimited', ignore_errors=true)
    WHERE asin IN ({item_list})
      AND reviewText IS NOT NULL AND length(reviewText) BETWEEN 100 AND 1200
    QUALIFY row_number() OVER (PARTITION BY asin ORDER BY length(reviewText) DESC)
            <= {REVIEWS_PER_ITEM}
""").fetchall()

reviews = {}
for item_id, text in rows:
    reviews.setdefault(item_id, []).append(text.strip().replace("\n", " "))
print(f"[ft] pulled reviews for {len(reviews):,} items")


# Openers that signal throat-clearing rather than a claim about the film.
JUNK_STARTS = ("watched", "bought", "purchased", "ordered", "received", "arrived",
               "i'll keep", "i will keep", "let me", "first off", "full disclosure",
               "i rented", "got this", "picked this")

def _first_sentence(review):
    """Return a usable opening sentence about the film, or None."""
    s = review.split(". ")[0].strip().strip(".").strip()
    if len(s) < 30 or len(s) > 220:          # too thin to be a claim, or too rambling
        return None
    low = s.lower()
    if low.startswith(JUNK_STARTS):          # logistics, not opinion
        return None
    if not any(c.isalpha() for c in s[:3]):  # starts with punctuation/numbers
        return None
    return s[0].lower() + s[1:]


def mechanical_target(title, revs):
    """Grounded-by-construction target, built only from substantive sentences."""
    sents = []
    for r in sorted(revs, key=len, reverse=True):
        s = _first_sentence(r)
        if s:
            sents.append(s)
        if len(sents) == 2:
            break
    if not sents:
        return None
    out = f"Reviewers highlight that {sents[0]}."
    if len(sents) > 1:
        second = f" Several also note that {sents[1]}."
        if len(out) + len(second) <= 400:
            out += second
    return out


examples = []
for item_id, revs in reviews.items():
    if len(revs) < 2:
        continue
    title = titles.get(item_id, "this movie")
    target = mechanical_target(title, revs)
    if target is None:          # no usable sentences — skip this item
        continue
    context = "\n".join(f"- {r[:400]}" for r in revs)
    user = f"Item: {title}\nReviews:\n{context}\n\nWhy might someone like this?"
    examples.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]})

random.shuffle(examples)
n_val = int(len(examples) * VALID_FRAC)

import os
os.makedirs("data/finetune", exist_ok=True)
for name, rows_ in [("valid", examples[:n_val]), ("train", examples[n_val:])]:
    with open(f"data/finetune/{name}.json", "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")

print(f"[ft] wrote {len(examples)-n_val} train / {n_val} valid -> data/finetune/")