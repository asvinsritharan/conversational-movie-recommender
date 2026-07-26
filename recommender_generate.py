import json, time
import duckdb
import pandas as pd
import random, os
from mlx_lm import load, generate

CATEGORY = "Movies_and_TV"
N_ITEMS = 2000
REVIEWS_PER_ITEM = 6
MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

SYSTEM = ("You are a movie/tv show expert that has watched every single movie and tv show in the planet with full understanding."
          "In 2-3 sentences, explain why someone might like the "
          "movie or tv show, using ONLY the reviews provided. Do not invent details.")

REVIEWER_PROMPT = """
You are a reviewer that writes small blurbs of movie and tv show recommendations.

Here are real reviews of "{title}". Write 2-3 sentences explaining why somebody might enjoy
this movie. Base the sentences solely on what the reviews say about the movie/tv show.

YOU MUST FOCUS ON THE MOVIE/TV SHOW. Look at the story, acting, tone, themes, what viewers enjoyed about the movie/tv show.
YOU WILL IGNORE ANYTHING THAT DOES NOT RELATE TO THE MOVIE ITSELF. That is, if the review mentions anything about shipping, DVD/Blu-ray quality, the reviewer's personal life, where they bought or rented it, or their collection. IGNORE THAT PART OF THE REVIEW.
DO NOT MENTION "Reviewers" or "Reviews". This is to be written as a DIRECT recommendation
YOU UNDER NO CIRCUMSTANCES ARE ALLOWS TO INVENT FACTS. ALL FACTS MUST BE TRUE FACTS FROM THE REVIEWS.

Here are the reviews: {context}
"""

inter = pd.read_parquet("data/interactions.parquet")
top_items = inter["item_id"].value_counts().head(N_ITEMS).index.tolist()
titles = pd.read_parquet("data/items.parquet").set_index("item_id")["title"].to_dict()

con = duckdb.connect(); con.execute("PRAGMA memory_limit='4GB'")
item_list = ",".join("'" + i.replace("'", "''") + "'" for i in top_items)
rows = con.execute(f"""
    SELECT asin AS item_id, reviewText
    FROM read_json_auto('data/raw/{CATEGORY}.json',
                        format='newline_delimited', ignore_errors=true)
    WHERE asin IN ({item_list})
      AND reviewText IS NOT NULL AND length(reviewText) BETWEEN 100 AND 1200
      AND CAST(overall AS DOUBLE) >= 4.0
    QUALIFY row_number() OVER (PARTITION BY asin ORDER BY length(reviewText) DESC)
            <= {REVIEWS_PER_ITEM}
""").fetchall()

reviews = {}
for item_id, text in rows:
    reviews.setdefault(item_id, []).append(text.strip().replace("\n", " "))
items = [(i, r) for i, r in reviews.items() if len(r) >= 2]
print(f"[teacher] generating targets for {len(items):,} items")

model, tok = load(MODEL)

def make_target(title, revs):
    context = "\n".join(f"- {r[:400]}" for r in revs)
    prompt = REVIEWER_PROMPT.format(title=title, context=context)
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(model, tok, text, max_tokens=160, verbose=False)
    return out.strip()

examples, t0 = [], time.time()
for n, (item_id, revs) in enumerate(items, 1):
    title = titles.get(item_id, "this movie")
    target = make_target(title, revs)
    if len(target) < 30:
        continue
    context = "\n".join(f"- {r[:400]}" for r in revs)
    user = f"Item: {title}\nReviews:\n{context}\n\nWhy might someone like this?"
    examples.append({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]})
    if n % 50 == 0:
        rate = n / (time.time() - t0)
        eta = (len(items) - n) / rate / 60
        print(f"[teacher] {n}/{len(items)}  ~{rate:.1f}/s  ETA {eta:.0f} min")
        with open("data/finetune/train_recommender.partial.json", "w") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")

random.seed(0); random.shuffle(examples)
n_val = int(len(examples) * 0.1)
os.makedirs("data/finetune", exist_ok=True)
for name, rows_ in [("valid", examples[:n_val]), ("train", examples[n_val:])]:
    with open(f"data/finetune/{name}.json", "w") as f:
        for e in rows_:
            f.write(json.dumps(e) + "\n")
print(f"[recommender] wrote {len(examples)-n_val} train / {n_val} valid")