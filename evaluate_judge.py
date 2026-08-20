import json
import numpy as np
import pandas as pd
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

data = json.load(open("data/retrieval_evaluation.json"))
items = pd.read_parquet("data/items.parquet").set_index("item_id")
titles, texts = items["title"].to_dict(), items["text"].to_dict()

jmodel, jtok = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
jsampler = make_sampler(temp=0.0)

def is_relevant(query, item_id):
    desc = (str(titles.get(item_id, "")) + ". " + str(texts.get(item_id, "")))[:600]
    prompt = (f'Query: "{query}"\n\nMovie: "{desc}"\n\n'
              f"Is this movie relevant to the query? Answer only YES or NO.")
    out = generate(jmodel, jtok,
        jtok.apply_chat_template([{"role":"user","content":prompt}],
            add_generation_prompt=True, tokenize=False),
        max_tokens=5, sampler=jsampler, verbose=False)
    return "yes" in out.strip().lower()[:5]

variants = ["dense", "sparse", "hybrid"]

# precision
prec = {v: [] for v in variants}
for row in data["precision"]:
    for v in variants:
        rel = [is_relevant(row["query"], i) for i in row[v]]
        prec[v].append(np.mean(rel) if rel else 0.0)

# recall
rec = {v: [] for v in variants}
for row in data["recall"]:
    for v in variants:
        rec[v].append(1.0 if row["target_id"] in row[v] else 0.0)

print("\n=== PRECISION@3 (relevance-judged) ===")
for v in variants:
    print(f"  {v:7s} = {np.mean(prec[v]):.3f}")
print(f"\n=== RECALL@3 (n={len(data['recall'])}) ===")
for v in variants:
    print(f"  {v:7s} = {np.mean(rec[v]):.3f}")