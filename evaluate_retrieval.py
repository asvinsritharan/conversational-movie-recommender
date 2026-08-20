import json
import pandas as pd
from retriever import HybridRetriever

K = 3
QUERIES = [
    "a tense psychological thriller with a twist ending",
    "animated movie about kids and feelings",
    "a gritty war drama based on true events",
    "classic screwball comedy from the golden age of Hollywood",
    "slow-burn romance with beautiful cinematography",
    "campy low-budget horror that's fun to laugh at",
    "epic fantasy adventure with sword fights",
    "a crime thriller about street racing",
    "a heartwarming underdog sports story",
    "mind-bending sci-fi that makes you think",
    "British period drama with elaborate costumes",
    "a documentary about music and musicians",
    "dark crime thriller with morally gray characters",
    "lighthearted family comedy for movie night",
    "a tearjerker about love and loss",
]

r = HybridRetriever()
variants = {
    "dense":  lambda q: list(r.search_dense(q, K)),
    "sparse": lambda q: list(r.search_sparse(q, K)),
    "hybrid": lambda q: [h["item_id"] for h in r.search(q, K)],
}

# precision
precision_results = []
for q in QUERIES:
    precision_results.append({"query": q,
                              **{v: [str(i) for i in fn(q)] for v, fn in variants.items()}})
items = pd.read_parquet("data/items.parquet").set_index("item_id")
val = [json.loads(l) for l in open("data/finetune/valid.jsonl")][:20]
# recall
recall_results = []
for ex in val:
    title = ex["messages"][1]["content"].split("\n")[0].replace("Item: ", "")
    match = items[items["title"] == title]
    if match.empty:
        continue
    target_id = str(match.index[0])
    q = title.replace("VHS", "").replace("DVD", "").replace("[Import]", "").strip()
    retrieved = {v: [str(i) for i in fn(q)] for v, fn in variants.items()}
    recall_results.append({"query": q, "target_id": target_id, **retrieved})

json.dump({"precision": precision_results, "recall": recall_results},
          open("data/retrieval_evaluation.json", "w"))
print(f"done -> {len(precision_results)} precision, {len(recall_results)} recall")