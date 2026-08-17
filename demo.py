import duckdb
import pandas as pd
import requests
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

sampler = make_sampler(temp=0.4)
logits_processors = make_logits_processors(repetition_penalty=1.15)

SERVICE = "http://127.0.0.1:8000/search"
SYSTEM = ("You write short movie and TV recommendations based strictly on provided reviews. "
          "You only know what the reviews tell you. In 1-3 sentences, explain why someone "
          "might like the title, using ONLY the reviews. If the reviews are thin, write less "
          "rather than inventing detail.")

print("loading explainer (MLX)...")
model, tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path="adapters/explainer_8L_r8")

# review store into memory once (DuckDB used only at startup, then closed)
rows = duckdb.sql("SELECT item_id, reviews FROM 'data/review_store.parquet'").fetchall()
store = {i: rv for i, rv in rows}
print(f"loaded reviews for {len(store):,} items")

def explain(title, reviews):
    if not reviews:
        return "(no reviews available to ground an explanation)"
    context = "\n".join(f"- {r[:400]}" for r in reviews[:6])
    user = f"Item: {title}\nReviews:\n{context}\n\nWhy might someone like this?"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt, max_tokens=160, sampler=sampler, logits_processors=logits_processors, verbose=False).strip()

print("\nReady. Type a query (or 'quit').")
while True:
    query = input("\n> ").strip()
    if query.lower() in ("quit", "exit", ""):
        break
    # ask the retrieval service (separate process) for item_ids
    resp = requests.post(SERVICE, json={"query": query, "k": 3}).json()
    for hit in resp["results"]:
        title = hit["title"]
        reviews = store.get(hit["item_id"], [])
        print(f"\n▶ {title}\n  {explain(title, reviews)}")