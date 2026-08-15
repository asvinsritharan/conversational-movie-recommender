import json
import numpy as np
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer

N = 50                                    # held-out items to test
BASE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER = "adapters/explainer"

SYSTEM = ("You are a movie/tv show expert that has watched every single movie and tv show in the planet with full understanding."
          "In 2-3 sentences, explain why someone might like the "
          "movie or tv show, using ONLY the reviews provided. Do not invent details.")

# --- load the held-out validation examples ---
val = [json.loads(l) for l in open("data/finetune/valid.jsonl")][:N]
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")

def context_of(ex):
    user = ex["messages"][1]["content"]
    return user.split("Reviews:\n", 1)[-1].split("\n\nWhy might")[0]

def grounding(texts, contexts):
    te = embedder.encode(texts, normalize_embeddings=True, batch_size=64)
    ce = embedder.encode(contexts, normalize_embeddings=True, batch_size=64)
    return (te * ce).sum(axis=1)

def gen_all(adapter):
    model, tok = load(BASE, adapter_path=adapter)
    outs = []
    for i, ex in enumerate(val):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ex["messages"][1]["content"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        outs.append(generate(model, tok, prompt, max_tokens=160, verbose=False).strip())
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(val)}")
    return outs

contexts = [context_of(ex) for ex in val]
teacher = [ex["messages"][-1]["content"] for ex in val]   # the targets we trained toward

print("generating base model outputs...")
base_out = gen_all(None)
print("generating fine-tuned outputs...")
ft_out = gen_all(ADAPTER)

g_teacher = grounding(teacher, contexts)
g_base = grounding(base_out, contexts)
g_ft = grounding(ft_out, contexts)

print("\n=== grounding scores (higher = more faithful to reviews) ===")
print(f"teacher targets (ceiling): mean={g_teacher.mean():.3f}")
print(f"base model (no adapter):   mean={g_base.mean():.3f}")
print(f"fine-tuned (your adapter): mean={g_ft.mean():.3f}")
print(f"\nfine-tuned vs base delta:  {g_ft.mean() - g_base.mean():+.3f}")
print(f"fine-tuned beats base on {(g_ft > g_base).sum()}/{len(val)} items")

# also report length — the fabrication failure mode came with wordiness
print(f"\navg words  base={np.mean([len(o.split()) for o in base_out]):.0f}  "
      f"fine-tuned={np.mean([len(o.split()) for o in ft_out]):.0f}  "
      f"teacher={np.mean([len(t.split()) for t in teacher]):.0f}")