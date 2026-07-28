import json
import numpy as np
from sentence_transformers import SentenceTransformer

GROUNDING_FLOOR = 0.35        # below this, target likely drifts from its reviews
NEG_PHRASES = ("uneventful", "not worth", "may not excite", "disappoint", "boring",
               "not recommend", "waste", "fails to", "lacks", "mediocre", "forgettable")

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")

def audit(path):
    targets, contexts, rows = [], [], []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            user = ex["messages"][1]["content"]
            target = ex["messages"][-1]["content"]
            # the review context is the middle of the user turn
            ctx = user.split("Reviews:\n", 1)[-1].split("\n\nWhy might")[0]
            rows.append((user.split("\n")[0].replace("Item: ", ""), target))
            targets.append(target); contexts.append(ctx)

    # embed both sides, compare row-by-row (normalized -> dot product = cosine)
    te = model.encode(targets, normalize_embeddings=True, batch_size=64)
    ce = model.encode(contexts, normalize_embeddings=True, batch_size=64)
    grounding = (te * ce).sum(axis=1)

    neg = np.array([any(p in t.lower() for p in NEG_PHRASES) for t in targets])
    low = grounding < GROUNDING_FLOOR

    print(f"\n=== {path} : {len(targets)} examples ===")
    print(f"grounding score  mean={grounding.mean():.3f}  min={grounding.min():.3f}")
    print(f"low-grounding (<{GROUNDING_FLOOR}): {low.sum()} ({100*low.mean():.1f}%)")
    print(f"negative-sentiment phrasing: {neg.sum()} ({100*neg.mean():.1f}%)")
    flagged = low | neg
    print(f"TOTAL flagged: {flagged.sum()} ({100*flagged.mean():.1f}%)")

    print("\n--- 5 worst-grounded targets ---")
    for idx in np.argsort(grounding)[:5]:
        title, tgt = rows[idx]
        print(f"[{grounding[idx]:.2f}] {title}: {tgt[:150]}")
    return grounding, flagged

audit("data/finetune/train.json")
audit("data/finetune/valid.json")