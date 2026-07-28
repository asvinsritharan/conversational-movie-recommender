import json

NEG_PHRASES = ("uneventful", "not worth", "may not excite", "disappoint", "boring",
               "not recommend", "waste", "fails to", " lacks ", "mediocre", "forgettable")

def clean(path):
    kept = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            target = ex["messages"][-1]["content"].lower()
            if not any(p in target for p in NEG_PHRASES):
                kept.append(ex)
    with open(path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex) + "\n")
    print(f"{path}: kept {len(kept)}")

clean("data/finetune/train.json")
clean("data/finetune/valid.json")