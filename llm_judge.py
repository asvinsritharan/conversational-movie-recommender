import json, re, os
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

N = 50
BASE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
JUDGE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
ADAPTERS = {"base": None, "finetuned": "adapters/explainer"}

SYSTEM = ("You are a movie/tv show expert that has watched every single movie and tv show in the planet with full understanding."
          "In 2-3 sentences, explain why someone might like the "
          "movie or tv show, using ONLY the reviews provided. Do not invent details.")

JUDGE_PROMPT = """You are a strict fact-checker. Below are REVIEWS of a movie and an
EXPLANATION of why someone might like it.

List each distinct factual claim in the explanation on its own line, and for each one
write SUPPORTED (if the reviews state or clearly imply it) or UNSUPPORTED (if it is not
in the reviews). Reason about each claim before deciding.

Format each line exactly as:
- <claim text> => SUPPORTED
- <claim text> => UNSUPPORTED

REVIEWS:
{context}

EXPLANATION:
{explanation}"""

val = [json.loads(l) for l in open("data/finetune/valid.jsonl")][:N]

def context_of(ex):
    u = ex["messages"][1]["content"]
    return u.split("Reviews:\n", 1)[-1].split("\n\nWhy might")[0]

def gen_all(adapter):
    model, tok = load(BASE, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)          # deterministic generation too
    outs = []
    for ex in val:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": ex["messages"][1]["content"]}]
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        outs.append(generate(model, tok, p, max_tokens=160, sampler=sampler, verbose=False).strip())
    return outs

contexts = [context_of(ex) for ex in val]

# --- generate (with caching so reruns skip straight to judging) ---
CACHE = "data/judge_outputs.json"
if os.path.exists(CACHE):
    outputs = json.load(open(CACHE))
    print("loaded cached generations")
else:
    outputs = {}
    for name, adapter in ADAPTERS.items():
        print(f"generating {name}...")
        outputs[name] = gen_all(adapter)
    json.dump(outputs, open(CACHE, "w"))
    print(f"cached generations -> {CACHE}")

# --- judge ---
print("loading judge (8B)...")
jmodel, jtok = load(JUDGE_MODEL)
jsampler = make_sampler(temp=0.0)

def judge(context, explanation):
    p = JUDGE_PROMPT.format(context=context[:2000], explanation=explanation)
    msgs = [{"role": "user", "content": p}]
    text = jtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # give it room to reason through every claim
    out = generate(jmodel, jtok, text, max_tokens=400, sampler=jsampler, verbose=False)
    supported = len(re.findall(r"=>\s*SUPPORTED", out, re.I))
    unsupported = len(re.findall(r"=>\s*UNSUPPORTED", out, re.I))
    total = supported + unsupported
    return supported / total if total > 0 else None

for name in ADAPTERS:
    scores = np.array([s for ctx, expl in zip(contexts, outputs[name])
                       if (s := judge(ctx, expl)) is not None])
    print(f"\n=== {name} ===")
    print(f"faithfulness (frac supported): mean={scores.mean():.3f}  n={len(scores)}")
    print(f"fully-grounded (100% supported): {(scores==1.0).mean()*100:.0f}% of items")
    print(f"hallucination rate (any unsupported): {(scores<1.0).mean()*100:.0f}% of items")