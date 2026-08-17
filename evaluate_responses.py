import json, re
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

N = 50
BASE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTER = "adapters/explainer_8L_r8"
JUDGE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

SYSTEM = ("You write short movie and TV recommendations based strictly on provided reviews. "
          "You only know what the reviews tell you. In 1-3 sentences, explain why someone "
          "might like the title, using ONLY the reviews. If the reviews are thin, write less "
          "rather than inventing detail.")

JUDGE_PROMPT = """You are a strict fact-checker. Below are REVIEWS of a movie and an
EXPLANATION of why someone might like it.

List each distinct factual claim in the explanation on its own line, and for each write
SUPPORTED (if the reviews state or clearly imply it) or UNSUPPORTED (if not in the reviews).
Reason about each claim before deciding.

Format each line exactly as:
- <claim> => SUPPORTED
- <claim> => UNSUPPORTED

REVIEWS:
{context}

EXPLANATION:
{explanation}"""

val = [json.loads(l) for l in open("data/finetune/valid.jsonl")][:N]

def context_of(ex):
    u = ex["messages"][1]["content"]
    return u.split("Reviews:\n", 1)[-1].split("\n\nWhy might")[0]

print("generating explanations (serving settings: temp=0.4)...")
model, tok = load(BASE, adapter_path=ADAPTER)
gen_sampler = make_sampler(temp=0.4)
gen_penalty = make_logits_processors(repetition_penalty=1.15)
outputs = []
for ex in val:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["messages"][1]["content"]}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    outputs.append(generate(model, tok, p, max_tokens=160,
                            sampler=gen_sampler, logits_processors=gen_penalty,
                            verbose=False).strip())
del model  

print("loading judge (8B)...")
jmodel, jtok = load(JUDGE_MODEL)
jsampler = make_sampler(temp=0.0)

def judge(context, explanation):
    p = JUDGE_PROMPT.format(context=context[:2000], explanation=explanation)
    msgs = [{"role": "user", "content": p}]
    text = jtok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    out = generate(jmodel, jtok, text, max_tokens=400, sampler=jsampler, verbose=False)
    sup = len(re.findall(r"=>\s*SUPPORTED", out, re.I))
    uns = len(re.findall(r"=>\s*UNSUPPORTED", out, re.I))
    return sup / (sup + uns) if (sup + uns) > 0 else None

contexts = [context_of(ex) for ex in val]
scores, per_item = [], []
for i, (ctx, expl) in enumerate(zip(contexts, outputs)):
    s = judge(ctx, expl)
    if s is not None:
        scores.append(s)
        per_item.append((s, expl))
    if (i + 1) % 10 == 0:
        print(f"  judged {i+1}/{len(outputs)}")

scores = np.array(scores)
print(f"\n=== truthfulness (fraction of claims supported) ===")
print(f"mean={scores.mean():.3f}  median={np.median(scores):.3f}  n={len(scores)}")
print(f"fully grounded (all claims supported): {(scores==1.0).mean()*100:.0f}%")
print(f"weak (<50% supported): {(scores<0.5).mean()*100:.0f}%")

print("\n--- 3 lowest-scoring explanations (inspect for real hallucination) ---")
for s, expl in sorted(per_item)[:3]:
    print(f"[{s:.2f}] {expl[:200]}")