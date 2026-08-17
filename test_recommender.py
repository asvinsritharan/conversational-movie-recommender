from mlx_lm import load, generate

SYSTEM = ("You are a movie/tv show expert that has watched every single movie and tv show in the planet with full understanding."
          "In 2-3 sentences, explain why someone might like the "
          "movie or tv show, using ONLY the reviews provided. Do not invent details.")

# a sample item + reviews (mimics what the retriever will supply in 2d)
USER = """Item: The Shawshank Redemption
Reviews:
- One of the most powerful films about hope and friendship I have ever seen. The ending left me in tears.
- Morgan Freeman and Tim Robbins are extraordinary. A slow burn that rewards your patience completely.
- I have watched this a dozen times and it never loses its emotional punch. The prison setting is grim but the story is uplifting.

Why might someone like this?"""

def run(adapter):
    model, tok = load("mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path=adapter)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tok, prompt, max_tokens=160, verbose=False).strip()

print("=== BASE MODEL (no adapter) ===")
print(run(None), "\n")
print("=== FINE-TUNED (your adapter) ===")
print(run("adapters/explainer_8L_r8"))