# build_index.py
import numpy as np
import pandas as pd
import faiss
import bm25s
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
BATCH = 64

items = pd.read_parquet("data/items.parquet")
items = items[items["text"].str.len() > 0].reset_index(drop=True)
corpus = items["text"].tolist()
ids = items["item_id"].to_numpy()
print(f"[index] embedding {len(corpus):,} items on the GPU (MPS)...")

model = SentenceTransformer(EMBED_MODEL, device="mps")
emb = model.encode(corpus, batch_size=BATCH, normalize_embeddings=True,
                   show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(EMBED_DIM)
index.add(emb)
faiss.write_index(index, "data/items.faiss")
np.save("data/item_ids.npy", ids)

retriever = bm25s.BM25()
retriever.index(bm25s.tokenize(corpus, stopwords="en"))
retriever.save("data/bm25")

print("[index] wrote data/items.faiss, data/item_ids.npy, data/bm25/")