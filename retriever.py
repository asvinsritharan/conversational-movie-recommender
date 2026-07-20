import numpy as np
import pandas as pd
import faiss
import bm25s
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5

class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL, device="mps")
        self.index = faiss.read_index("data/items.faiss")
        self.ids = np.load("data/item_ids.npy", allow_pickle=True)
        self.bm25 = bm25s.BM25.load("data/bm25")
        items = pd.read_parquet("data/items.parquet").set_index("item_id")
        self.title = items["title"].to_dict()
        self.text = items["text"].to_dict()

    def _rrf(self, dense_ids, sparse_ids, k=TOP_K, c=60):
        """Reciprocal Rank Fusion: combine two ranked id lists by rank, not score."""
        score = {}
        for rank, i in enumerate(dense_ids):
            score[i] = score.get(i, 0) + 1.0 / (c + rank)
        for rank, i in enumerate(sparse_ids):
            score[i] = score.get(i, 0) + 1.0 / (c + rank)
        return sorted(score, key=score.get, reverse=True)[:k]

    def search(self, query, k=TOP_K):
        qv = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, idx = self.index.search(qv, k * 4)
        dense_ids = [self.ids[j] for j in idx[0] if j != -1]

        res, _ = self.bm25.retrieve(bm25s.tokenize(query, stopwords="en"), k=k * 4)
        sparse_ids = [self.ids[j] for j in res[0]]

        fused = self._rrf(dense_ids, sparse_ids, k=k)
        return [{"item_id": i, "title": self.title.get(i, "?"),
                 "text": self.text.get(i, "")} for i in fused]


if __name__ == "__main__":
    r = HybridRetriever()
    for q in ["a tense psychological thriller with a twist ending",
              "feel-good animated movie for kids"]:
        print(f"\nquery: {q}")
        for rank, hit in enumerate(r.search(q), 1):
            print(f"  {rank}. {hit['title']}")