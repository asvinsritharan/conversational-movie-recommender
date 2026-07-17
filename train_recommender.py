import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

K = 10
FACTORS = 256
ALPHA = 40.0

# --- load the slim interactions table (fast: Parquet, only the columns we need) ---
df = pd.read_parquet("data/interactions.parquet")

# --- map string IDs -> integer row/column indices for the matrix ---
# factorize returns (integer_codes, unique_values); the codes ARE our matrix indices.
df["uidx"], user_ids = pd.factorize(df["user_id"])
df["iidx"], item_ids = pd.factorize(df["item_id"])
n_users, n_items = len(user_ids), len(item_ids)
print(f"[reco] matrix: {n_users:,} users x {n_items:,} items")

# --- leave-one-out split: each user's most RECENT interaction is the test item ---
df = df.sort_values("ts")
test = df.groupby("uidx").tail(1)      # last row per user = most recent
train = df.drop(test.index)            # everything else is training
# (5-core guarantees every user keeps >=4 training interactions, so no user is empty)

# --- build the confidence matrix (users x items), values = 1 + alpha*rating ---
conf = (1.0 + ALPHA * (train["rating"].to_numpy() / 5.0)).astype(np.float32)
train_m = sp.csr_matrix((conf, (train["uidx"], train["iidx"])),
                        shape=(n_users, n_items), dtype=np.float32)

# --- train ALS ---
model = AlternatingLeastSquares(factors=FACTORS, regularization=0.05,
                                iterations=15, use_gpu=False)
model.fit(train_m)   # NOTE: implicit expects (users x items) orientation

# --- evaluate on a random sample of users (for speed; full set is ~305K) ---
rng = np.random.default_rng(0)
tu, ti = test["uidx"].to_numpy(), test["iidx"].to_numpy()
S = min(20_000, len(tu))
pick = rng.choice(len(tu), size=S, replace=False)
su, si = tu[pick], ti[pick]

# batch recommend: ids has shape (S, K); filter items the user already saw in training
ids, _ = model.recommend(su, train_m[su], N=K, filter_already_liked_items=True)

hits = (ids == si[:, None])            # where did the hidden item appear?
hit_any = hits.any(axis=1)
recall = hit_any.mean()
ranks = np.argmax(hits, axis=1)        # position of the hit (only meaningful if hit_any)
ndcg = np.where(hit_any, 1.0 / np.log2(ranks + 2), 0.0).mean()
print(f"[reco] Recall@{K}={recall:.4f}  NDCG@{K}={ndcg:.4f}  (on {S:,} users)")

# --- save the learned vectors for later phases ---
np.savez("data/als_model.npz",
         user_factors=model.user_factors, item_factors=model.item_factors)
print("[reco] saved taste vectors -> data/als_model.npz")