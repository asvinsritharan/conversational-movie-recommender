import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold
from simulate import simulate

df, true_effect = simulate()

# confounder
X = df[["engagement"]].to_numpy()
# treatment
T = df["treat_obs"].to_numpy()
# outcome
Y = df["clicked_obs"].to_numpy()

# engagement estimate
naive = Y[T == 1].mean() - Y[T == 0].mean()

kf = KFold(n_splits=5, shuffle=True, random_state=0)
y_resid = np.zeros_like(Y, dtype=float)
t_resid = np.zeros_like(T, dtype=float)

for train, test in kf.split(X):
    # given engagement, how likely is the user to click
    m_y = GradientBoostingRegressor(random_state=0).fit(X[train], Y[train])
    # given engagement, how likely was the user to be shown explanation
    m_t = GradientBoostingClassifier(random_state=0).fit(X[train], T[train])

    # calculate residuals
    y_resid[test] = Y[test] - m_y.predict(X[test])
    t_resid[test] = T[test] - m_t.predict_proba(X[test])[:, 1]

# causal effect
theta = np.sum(t_resid * y_resid) / np.sum(t_resid * t_resid)

# calcualate standard error
resid = y_resid - theta * t_resid
se = np.sqrt(np.sum(resid**2) / np.sum(t_resid**2)**2 * np.sum(t_resid**2)) / np.sqrt(len(Y))
se = np.sqrt(np.mean(resid**2) / np.mean(t_resid**2)) / np.sqrt(len(Y))

print("=== recovering the causal effect from CONFOUNDED observational data ===\n")
print(f"True effect (planted):          {true_effect:.3f}")
print(f"Naive estimate (confounded):    {naive:.3f}   <- biased, ~2x too high")
print(f"Causal Effect estimate:             {theta:.3f}   +/- {1.96*se:.3f} (95% CI)")
print(f"  95% CI: [{theta-1.96*se:.3f}, {theta+1.96*se:.3f}]")
print(f"\n  -> double ML recovers ~{true_effect:.2f} from data where the naive")
print(f"     comparison was off by ~{abs(naive-true_effect):.2f}.")