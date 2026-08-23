import numpy as np
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from simulate import simulate

df, true_effect = simulate()

treated = df[df.treat_rct == 1].clicked_rct
control = df[df.treat_rct == 0].clicked_rct

n_t, n_c = len(treated), len(control)
clicks_t, clicks_c = treated.sum(), control.sum()
rate_t, rate_c = treated.mean(), control.mean()

print("=== A/B TEST: does showing explanations increase clicks? ===\n")
print(f"Treatment (explanation shown): {clicks_t}/{n_t} clicked = {rate_t:.3f}")
print(f"Control   (no explanation):    {clicks_c}/{n_c} clicked = {rate_c:.3f}")
print(f"Observed lift: {rate_t - rate_c:+.3f}   (true effect: {true_effect:.3f})\n")

# two proportion z test
stat, pval = proportions_ztest([clicks_t, clicks_c], [n_t, n_c])
print(f"Two-proportion z-test: z={stat:.2f}, p={pval:.4g}")
verdict = "REJECT H0 — effect is statistically significant" if pval < 0.05 else \
          "fail to reject H0 — not significant"
print(f"  -> {verdict}\n")

# CI
lo, hi = confint_proportions_2indep(clicks_t, n_t, clicks_c, n_c, method="wald")
print(f"95% CI for the lift: [{lo:.3f}, {hi:.3f}]")
print(f"  -> the true effect {true_effect:.3f} {'falls inside' if lo<=true_effect<=hi else 'falls OUTSIDE'} the interval\n")

pooled = (clicks_t + clicks_c) / (n_t + n_c)
effect_size = (rate_t - rate_c) / np.sqrt(pooled * (1 - pooled))
power = NormalIndPower().power(effect_size=effect_size, nobs1=n_t, alpha=0.05, ratio=n_c/n_t)
print(f"Statistical power at this sample size: {power:.2f}")
print("  -> power > 0.8 means the test was well-powered to detect this effect.")