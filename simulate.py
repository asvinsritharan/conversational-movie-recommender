import numpy as np
import pandas as pd

def simulate(n=5000, seed=0):
    """Generate users with a KNOWN causal effect of explanations on clicking.

    engagement : the confounder (some users are more active)
    treatment  : shown an explanation? (1/0)
    clicked    : the outcome

    TRUE_EFFECT is the causal bump in click probability from an explanation.
    We plant it here so we can later check whether our estimators recover it.
    """
    rng = np.random.default_rng(seed)
    TRUE_EFFECT = 0.10

    engagement = rng.uniform(0, 1, n)

    p_treat = 0.2 + 0.6 * engagement
    treat_obs = rng.binomial(1, p_treat)

    treat_rct = rng.binomial(1, 0.5, n)

    def outcome(treat):
        p_click = 0.1 + 0.5 * engagement + TRUE_EFFECT * treat
        return rng.binomial(1, np.clip(p_click, 0, 1))

    return pd.DataFrame({
        "engagement": engagement,
        "treat_obs": treat_obs, "clicked_obs": outcome(treat_obs),
        "treat_rct": treat_rct, "clicked_rct": outcome(treat_rct),
    }), TRUE_EFFECT

if __name__ == "__main__":
    df, true_effect = simulate()
    print(f"TRUE causal effect (planted): {true_effect:.3f}\n")

    naive = df[df.treat_obs==1].clicked_obs.mean() - df[df.treat_obs==0].clicked_obs.mean()
    print(f"Naive difference (observational, CONFOUNDED): {naive:.3f}")
    print("  ^ this is BIASED — inflated because engaged users are both more")
    print("    treated AND more likely to click regardless of treatment.\n")

    rct = df[df.treat_rct==1].clicked_rct.mean() - df[df.treat_rct==0].clicked_rct.mean()
    print(f"Randomized difference (A/B, UNCONFOUNDED): {rct:.3f}")
    print("  ^ this should land near the true 0.10 — randomization killed the confounder.")