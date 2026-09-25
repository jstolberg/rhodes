# %% [markdown]
# # Step 2, start values: many starts, keep the best
#
# The fit of step 2 (`step2_fft.py`) needs a good start for the geometry of every
# key.  Why: the loss has several valleys in $p_o$.  Whenever one harmonic of the
# model passes through zero as $p_o$ changes, its log magnitude drops to minus
# infinity and the loss has a wall there.  Gradient descent cannot cross such a
# wall, so it ends in whatever valley it starts in.
#
# So every key gets 60 starts: 20 values of $p_o$ (0.3 mm apart) x 3 values of
# $A_0$, all with $p_d$ = 2 mm and $\sigma_0$ from step 1.  All starts of all keys
# are improved at the same time with Adam, in three rounds.  After each round but
# the last only the best third of a key's starts survive (60 -> 20 -> 7).  To be
# fast, only the dynamic `FIT_DYN` and a coarse view (4 frames, 768 samples each)
# is used, and the level is fitted per key, so only the shape counts.
#
# The survivors are saved to `cache/init_{TAG}_{stride}.pkl`; `step2_fft.py`
# starts from the best one of every key (`winner`).
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `TAG` | real / syn | recordings or synthetic test data |
# | `PD_INIT` | 2 mm | $p_d$ of every start |
# | `PO_STARTS`, `A0_STARTS` | 20 x 3 | start values of $p_o$ and $A_0$ |
# | `ROUND_STEPS`, `KEEP` | (20, 20, 40), 1/3 | Adam steps per round; fraction kept after each round but the last |
# | `K_INIT`, `L_INIT` | 4, 768 | frames and samples per frame of the coarse view |

# %%
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from step2_lib import (CACHE, J_FIT, K_FRAMES, KEY_STRIDE, PO_MAX, features, fit, load_keys,
                       loss_per_key, raw_a0, raw_pd, raw_po, report_mask, unpack)

TAG          = "syn"
PD_INIT      = 2.0e-3
PO_STARTS    = np.linspace(0.15e-3, PO_MAX - 0.15e-3, 20)
A0_STARTS    = np.array([0.1e-3, 0.5e-3, 2.5e-3])
ROUND_STEPS  = (20, 20, 40)
KEEP         = 1 / 3
K_INIT, L_INIT = 4, 768


def init_path(tag, stride=KEY_STRIDE):
    """Where the start values are cached."""
    return f"{CACHE}/init_{tag}_{stride}.pkl"


def scorer(keys, Ht, eta, mask):
    """Function that scores a batch of starts: q (arrays (S, Mk)) -> loss (S, Mk).
    Uses only the FIT_DYN recordings and K_INIT of the frames."""
    kidx = np.round(np.linspace(0, K_FRAMES - 1, K_INIT)).astype(int)     # frames 0, 4, 7, 11
    sl = np.s_[:, J_FIT:J_FIT + 1, :, kidx]
    logHt, eta_r, mask_r = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    log_sig = jnp.log(jnp.asarray(keys["sig1"]))                        # held at step 1

    def scores(q):
        def one(po, pd, a0):                                            # one start for every key
            theta = dict(po_raw=po, pd_raw=pd, log_sig=log_sig, A0_raw=a0[:, None])
            return loss_per_key(theta, keys, logHt, eta_r, mask_r, kidx, L_INIT)
        return jax.vmap(one)(q["po_raw"], q["pd_raw"], q["A0_raw"])     # over the starts
    return scores


def multi_start(keys, Ht, eta, mask):
    """Run the rounds.  Returns dict(q = surviving starts (S, Mk), sorted best first per
    key, scores = their losses (S, Mk), notes, rounds)."""
    Mk = len(keys["f0s"])
    scores = scorer(keys, Ht, eta, mask)

    # all combinations of PO_STARTS x A0_STARTS, the same for every key
    po, a0 = np.meshgrid(PO_STARTS, A0_STARTS, indexing="ij")
    S = po.size
    q = dict(po_raw=raw_po(np.repeat(po.reshape(S, 1), Mk, 1)),
             pd_raw=raw_pd(np.full((S, Mk), PD_INIT)),
             A0_raw=raw_a0(np.repeat(a0.reshape(S, 1), Mk, 1)))

    state, kk = None, np.arange(Mk)
    for r, steps in enumerate(ROUND_STEPS):
        # Adam on the sum of all scores: the starts do not interact, so this improves
        # each of them independently.  The Adam state is carried from round to round.
        q, _, state = fit(q, lambda q: scores(q).sum(), steps,
                          label=f"round {r + 1}, {S} starts x {Mk} keys: ", state=state)
        L = np.asarray(jax.jit(scores)(q))

        # keep the best starts of every key (all of them after the last round, sorted)
        S = S if r == len(ROUND_STEPS) - 1 else max(1, int(np.ceil(S * KEEP)))
        top = np.argsort(L, axis=0)[:S]                                 # (S, Mk)
        q = {k: v[top, kk] for k, v in q.items()}
        state = jax.tree_util.tree_map(lambda v: v[top, kk] if getattr(v, "ndim", 0) == 2 else v, state)
        L = L[top, kk]
        print("  best per key:", np.round(L[0], 4), " runner-up:", np.round(L[1], 4))

    return dict(q={k: np.asarray(v) for k, v in q.items()}, scores=L,
                notes=keys["notes"], rounds=ROUND_STEPS)


def winner(cache):
    """Start of the fit: the best surviving start of every key (A0_raw as (Mk, 1))."""
    q = cache["q"]
    return dict(po_raw=jnp.asarray(q["po_raw"][0]), pd_raw=jnp.asarray(q["pd_raw"][0]),
                A0_raw=jnp.asarray(q["A0_raw"][0])[:, None])


# %%
if __name__ == "__main__":
    keys = load_keys()
    Ht, eta, mask = features(TAG, keys)
    report_mask(mask)

    t0 = time.time()
    cache = multi_start(keys, Ht, eta, mask)
    print(f"multi-start in {time.time() - t0:.0f} s")

    os.makedirs(CACHE, exist_ok=True)
    with open(init_path(TAG), "wb") as f:
        pickle.dump(cache, f)
    print(f"-> {init_path(TAG)}")

    g, A0 = unpack(dict(winner(cache), log_sig=jnp.log(jnp.asarray(keys["sig1"]))))
    print("winner p_o (mm):", np.round(np.asarray(g["p_o"]) * 1e3, 2),
          " p_d (mm):", np.round(np.asarray(g["p_d"]) * 1e3, 2),
          " A_0(f) (mm):", np.round(np.asarray(A0[:, 0]) * 1e3, 2))
