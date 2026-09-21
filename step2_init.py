# %% [markdown]
# # Step 2, initial values: multi-start with successive elimination
#
# Not the optimiser -- an initial-value finder for the geometry of every key.
# The log-magnitude loss is rugged in $p_o$: wherever a harmonic of the model
# passes through zero as a function of the operating point, that harmonic's
# residual spikes, and these walls separate basins that gradient descent
# cannot cross.  $p_o$, $p_d$ and $\sigma_0$ are per key, so one cell per key
# (dynamic `FIT_DYN`) decides the basin.
#
# Every key starts from `PO_STARTS` x `A0_STARTS` starts ($p_d$ = `PD_INIT`,
# $\sigma_0$ from step 1, held).  All starts of all keys are refined in
# parallel with Adam on the gain-free per-key loss, evaluated on `K_INIT` of
# the frames with `L_INIT` samples each (a coarse view is enough to rank
# basins); after each round only the best `KEEP` of a key's starts survive.
# The result -- the surviving candidates, their Adam state and scores -- is
# cached, so it can be inspected or continued.
#
# | Name | Value | Meaning |
# | --- | --- | --- |
# | `TAG` | real / syn | which target features |
# | `PD_INIT` | 2 mm | $p_d$ of every start |
# | `PO_STARTS`, `A0_STARTS` | 20 x 3 | starts in $p_o$ (0.3 mm apart, closer than any basin) and $A_0$ |
# | `ROUND_STEPS`, `KEEP` | (20, 20, 40), 1/3 | Adam steps per round; fraction of a key's starts kept after each round but the last: 60 -> 20 -> 7 |
# | `K_INIT`, `L_INIT` | 4, 768 | frames and samples per frame of the reduced evaluation |

# %%
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from step2_lib import *

TAG          = "syn"
PD_INIT      = 2.0e-3
PO_STARTS    = np.linspace(0.15e-3, PO_MAX - 0.15e-3, 20)
A0_STARTS    = np.array([0.1e-3, 0.5e-3, 2.5e-3])
ROUND_STEPS  = (20, 20, 40)
KEEP         = 1 / 3
K_INIT, L_INIT = 4, 768


def init_path(tag, stride=KEY_STRIDE):
    return f"{CACHE}/init_{tag}_{stride}.pkl"


def scorer(keys, Ht, eta, mask):
    """Gain-free per-key loss of the FIT_DYN cell on the reduced evaluation, for a batch of
    starts: q (S, Mk) arrays -> scores (S, Mk)."""
    kidx = np.round(np.linspace(0, K_FRAMES - 1, K_INIT)).astype(int)
    sl = np.s_[:, J_FIT:J_FIT + 1, :, kidx]
    logHt, eta_r, mask_r = jnp.log(Ht + eta)[sl], eta[sl], mask[sl]
    log_sig = jnp.log(jnp.asarray(keys["sig1"]))

    def scores(q):
        one = lambda po, pd, a0: loss_per_key(dict(po_raw=po, pd_raw=pd, log_sig=log_sig, A0_raw=a0[:, None]),
                                              keys, logHt, eta_r, mask_r, kidx, L_INIT)
        return jax.vmap(one)(q["po_raw"], q["pd_raw"], q["A0_raw"])
    return scores


def multi_start(keys, Ht, eta, mask):
    """Returns the cache dict: candidates q (S, Mk), their Adam state, scores (S, Mk), and
    the winner per key (index 0 after the last round's sort)."""
    Mk = len(keys["f0s"])
    scores = scorer(keys, Ht, eta, mask)
    po, a0 = np.meshgrid(PO_STARTS, A0_STARTS, indexing="ij")
    S = po.size
    q = dict(po_raw=raw_po(np.repeat(po.reshape(S, 1), Mk, 1)),
             pd_raw=raw_pd(np.full((S, Mk), PD_INIT)),
             A0_raw=raw_a0(np.repeat(a0.reshape(S, 1), Mk, 1)))
    state, kk = None, np.arange(Mk)
    for r, steps in enumerate(ROUND_STEPS):
        q, _, state = fit(q, lambda q: scores(q).sum(), steps, label=f"round {r + 1}, {S} starts x {Mk} keys: ", state=state)
        L = np.asarray(jax.jit(scores)(q))
        S = S if r == len(ROUND_STEPS) - 1 else max(1, int(np.ceil(S * KEEP)))
        top = np.argsort(L, axis=0)[:S]                       # (S, Mk): each key keeps its own best starts
        q = {k: v[top, kk] for k, v in q.items()}
        state = jax.tree_util.tree_map(lambda v: v[top, kk] if getattr(v, "ndim", 0) == 2 else v, state)
        L = L[top, kk]
        print("  best per key:", np.round(L[0], 4), " runner-up:", np.round(L[1], 4))
    return dict(q={k: np.asarray(v) for k, v in q.items()}, state=jax.device_get(state), scores=L,
                notes=keys["notes"], rounds=ROUND_STEPS)


def continue_init(cache, keys, Ht, eta, mask, steps):
    """More Adam steps on the cached candidates from their cached Adam state."""
    scores = scorer(keys, Ht, eta, mask)
    q = {k: jnp.asarray(v) for k, v in cache["q"].items()}
    q, _, state = fit(q, lambda q: scores(q).sum(), steps, label=f"continue, {len(cache['scores'])} starts: ", state=cache["state"])
    L = np.asarray(jax.jit(scores)(q))
    top, kk = np.argsort(L, axis=0), np.arange(len(keys["f0s"]))
    q = {k: v[top, kk] for k, v in q.items()}
    state = jax.tree_util.tree_map(lambda v: v[top, kk] if getattr(v, "ndim", 0) == 2 else v, state)
    print("  best per key:", np.round(L[top, kk][0], 4))
    return dict(cache, q={k: np.asarray(v) for k, v in q.items()}, state=jax.device_get(state), scores=L[top, kk],
                rounds=tuple(cache["rounds"]) + (steps,))


def winner(cache):
    """Starting theta of the fit: the best candidate of every key (A0_raw (Mk, 1))."""
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
    print("winner p_o (mm):", np.round(np.asarray(g["p_o"]) * 1e3, 2), " p_d (mm):", np.round(np.asarray(g["p_d"]) * 1e3, 2),
          " A_0(f) (mm):", np.round(np.asarray(A0[:, 0]) * 1e3, 2))
