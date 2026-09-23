# %% [markdown]
# # Tine modes from the pickup recordings
#
# Read the frequency and decay of the tine's inharmonic modes (Gabrielli et al.
# 2020: ratios 7.1, 20.4, 39.7, 62.9, 93.1 to $f_0$) off the recorded pickup
# signal, without any model of the pickup.
#
# ## What a mode looks like at the pickup output
#
# The tine tip moves as $x(t) = p_o + A_1\sin\theta(t) + a_m\cos(\omega_m t)$
# with $\theta = \omega_1 t$ and $a_m \ll A_1$. The output is
# $\epsilon = -\mathrm{d}\Psi(x)/\mathrm{d}t$. To first order in $a_m$,
#
# $$\Psi(x) \approx \Psi(p_o + A_1\sin\theta) + a_m\cos(\omega_m t)\,\Psi'(p_o + A_1\sin\theta).$$
#
# The first term is the harmonic series. The second is the mode multiplied by
# $g(\theta) = \Psi'(p_o + A_1\sin\theta)$, a real periodic function of $\theta$:
# $g = \sum_k c_k e^{ik\theta}$ with $c_{-k} = \overline{c_k}$. So every mode
# shows up as a **comb**, with lines at $f_m + k f_0$ and flux amplitudes
# $\tfrac{a_m}{2}|c_k|$. It follows that:
#
# * every line of one comb sits the same fraction $\delta = \mathrm{frac}(f_m/f_0)$
#   off the harmonic grid, which is how lines are grouped into combs;
# * the comb is **symmetric in magnitude about the true mode** ($|c_k| = |c_{-k}|$)
#   once the output's $\mathrm{d}/\mathrm{d}t$ is divided out. This holds for any
#   pickup shape, and for either polarisation of the mode;
# * at low dynamics $g$ is nearly constant, so $|c_0|$ dominates and the true
#   mode is the tallest line of its comb.
#
# From the output alone, $f_m$ is known exactly only modulo $f_0$. The integer
# part comes from carrier dominance and symmetry at low dynamics, checked
# against Gabrielli's keyboard statistics ($\sigma \le 0.4$ for modes 2 and 3).
#
# ## Procedure (per recording)
#
# 1. **Attack window.** Gabrielli used the first 300 ms at very low dynamic,
#    and so do we. The window starts just after the hammer and runs
#    $T = \mathrm{clip}(60/f_0,\,0.3,\,1.5)$ s: longer in the bass, where lines
#    crowd and modes decay slowly.
# 2. **$f_0$ in that window**, from the harmonic series. $f_0$ depends on the
#    dynamic and glides, so the value from the whole note is not good enough.
# 3. **Baseband** around the mode's prior window: mix, low-pass, decimate.
# 4. **Nuisance.** Every harmonic $k f_0$ in the band, with a slowly varying
#    envelope (Legendre polynomial of order `P_HARM`), and every 60 Hz hum line
#    (hum frequency measured per recording from the tail). Both are projected out.
# 5. **Comb pursuit.** Scan the residual with decaying atoms
#    $e^{(i 2\pi f - \lambda)t}$ (a matched filter for decaying lines). Take the
#    strongest line, add its whole comb $(k+\delta) f_0$ to the nuisance, and
#    repeat until nothing stands `MIN_SNR_DB` over the floor.
# 6. **Assign.** The mode is the comb whose tallest line (after dividing by
#    frequency) falls inside the prior window $\mu \pm 3\sigma$. Symmetry and the
#    margin to the runner-up are reported as a confidence.
# 7. **Refine** frequency and decay of that line by non-linear least squares,
#    with everything else held as nuisance.

# %%
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.optimize as so
import scipy.signal as sg
from scipy.ndimage import median_filter

SAMPLES = "Samples"

# Gabrielli et al. 2020, Fig. 20: keyboard mean and std of f_m / f0.
MODES = {                 # name: (mu, sigma)
    "m2": (7.1, 0.3),
    "m3": (20.4, 0.4),
    "m4": (39.7, 0.9),
    "m5": (62.9, 1.9),
    "m6": (93.1, 2.7),
}
SUB = (0.48, 0.19)        # tonebar sub-fundamental, same figure

F_MAX = 12000.0           # Gabrielli: no overtones above ~10 kHz
HUM_F = 60.0
P_HARM = (3, 5)           # envelope order for harmonics: windows up to 0.6 s, longer
P_HUM = 1
P_COMB = 2                # envelope order for combs already found
MIN_SNR_DB = 12.0         # a line must stand this far over the local floor
MAX_COMBS = 14            # pursuit iterations per band
DUP_CELLS = 2.5           # combs closer than this many 1/T cells are one comb
LAMS = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])   # 1/s, decay grid for the scan
LAM_MAX = 150.0           # 1/s = 1300 dB/s: gone in 7 ms, the hammer, not a mode
                          # (Gabrielli's fastest mode: 294 dB/s)

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def f0_nominal(name: str) -> float:
    """Sample-library note name -> Hz. The files are labelled an octave low (E0 = 41.2 Hz)."""
    midi = (int(name[-1]) + 2) * 12 + _NOTE_NAMES.index(name[:-1])
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def load(note, dyn, samples=SAMPLES):
    fs, x = wavfile.read(f"{samples}/{note}-{dyn}.wav")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x / np.abs(x).max(), float(fs)


def onset(x, thresh=0.1):
    return int(np.argmax(np.abs(x) > thresh * np.abs(x).max()))


def window_len(f0):
    return float(np.clip(60.0 / f0, 0.3, 1.5))


def p_harm(T):
    """Harmonic envelopes move more over a long window (bass: decaying A1 through a
    strongly non-linear pickup). Order 5 over 1.4 s absorbs about +-2 Hz around each
    harmonic; lines closer than that cannot be told from the harmonic anyway."""
    return P_HARM[0] if T <= 0.6 else P_HARM[1]


# %%
# ---------- f0 and hum ----------

def _dtft(seg, fs, freqs):
    """DTFT of seg at arbitrary frequencies (Hz)."""
    n = np.arange(seg.size) / fs
    return np.exp(-2j * np.pi * np.outer(freqs, n)) @ seg


def refine_f0(seg, fs, f0_guess, kmax=12, span=0.008, n_grid=161, iters=3):
    """f0 maximising the summed energy of harmonics 1..kmax in seg (Hann-windowed)."""
    w = sg.windows.hann(seg.size) * seg
    ks = np.arange(1, kmax + 1)
    ks = ks[ks * f0_guess < min(F_MAX, 0.45 * fs)]
    lo, hi = f0_guess * (1 - span), f0_guess * (1 + span)
    for _ in range(iters):
        grid = np.linspace(lo, hi, n_grid)
        E = np.zeros(n_grid)
        for k in ks:
            E += np.abs(_dtft(w, fs, k * grid)) ** 2
        j = int(np.clip(np.argmax(E), 1, n_grid - 2))
        step = grid[1] - grid[0]
        lo, hi = grid[j] - 2 * step, grid[j] + 2 * step
    a, b, c = np.log(E[j - 1:j + 2])
    return float(grid[j] + 0.5 * (a - c) / (a - 2 * b + c) * step)


def hum_freq(x, fs, f0, t0=4.0, t1=7.2):
    """Mains frequency from the tail, using odd hum harmonics away from the note's harmonics."""
    t1 = min(t1, x.size / fs - 0.05)
    t0 = min(t0, 0.5 * t1)
    seg = x[int(t0 * fs):int(t1 * fs)]
    w = sg.windows.blackmanharris(seg.size) * seg
    est, wts = [], []
    for h in (1, 3, 5, 7, 9):
        fh = HUM_F * h
        if abs(fh / f0 - round(fh / f0)) * f0 < 3.0 or fh > 0.45 * fs:
            continue
        grid = np.linspace(fh - 0.1 * h, fh + 0.1 * h, 401)
        E = np.abs(_dtft(w, fs, grid))
        j = int(np.clip(np.argmax(E), 1, 399))
        a, b, c = np.log(E[j - 1:j + 2])
        est.append((grid[j] + 0.5 * (a - c) / (a - 2 * b + c) * (grid[1] - grid[0])) / h)
        wts.append(E[j])
    if not est:
        return HUM_F
    return float(np.average(est, weights=wts))


# %%
# ---------- phase of the fundamental ----------
#
# The pickup is a static non-linearity, so harmonic k is phase-locked to the fundamental:
# its phase is exactly k times the fundamental's. The tine's pitch glides during the
# attack (0.2-0.4 rad of phase wander over the bass window at p, ~1.5 rad at f), and
# harmonic 20 wanders 20 times as much -- far too much for an envelope polynomial.
# So the harmonic columns carry k * psi(t), with psi measured on the fundamental.

P_PHASE = 6               # envelope order for the fundamental: follows the glide, but is
                          # too slow to follow a beat with the sub-fundamental


class PhaseTrack:
    """Slow phase psi(t) of the f0 line, t in s since the onset."""

    def __init__(self, x, fs, f0, i_on, t0, t1):
        i0, i1 = i_on + int(t0 * fs), i_on + int(t1 * fs)
        t = np.arange(i0, i1) / fs - i_on / fs
        self.t0, self.t1 = t[0], t[-1]
        cols = [line_cols(t, np.array([k * f0]), P_PHASE) for k in (1, 2, 3)]
        A = np.hstack(cols + [c.conj() for c in cols])
        c, *_ = np.linalg.lstsq(A, x[i0:i1].astype(complex), rcond=None)
        self.c1 = c[:P_PHASE + 1]
        e = self.envelope(t)
        self.psi_c = np.polynomial.legendre.Legendre.fit(t, np.unwrap(np.angle(e)), P_PHASE)

    def envelope(self, t):
        tau = 2 * (t - self.t0) / (self.t1 - self.t0) - 1
        return np.polynomial.legendre.legval(tau, self.c1)

    def psi(self, t):
        return self.psi_c(np.clip(t, self.t0, self.t1))


# %%
# ---------- band model ----------

HUM_H_MAX = 40            # hum lines above 40 x 60 Hz are below the floor


def legendre_cols(t, P):
    tau = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
    return np.stack([np.polynomial.legendre.Legendre.basis(p)(tau) for p in range(P + 1)], 1)


def line_cols(t, f_rel, P, lam=0.0):
    """Columns e^{(i2pi f - lam) t} * Legendre_p(t) for each f in f_rel (Hz, relative to fc)."""
    if len(f_rel) == 0:
        return np.zeros((t.size, 0), complex)
    L = legendre_cols(t, P) * np.exp(-lam * (t - t[0]))[:, None]
    E = np.exp(2j * np.pi * np.outer(t, f_rel))
    return (E[:, :, None] * L[:, None, :]).reshape(t.size, -1)


def _orth(A):
    if A.shape[1] == 0:
        return A
    Q, R = np.linalg.qr(A)
    d = np.abs(np.diag(R))
    return Q[:, d > 1e-9 * d.max()]


@dataclass
class Band:
    fc: float
    f0: float
    T: float
    t: np.ndarray                     # s since onset
    y: np.ndarray                     # complex baseband (a real cosine of amplitude a -> a/2)
    fs_d: float
    lo: float                         # modelled region, Hz
    hi: float
    half_bw: float                    # scanned region |f - fc| <= half_bw
    base: np.ndarray                  # harmonics and hum
    combs: list = field(default_factory=list)   # dicts: delta, lam, lines {k: (f, amp, snr_db, lam)}
    floor: float = np.nan


def comb_lines(band, delta):
    """k and frequency of every line (k+delta) f0 of one comb inside the modelled region,
    plus the mirror images -(k+delta) f0 (these matter only for bands near 0 Hz)."""
    kmax = int(np.ceil(max(abs(band.lo), abs(band.hi)) / band.f0)) + 2
    kk = np.arange(0, kmax)
    f = (kk + delta) * band.f0
    ks = np.concatenate([kk, -kk - 1000])          # images tagged with k <= -1000
    f = np.concatenate([f, -f])
    m = (f > band.lo) & (f < band.hi)
    return ks[m], f[m]


def nuisance(band, skip=None):
    """Orthonormal basis: harmonics, hum and the visible lines of every comb found so far,
    each comb with its own decay and envelope order. skip = (comb index, k) leaves one
    line out. Lines under the floor get no columns: they carry nothing, and every column
    costs degrees of freedom the scan needs."""
    cols = [band.base]
    for i, c in enumerate(band.combs):
        ks = [k for k in c["ks"] if not (skip is not None and skip[0] == i and k == skip[1])]
        fs_ = np.array([(k + c["delta"]) * band.f0 for k in ks])
        cols.append(line_cols(band.t, fs_ - band.fc, c["P"], c["lam"]))
    return _orth(np.hstack(cols))


def _scan(band, r):
    """Matched-filter score over (lam, f) with one FFT per decay, *without* the nuisance
    correction of the atom norm: |a^H r|^2 / ||a||^2. Since r is orthogonal to the
    nuisance this never exceeds the exact score |a^H r|^2 / ||P_perp a||^2; the two differ
    only next to nuisance lines. Used to find peaks; `exact` scores the chosen cells.
    """
    t = band.t - band.t[0]
    nfft = int(2 ** np.ceil(np.log2(8 * t.size)))
    f = np.fft.fftfreq(nfft, 1 / band.fs_d)
    sel = np.abs(f) <= band.half_bw
    order = np.argsort(f[sel])
    f_rel = f[sel][order]
    S = np.zeros((LAMS.size, f_rel.size))
    for i, lam in enumerate(LAMS):
        e = np.exp(-lam * t)
        S[i] = np.abs(np.fft.fft(r * e, nfft)[sel][order]) ** 2 / np.sum(e ** 2)
    return f_rel, S


def exact(band, Q, r, f_abs):
    """Exact score, amplitude and resolved fraction ||P_perp a||^2/||a||^2 of the atom at
    f_abs (Hz) for every decay in LAMS. For white noise of power sigma^2 the score has
    mean sigma^2."""
    t = band.t - band.t[0]
    A = np.exp((2j * np.pi * (f_abs - band.fc) - LAMS[None, :]) * t[:, None])   # (n_t, n_lam)
    na = np.sum(np.abs(A) ** 2, 0)
    den = na - np.sum(np.abs(Q.conj().T @ A) ** 2, 0) if Q.shape[1] else na
    den = np.maximum(den, 1e-9 * na)
    num = A.conj().T @ r
    return np.abs(num) ** 2 / den, num / den, den / na


def make_band(x, fs, f0, fc, half_bw, i_on, T, f_hum, track=None):
    """Baseband around fc plus the harmonic and hum nuisance.

    The window starts after the onset by the larger of 5 ms and the FIR half-length, so
    the filtered hammer impact does not leak into it. The FIR runs over the file from its
    start: the silence before the onset keeps its edge clean.
    """
    guard = max(1.5 * f0, 30.0)
    n = int(T * fs)
    ntaps, beta = sg.kaiserord(100.0, guard / (fs / 2))
    ntaps |= 1
    i0 = i_on + max(int(0.005 * fs), ntaps // 2)
    taps = sg.firwin(ntaps, half_bw + 1.5 * guard, fs=fs, window=("kaiser", beta))
    stop = min(x.size, i0 + n + ntaps)
    xm = x[:stop] * np.exp(-2j * np.pi * fc * np.arange(stop) / fs)
    yf = sg.fftconvolve(xm, taps, mode="same")
    d = max(1, int(np.floor(fs / (2.5 * (half_bw + 2 * guard)))))
    y = yf[i0:i0 + n:d]
    fs_d = fs / d
    t = (i0 - i_on) / fs + np.arange(y.size) / fs_d
    lo, hi = fc - half_bw - 2 * guard, fc + half_bw + 2 * guard
    ks = np.arange(int(np.ceil(lo / f0)), int(np.floor(hi / f0)) + 1)       # k = 0: DC drift
    hs = np.arange(int(np.ceil(lo / f_hum)), int(np.floor(hi / f_hum)) + 1)
    hs = hs[(hs != 0) & (np.abs(hs) <= HUM_H_MAX)]
    H = line_cols(t, ks * f0 - fc, p_harm(T))
    if track is not None:                       # phase-lock harmonic k to k * psi
        lock = np.exp(1j * np.outer(track.psi(t), ks))                # (n_t, n_k)
        H = (H.reshape(t.size, ks.size, -1) * lock[:, :, None]).reshape(t.size, -1)
    base = [H]
    if hs.size:
        base.append(line_cols(t, hs * f_hum - fc, P_HUM))
    return Band(fc=fc, f0=f0, T=T, t=t, y=y, fs_d=fs_d, lo=lo, hi=hi, half_bw=half_bw,
                base=np.hstack(base))


TRIM = 0.8                # floor: mean of the lowest 80 % of periodogram bins ...
TRIM_BIAS = 1 - (1 - TRIM) * (1 - np.log(1 - TRIM)) / TRIM   # ... over that mean for noise


def noise_floor(band, r):
    """Noise power in the units of the scan score (for white noise the score's mean).

    A trimmed mean of the Hann periodogram of the residual over the scanned band. Hann
    keeps sidelobes of the remaining lines local, the trim drops the lines themselves and
    leftover humps at harmonics, and TRIM_BIAS undoes the trim for exponential noise bins.
    One number for the band: a local floor lets the scan's own (rectangular) sidelobes
    pass as lines.
    """
    t = band.t - band.t[0]
    w = sg.windows.hann(t.size)
    nfft = int(2 ** np.ceil(np.log2(4 * t.size)))
    P = np.abs(np.fft.fft(r * w, nfft)) ** 2 / np.sum(w ** 2)
    f = np.fft.fftfreq(nfft, 1 / band.fs_d)
    P = np.sort(P[np.abs(f) <= band.half_bw])
    return float(P[:int(TRIM * P.size)].mean() / TRIM_BIAS)


def _comb_energy(band, f_rel, S, delta):
    """Summed matched energy of a comb's lines, per decay (nearest grid cell)."""
    df = f_rel[1] - f_rel[0]
    _, fs_ = comb_lines(band, delta)
    fs_ = fs_[np.abs(fs_ - band.fc) <= band.half_bw]
    j = np.clip(np.round((fs_ - band.fc - f_rel[0]) / df).astype(int), 0, f_rel.size - 1)
    return S[:, j].sum(1)


def pursue(band):
    """Find combs one at a time until nothing stands MIN_SNR_DB over the floor.

    Strongest first, so a sidelobe never outranks its line. A comb that keeps coming back
    after its envelope has been loosened is left as it is and its lines are masked out
    of the scan (CLEAN-style), so one imperfect removal cannot stall the search.
    """
    masked = []                                  # (f_lo, f_hi) absolute Hz
    for _ in range(MAX_COMBS):
        Q = nuisance(band)
        r = band.y - Q @ (Q.conj().T @ band.y)
        f_rel, S = _scan(band, r)
        s = S.max(0)
        f_abs = band.fc + f_rel
        for a, b in masked:
            s[(f_abs > a) & (f_abs < b)] = 0.0
        band.floor = floor = noise_floor(band, r)
        lf_f, lf = local_floor(band, r)
        loc = np.interp(f_rel, lf_f, lf)
        j = int(np.argmax(s))
        Sj, _, _ = exact(band, Q, r, f_abs[j])
        if 10 * np.log10(max(Sj.max(), 1e-300) / floor) < MIN_SNR_DB:
            break
        if 10 * np.log10(max(Sj.max(), 1e-300) / loc[j]) < MIN_SNR_DB:
            # loud, but no louder than its own neighbourhood: the leakage skirt of a
            # harmonic. Modelling it as a comb would plant columns all over the band.
            masked.append((f_abs[j] - 1.0 / band.T, f_abs[j] + 1.0 / band.T))
            continue
        # delta and decay from the whole comb: maximise its summed energy
        df = f_rel[1] - f_rel[0]
        d0 = (f_abs[j] / band.f0) % 1.0
        cand = d0 + np.arange(-3, 4) * df / band.f0
        E = np.array([_comb_energy(band, f_rel, S, d) for d in cand])       # (cand, lam)
        ic, il = np.unravel_index(int(np.argmax(E)), E.shape)
        delta = float(cand[ic] % 1.0)
        # Within ~2.5/T Hz of a comb already found, a "new" comb is that comb's leftover
        # skirt (its sideband envelopes are not a clean exponential where the pickup is
        # strongly non-linear), and it would replicate on every k. Loosen the old one.
        tol = max(DUP_CELLS / (band.T * band.f0), 0.002)
        dup = [c for c in band.combs if _dist(c["delta"], delta) < tol]
        comb = {"delta": delta, "lam": float(LAMS[il]), "floor": floor, "lines": {},
                "P": P_COMB, "ks": []}
        w = max(1, int(round(0.25 / (band.T * df))))          # +- 1/(4T) Hz
        ks, fs_ = comb_lines(band, delta)
        # lines already owned by earlier (stronger) combs: a later comb gets no columns
        # within the resolution of them, or weak junk combs end up absorbing a real line
        owned = np.array([(k2 + c2["delta"]) * band.f0 for c2 in band.combs for k2 in c2["ks"]])
        for k, fk in zip(ks, fs_):
            if abs(fk - band.fc) > band.half_bw:
                continue
            jj = int(round((fk - band.fc - f_rel[0]) / df))
            a, b = max(0, jj - w), min(s.size, jj + w + 1)
            if b <= a:
                continue
            m = a + int(np.argmax(s[a:b]))
            Sm, Cm, Dm = exact(band, Q, r, f_abs[m])
            li = int(np.argmax(Sm))
            snr = float(10 * np.log10(max(Sm[li], 1e-300) / floor))
            snr_l = float(10 * np.log10(max(Sm[li], 1e-300) / loc[m]))
            comb["lines"][int(k)] = (f_abs[m], float(abs(Cm[li])), snr, float(LAMS[li]),
                                     float(Dm[li]), float(Sm[li]))
            free = owned.size == 0 or np.min(np.abs(owned - fk)) > 2.0 / band.T
            if min(snr, snr_l) >= MIN_SNR_DB - 3 and (free or dup):
                comb["ks"].append(int(k))
        if dup:
            # The same comb again. Either lines that were under the (then higher) floor
            # have surfaced -- give them columns -- or its envelope was too stiff.
            old = dup[0]
            new = set(comb["ks"]) - set(old["ks"])
            if new:
                old["ks"] = sorted(set(old["ks"]) | new)
                for k in new:
                    old["lines"][k] = comb["lines"][k]
            elif old["P"] < P_COMB + 4:
                old["P"] += 2
            else:
                hw = 1.0 / band.T
                masked += [((k + old["delta"]) * band.f0 - hw, (k + old["delta"]) * band.f0 + hw)
                           for k in old["ks"]]
                masked.append((f_abs[j] - hw, f_abs[j] + hw))
            continue
        if not comb["ks"]:
            masked.append((f_abs[j] - 1.0 / band.T, f_abs[j] + 1.0 / band.T))
            continue
        band.combs.append(comb)
    _local_visibility(band)
    return band


RESOLVED = 0.5            # a line whose atom keeps less than half its norm outside the
                          # nuisance span is mostly harmonic: its amplitude is not measurable


def local_floor(band, r):
    """(f_rel, floor): running median (+-f0/4) of a Hann periodogram of r, over ln 2."""
    t = band.t - band.t[0]
    w = sg.windows.hann(t.size)
    nfft = int(2 ** np.ceil(np.log2(8 * t.size)))
    P = np.abs(np.fft.fft(r * w, nfft)) ** 2 / np.sum(w ** 2)
    f = np.fft.fftfreq(nfft, 1 / band.fs_d)
    o = np.argsort(f)
    f, P = f[o], P[o]
    W = max(5, int(round(0.5 * band.f0 / (f[1] - f[0]))) | 1)
    return f, median_filter(P, size=W, mode="nearest") / np.log(2)


def _local_visibility(band):
    """Judge every comb line against the floor in its own neighbourhood.

    Detection uses one floor for the band (a local one lets the scan's sidelobes pass as
    lines). Deciding which comb lines are *visible* is different: a line in the leakage
    skirt of a huge harmonic can be loud and still be leakage. So compare its score with
    a running median (+-f0/4) of a Hann periodogram of the final residual.
    """
    Q = nuisance(band)
    r = band.y - Q @ (Q.conj().T @ band.y)
    f, loc = local_floor(band, r)
    for c in band.combs:
        for k, v in c["lines"].items():
            fl = float(np.interp(v[0] - band.fc, f, loc))
            c["lines"][k] = v[:6] + (float(10 * np.log10(v[5] / max(fl, 1e-300))),)


def refine_mode(band, ci, n, J=3, n_f=61, n_lam=41):
    """Frequency and decay of the mode whose carrier is line n of comb ci.

    Every line of a mode's comb sits at f_m + j f0 with one decay, so fit (f_m, lam)
    jointly on the carrier and its visible sidebands |j| <= J, each line with its own
    complex amplitude, the whole comb taken out of the nuisance. Grid search (the score
    is not convex in lam for short-lived lines) over f_m +- 1/T and lam in [0, LAM_MAX],
    summing the lines' matched scores (they are ~orthogonal, f0 >> 1/T apart).
    Returns f_m (Hz), lam (1/s), |c| of the carrier (baseband amplitude at window start).
    """
    comb = band.combs[ci]
    prof = comb_profile(band, comb)
    js = [0] + [j for j in range(-J, J + 1) if j and n + j in prof and prof[n + j][1]]
    saved = comb["ks"]
    comb["ks"] = []
    Q = nuisance(band)
    comb["ks"] = saved
    t = band.t - band.t[0]
    y = band.y - Q @ (Q.conj().T @ band.y)
    def grid(fg, lg):
        total = np.zeros((lg.size, fg.size))
        carrier = np.zeros((lg.size, fg.size), complex)
        for j in js:
            E = np.exp(2j * np.pi * np.outer(t, fg + j * band.f0 - band.fc))
            for il, lam in enumerate(lg):
                A = E * np.exp(-lam * t)[:, None]
                A = A - Q @ (Q.conj().T @ A)
                na = np.sum(np.abs(A) ** 2, 0)
                num = A.conj().T @ y
                total[il] += np.abs(num) ** 2 / na
                if j == 0:
                    carrier[il] = num / na
        il, jf = np.unravel_index(int(np.argmax(total)), total.shape)
        return fg[jf], lg[il], carrier[il, jf]

    f_c = comb["lines"][n][0]
    span = 1.0 / band.T
    f1, l1, _ = grid(f_c + np.linspace(-span, span, n_f), np.linspace(0.0, LAM_MAX, n_lam))
    # second, fine pass around the coarse optimum
    df, dl = 2 * span / (n_f - 1), LAM_MAX / (n_lam - 1)
    f2, l2, c2 = grid(f1 + np.linspace(-df, df, 21),
                      np.clip(l1 + np.linspace(-dl, dl, 21), 0.0, LAM_MAX))
    return float(f2), float(l2), float(abs(c2)), js


# %%
# ---------- which comb line is the mode ----------

def _dist(a, b):
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def comb_profile(band, comb):
    """k -> level in dB with d/dt divided out; lines under MIN_SNR_DB get the floor
    level at that frequency instead, marked as an upper bound."""
    prof = {}
    for k, (f, amp, snr, _, frac, _, snr_loc) in comb["lines"].items():
        if k < 0 or frac < RESOLVED:
            continue
        lev = 20 * np.log10(amp / f)
        snr_v = min(snr, snr_loc)
        if snr_v >= MIN_SNR_DB:
            prof[k] = (lev, True)
        else:
            prof[k] = (lev - snr_v + MIN_SNR_DB, False)   # what MIN_SNR_DB would have needed
    return prof


SYM_W = (1.0, 0.5, 0.25)  # weight of pair j = 1, 2, 3: the nearest sidebands are the loudest
SYM_CLIP_DB = 12.0        # one pair spoilt by a leaking harmonic must not decide alone


def symmetry(prof, n):
    """Weighted mean |L(n+j) - L(n-j)| (dB) over the pairs where one line is visible.

    Two lines under the floor carry no information and are skipped; one visible and one
    under the floor counts only if the visible one is the louder (else they could be equal).
    """
    d, w = [], []
    for j, wj in enumerate(SYM_W, start=1):
        if n + j not in prof or n - j not in prof:
            continue
        (a, va), (b, vb) = prof[n + j], prof[n - j]
        if va and vb:
            dj = abs(a - b)
        elif va and a > b:
            dj = a - b
        elif vb and b > a:
            dj = b - a
        else:
            continue
        d.append(min(dj, SYM_CLIP_DB))
        w.append(wj)
    return (float(np.average(d, weights=w)) if d else 0.0), len(d)


def choose_carrier(band, comb, mu, sigma):
    """Candidate k with (k + delta) in mu +- 3 sigma: the one about which the comb is most
    symmetric, among those that are visible themselves. Returns dict or None."""
    prof = comb_profile(band, comb)
    cands = [k for k in prof if prof[k][1] and abs(k + comb["delta"] - mu) <= 3 * sigma]
    if not cands:
        return None
    vis = {k: v[0] for k, v in prof.items() if v[1]}
    rows = []
    for n in cands:
        asym, npairs = symmetry(prof, n)
        others = [vis[k] for k in vis if 0 < abs(k - n) <= len(SYM_W)]
        margin = vis[n] - max(others) if others else np.inf
        z = (n + comb["delta"] - mu) / sigma
        rows.append(dict(k=n, asym=asym, npairs=npairs, margin=margin, z=z))
    rows.sort(key=lambda r: (r["asym"] + 0.5 * r["z"] ** 2))
    best = rows[0]
    best["runner_up"] = rows[1]["asym"] + 0.5 * rows[1]["z"] ** 2 if len(rows) > 1 else np.inf
    best["score"] = best["asym"] + 0.5 * best["z"] ** 2
    return best


# %%
# ---------- one recording ----------

LOW_BAND = (0.1, 0.95)    # x f0: below the fundamental only carriers of low modes live
SUB_LAM_MAX = 20.0        # 1/s (174 dB/s); Gabrielli's subs decay at 9 and 138 dB/s
HALF_SIDE = 3.2           # band reaches this many f0 past the prior window, for sidebands
DOMINANT_DB = 6.0         # a comb sharing delta with a low comb must dominate this much
MAX_ASYM_DB = 6.0         # a mode's comb is symmetric about it; worse than this is not a comb
MIN_MARGIN_DB = -10.0     # carrier vs loudest other line of its comb. Where the tine swings
                          # across the pole the carrier drops below its sidebands (~ -3 dB in
                          # the synthetic bass), but not by tens of dB: that is leakage.


def fundamental_amp(x, fs, f0, i_start, n=None):
    """Amplitude of the f0 line at i_start (full rate, cubic envelope over 50 ms)."""
    n = n or int(0.05 * fs)
    t = np.arange(n) / fs
    A = line_cols(t, np.array([f0]), 3)
    A = np.hstack([A, A.conj()])
    c, *_ = np.linalg.lstsq(A, x[i_start:i_start + n].astype(complex), rcond=None)
    return float(2 * abs(c[0] - c[1] + c[2] - c[3]))   # Legendre_p(-1) = (-1)^p


def analyze_recording(x, fs, f0_guess, modes=MODES, name=("", "")):
    """All modes of one recording. Returns (list of dicts, context dict)."""
    i_on = onset(x)
    T = window_len(f0_guess)
    i5 = i_on + int(0.005 * fs)
    f0 = refine_f0(x[i5:i5 + int(T * fs)], fs, f0_guess)
    f_hum = hum_freq(x, fs, f0)
    tol = max(0.01, 1.0 / (T * f0))           # combs closer than one resolution cell
    track = PhaseTrack(x, fs, f0, i_on, 0.005, T + 0.12)

    lo_c = 0.5 * (LOW_BAND[0] + LOW_BAND[1]) * f0
    low = pursue(make_band(x, fs, f0, lo_c, 0.5 * (LOW_BAND[1] - LOW_BAND[0]) * f0, i_on, T, f_hum,
                           track))
    # The sub-fundamental: the strongest *slowly decaying* comb below f0. The low band also
    # holds the hammer's thump, which dies within tens of ms; taking that for the sub
    # would only cause false collisions.
    low_deltas, sub = [], None
    slow = [c for c in low.combs if c["lam"] <= SUB_LAM_MAX]
    if slow:
        sub = slow[0]
        low_deltas = [sub["delta"], (1.0 - sub["delta"]) % 1.0]

    out = []
    for name_m, (mu, sigma) in modes.items():
        row = dict(note=name[0], dyn=name[1], mode=name_m, mu=mu, f0=f0, found=False, T=T)
        # Skip only if the expected range itself is above F_MAX (Gabrielli: nothing above
        # ~10 kHz). The sideband margin on top is trimmed to what the sample rate allows.
        if (mu - 3 * sigma) * f0 > F_MAX:
            row["why"] = "above F_MAX"
            out.append(row)
            continue
        guard = max(1.5 * f0, 30.0)
        lo_b = (mu - 3 * sigma - HALF_SIDE) * f0
        hi_b = min((mu + 3 * sigma + HALF_SIDE) * f0, 0.45 * fs - 2 * guard)
        band = pursue(make_band(x, fs, f0, 0.5 * (lo_b + hi_b), 0.5 * (hi_b - lo_b), i_on, T, f_hum,
                                track))
        row["n_combs"] = len(band.combs)
        picks = []
        for ci, comb in enumerate(band.combs):
            ch = choose_carrier(band, comb, mu, sigma)
            if ch is None:
                continue
            if ch["asym"] > MAX_ASYM_DB or ch["margin"] < MIN_MARGIN_DB:
                continue
            # inside a harmonic's envelope bandwidth a line cannot be told from the harmonic
            if min(comb["delta"], 1 - comb["delta"]) * f0 < (p_harm(T) + 1) / (2 * T):
                continue
            near_low = any(_dist(comb["delta"], d) < tol for d in low_deltas)
            if near_low and not ch["margin"] >= DOMINANT_DB:
                continue
            # compare combs by carrier level, not SNR: the floor at detection drops as
            # combs are removed, so later (weaker) combs would look better
            f_k, amp_k, snr = comb["lines"][ch["k"]][:3]
            picks.append((20 * np.log10(amp_k / f_k), snr, ci, ch, near_low))
        if not picks:
            row["why"] = "no comb centred in the prior window"
            out.append(row)
            continue
        _, snr, ci, ch, near_low = max(picks, key=lambda p: p[0])
        f, lam, amp, js = refine_mode(band, ci, ch["k"])
        if lam > 0.95 * LAM_MAX:
            row["why"] = "decay at the limit: an impact transient"
            out.append(row)
            continue
        a1 = fundamental_amp(x, fs, f0, i_on + int(round(band.t[0] * fs)))
        row.update(found=True, ratio=f / f0, f=f, lam=lam, lam_db=8.686 * lam,
                   amp_db=20 * np.log10(2 * amp / a1), snr_db=snr, delta=band.combs[ci]["delta"],
                   k=ch["k"], asym_db=ch["asym"], npairs=ch["npairs"], margin_db=ch["margin"],
                   score=ch["score"], runner_up=ch["runner_up"], near_low=near_low,
                   t_start=float(band.t[0]), n_candidates=len(picks), n_lines=len(js))
        out.append(row)
    ctx = dict(f0=f0, f_hum=f_hum, T=T, low_deltas=low_deltas, low=low, track=track, sub=sub)
    return out, ctx


# %%
# ---------- the whole keyboard ----------

RAW_FILE = "tine_modes_raw.json"
DYNS = ("p", "mp", "mf", "f")


def _job(args):
    note, dyn, f0_guess = args
    import warnings
    warnings.filterwarnings("ignore")
    x, fs = load(note, dyn)
    rows, ctx = analyze_recording(x, fs, f0_guess, name=(note, dyn))
    sub = ctx["sub"]
    for r in rows:
        r["sub_ratio"] = np.nan if sub is None else float(
            max(sub["lines"].items(), key=lambda kv: kv[1][1])[1][0] / ctx["f0"]) if sub["lines"] else np.nan
        r["f_hum"] = ctx["f_hum"]
    return [{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
            for r in rows]


def run_all(workers=None, out=RAW_FILE):
    import json
    from concurrent.futures import ProcessPoolExecutor
    d = np.load("fundamentals.npz", allow_pickle=True)
    notes, dyns, table = list(d["notes"]), list(d["dyns"]), d["table"]
    jobs = [(n, dy, float(table[i, dyns.index(dy)]["f"])) for i, n in enumerate(notes) for dy in DYNS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_job, jobs)):
            rows += res
            print(f"{i + 1}/{len(jobs)} {jobs[i][0]}-{jobs[i][1]}", flush=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=0, default=float)
    return rows



# %%
# ---------- one answer per key ----------
#
# A tine mode is a property of the tine, so the four dynamics of a key must agree on its
# frequency in Hz (f0 itself shifts a little with dynamic, which is why Hz and not the
# ratio is compared). Detections are grouped by frequency; a mode is *confirmed* when at
# least two dynamics land in the same group. Its ratio is given against the p recording's
# f0, the smallest swing and so the closest to the tine's own small-amplitude pitch.

TABLE_FILE = "tine_modes.npz"
MIN_AGREE = 2


def summarise(rows):
    notes = list(dict.fromkeys(r["note"] for r in rows))
    out = []
    for note in notes:
        rn = [r for r in rows if r["note"] == note]
        f0_p = next(r["f0"] for r in rn if r["dyn"] == "p")
        for mode, (mu, sigma) in MODES.items():
            hits = [r for r in rn if r["mode"] == mode and r["found"]]
            row = dict(note=note, mode=mode, f0=f0_p, n_found=len(hits), n_agree=0,
                       f=np.nan, ratio=np.nan, lam_db=np.nan, amp_db=np.nan, spread_hz=np.nan,
                       dyns="", near_low=False)
            if hits:
                T = hits[0]["T"]
                tol = lambda f: max(1.5 / T, 0.0025 * f)
                best = []
                for h in hits:
                    grp = [g for g in hits if abs(g["f"] - h["f"]) <= tol(h["f"])]
                    if len(grp) > len(best) or (len(grp) == len(best) and
                                                sum(g["snr_db"] for g in grp) > sum(g["snr_db"] for g in best)):
                        best = grp
                fs_ = np.array([g["f"] for g in best])
                row.update(n_agree=len(best), f=float(np.median(fs_)), ratio=float(np.median(fs_) / f0_p),
                           lam_db=float(np.median([g["lam_db"] for g in best])),
                           amp_db=float(np.median([g["amp_db"] for g in best])),
                           spread_hz=float(np.ptp(fs_)), dyns=",".join(g["dyn"] for g in best),
                           near_low=any(g["near_low"] for g in best))
            row["confirmed"] = row["n_agree"] >= MIN_AGREE
            out.append(row)
    return out


def save_table(summary, path=TABLE_FILE):
    notes = list(dict.fromkeys(r["note"] for r in summary))
    modes = list(MODES)
    get = lambda key: np.array([[next(r[key] for r in summary if r["note"] == n and r["mode"] == m)
                                 for m in modes] for n in notes])
    np.savez(path, notes=np.array(notes), modes=np.array(modes), f=get("f"), ratio=get("ratio"),
             lam_db=get("lam_db"), amp_db=get("amp_db"), n_agree=get("n_agree"),
             confirmed=get("confirmed"), f0=get("f0")[:, 0])


if __name__ == "__main__":
    import json, os
    if not os.path.exists(RAW_FILE):
        run_all()
    with open(RAW_FILE) as fh:
        summary = summarise(json.load(fh))
    save_table(summary)
