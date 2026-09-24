# %% [markdown]
# # Tine modes from the pickup recordings
#
# This script finds the frequency and decay of the tine's inharmonic modes in the
# recorded pickup signal. Gabrielli et al. (2020) measured them at about 7.1, 20.4,
# 39.7, 62.9 and 93.1 times $f_0$. No pickup model is needed.
#
# ## Idea
#
# The tine tip moves as $x = p_o + A_1\sin\theta + a_m\cos\omega_m t$, with
# $\theta = \omega_1 t$ and a small mode amplitude $a_m$. The pickup turns this into
#
# $$\Psi(x) \approx \Psi(p_o + A_1\sin\theta) + a_m\cos(\omega_m t)\,\Psi'(p_o + A_1\sin\theta).$$
#
# The first term is the harmonic series. In the second, the mode is multiplied by a
# function that repeats with every period of the fundamental. So a mode does not show
# up as one line. It shows up as a comb of lines at $f_m + k f_0$.
#
# Three facts about the comb make it usable:
#
# * All lines of a comb sit the same distance off the harmonic grid. This is how
#   lines are grouped into combs.
# * Once the output's $\mathrm{d}/\mathrm{d}t$ is divided out, the comb is
#   symmetric about the true mode, whatever the pickup's shape.
# * At soft dynamics the true mode is the loudest line of its comb.
#
# A comb gives $f_m$ only up to a multiple of $f_0$. Symmetry and Gabrielli's
# ratios decide which line is the mode.
#
# ## Steps, per recording
#
# 1. Take a window that starts 5 ms after the onset and runs
#    $T = \mathrm{clip}(60/f_0,\,0.3,\,1.5)$ s. Bass notes get the longer windows.
# 2. Measure $f_0$ in that window, and the mains frequency in the tail.
# 3. For each mode, cut out the band around its expected ratio, $\mu \pm 3\sigma$
#    plus a few $f_0$ on each side for the comb lines.
# 4. Remove the harmonics and the hum from the band.
# 5. Find the combs. Take the strongest line left, remove its whole comb, and
#    repeat until nothing stands `MIN_SNR_DB` above the noise.
# 6. Pick the mode. In each comb, the candidate is the line inside $\mu \pm 3\sigma$
#    that the comb is most symmetric about. Lopsided combs and combs on top of a
#    harmonic are dropped. The loudest remaining comb is the mode.
# 7. Measure the frequency and decay of that line with a grid search over decaying
#    test tones.
#
# Over the keyboard, a mode counts as confirmed when at least two dynamics of a key
# agree on its frequency. The last cell plots the result in the layout of
# Gabrielli's Fig. 20.

# %%
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg

SAMPLES = "Samples"

# Gabrielli et al. 2020, Fig. 20: keyboard mean and std of f_m / f0.
MODES = {                 # name: (mu, sigma)
    "m2": (7.1, 0.3),
    "m3": (20.4, 0.4),
    "m4": (39.7, 0.9),
    "m5": (62.9, 1.9),
    "m6": (93.1, 2.7),
}

F_MAX = 12000.0           # Gabrielli: no overtones above ~10 kHz
HUM_F = 60.0
P_HARM = (3, 5)           # envelope order for harmonics: windows up to 0.6 s, longer
P_HUM = 1
P_COMB = 2                # envelope order for combs already found
MIN_SNR_DB = 12.0         # a line must stand this far over the floor
MAX_COMBS = 14            # pursuit iterations per band
DUP_CELLS = 2.5           # combs closer than this many 1/T cells are one comb
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
    # check up to 12 harmonics at the same 
    ks = np.arange(1, kmax + 1)
    ks = ks[ks * f0_guess < min(F_MAX, 0.45 * fs)]
    lo, hi = f0_guess * (1 - span), f0_guess * (1 + span)
    # refine 3 times 
    for _ in range(iters):
        grid = np.linspace(lo, hi, n_grid)
        E = np.zeros(n_grid)
        # calculate energy for all k*f0 harmonics
        for k in ks:
            E += np.abs(_dtft(w, fs, k * grid)) ** 2
        # get index of highest scoring candidate 
        j = int(np.argmax(E))
        # finer step size for next iteration
        step = grid[1] - grid[0]
        lo, hi = grid[j] - 2 * step, grid[j] + 2 * step
    return float(grid[j])


def hum_freq(x, fs, f0, t0=4.0, t1=7.2):
    """Mains frequency from the tail, using odd hum harmonics away from the note's harmonics."""
    t1 = min(t1, x.size / fs - 0.05)
    t0 = min(t0, 0.5 * t1)
    seg = x[int(t0 * fs):int(t1 * fs)]
    w = sg.windows.blackmanharris(seg.size) * seg
    est, wts = [], []
    # hum harmonic-ratios
    for h in (1, 3, 5, 7, 9):
        fh = HUM_F * h
        # skip if line to close to fundamental
        if abs(fh / f0 - round(fh / f0)) * f0 < 3.0 or fh > 0.45 * fs:
            continue
        grid = np.linspace(fh - 0.1 * h, fh + 0.1 * h, 401)
        # energy for every freq on grid
        E = np.abs(_dtft(w, fs, grid))
        # index of highest score
        j = int(np.argmax(E))
        # append base freq
        est.append(grid[j] / h)
        # store energy for the candidate 
        wts.append(E[j])
    if not est:
        return HUM_F
    # weighted average
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
    """legendre polynomial with steady ramp U and S shape. The P+1 volume shapes (flat, ramp, U, S, ...) over the window, one per column. 
    Mixed together they can draw any slow change in volume."""
    # rescale time axis to be between -1 and +1 (needed for legendre polynomial)
    tau = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
    return np.stack([np.polynomial.legendre.Legendre.basis(p)(tau) for p in range(P + 1)], 1)


def line_cols(t, f_rel, P):
    """Columns e^{i2pi f t} * Legendre_p(t) for each f in f_rel (Hz, relative to fc). One line = a tone at frequency f whose volume may change slowly. For each f, build P+1 versions of the tone (steady, ramp, U, S, ...); a fit later finds how much of each to mix."""
    if len(f_rel) == 0:
        return np.zeros((t.size, 0), complex)
    # create the four shapes
    L = legendre_cols(t, P)
    # create tones with constant volume at given frequencies
    E = np.exp(2j * np.pi * np.outer(t, f_rel))
    # compute tones with shapes
    return (E[:, :, None] * L[:, None, :]).reshape(t.size, -1)


def _orth(A):
    """Rewrite the columns as a non-overlapping set that spans the same signals (exact duplicates dropped), so removing them from the audio is one line: r = y - Q @ (Q^H y)."""
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
    # dicts: delta, floor, P, ks (lines that get nuisance columns) and
    # lines {k: (f, amp, snr_db, resolved)}
    combs: list = field(default_factory=list)
    floor: float = np.nan


def comb_lines(band, delta):
    """k and frequency of every line (k+delta) f0 of one comb inside the modelled region."""
    # how many ks
    kmax = int(np.ceil(band.hi / band.f0)) + 2
    ks = np.arange(0, kmax)
    # freqs in comb
    f = (ks + delta) * band.f0
    # only keep f in band
    m = (f > band.lo) & (f < band.hi)
    return ks[m], f[m]


def nuisance(band):
    """Orthonormal basis: harmonics, hum and the visible lines of every comb found so far,
    each comb with its own envelope order. Lines under the floor get no columns: they
    carry nothing, and every column costs degrees of freedom the scan needs."""
    cols = [band.base]
    for c in band.combs:
        fs_ = np.array([(k + c["delta"]) * band.f0 for k in c["ks"]])
        cols.append(line_cols(band.t, fs_ - band.fc, c["P"]))
    return _orth(np.hstack(cols))


def _scan(band, r):
    """Spectrum of r as a score per frequency, *without* the nuisance correction of the atom
    norm: |a^H r|^2 / ||a||^2. Since r is orthogonal to the nuisance this never exceeds the
    exact score |a^H r|^2 / ||P_perp a||^2; the two differ only next to nuisance lines.
    Used to find peaks; `exact` scores the chosen cells.
    """
    t = band.t - band.t[0]
    nfft = int(2 ** np.ceil(np.log2(8 * t.size)))
    f = np.fft.fftfreq(nfft, 1 / band.fs_d)
    sel = np.abs(f) <= band.half_bw
    # sort freqs
    order = np.argsort(f[sel])
    f_rel = f[sel][order]
    # calculate energy for each freq and normalizes
    S = np.abs(np.fft.fft(r, nfft)[sel][order]) ** 2 / t.size
    return f_rel, S


def exact(band, Q, r, f_abs):
    """Is there a line at f_abs (Hz), how loud is it, and can the answer be trusted?

    Holds a test tone at f_abs against the residual r. Near a harmonic the nuisance
    subtraction has already taken part of anything there, so the match is scaled up by
    how much of the test tone survives that subtraction (den).
    Returns (score, amplitude, trust): score is in noise-floor units (white noise of
    power sigma^2 scores sigma^2 on average); trust = den / na is the surviving fraction,
    1 = far from any harmonic, near 0 = hidden under one."""
    t = band.t - band.t[0]
    a = np.exp(2j * np.pi * (f_abs - band.fc) * t)
    na = float(t.size)
    den = na - np.sum(np.abs(Q.conj().T @ a) ** 2) if Q.shape[1] else na
    den = max(den, 1e-9 * na)
    num = np.vdot(a, r)
    return abs(num) ** 2 / den, num / den, den / na


def make_band(x, fs, f0, fc, half_bw, i_on, T, f_hum, track=None):
    """Zoom in on the frequency slice fc +- half_bw and prepare what is known in it.

    1. Cut the slice out: shift fc down to 0 Hz, low-pass away everything else, and keep
       only every d-th sample. After the filter the fastest wiggle left is at the edge of
       the slice, so far fewer samples describe it just as well (D3: 14400 -> 1600),
       and everything downstream is that much faster.
    2. Build the columns of the harmonics and hum in the slice (the nuisance base), each
       with a slowly changing volume, harmonics phase-locked to the fundamental.

    The window starts after the onset by the larger of 5 ms and the FIR half-length, so
    the filtered hammer impact does not leak into it. The FIR runs over the file from its
    start: the silence before the onset keeps its edge clean.
    """
    guard = max(1.5 * f0, 30.0)
    n = int(T * fs)
    # prepare filter for carving out the slice 
    ntaps, beta = sg.kaiserord(100.0, guard / (fs / 2))
    ntaps |= 1
    # start after the initial hammer transient
    i0 = i_on + max(int(0.005 * fs), ntaps // 2)
    taps = sg.firwin(ntaps, half_bw + 1.5 * guard, fs=fs, window=("kaiser", beta))
    # shift target freq down to 0 Hz  
    stop = min(x.size, i0 + n + ntaps)
    xm = x[:stop] * np.exp(-2j * np.pi * fc * np.arange(stop) / fs)
    # apply the filter 
    yf = sg.fftconvolve(xm, taps, mode="same")
    # keep only every d-th sample to speed up process 
    d = max(1, int(np.floor(fs / (2.5 * (half_bw + 2 * guard)))))
    y = yf[i0:i0 + n:d]
    fs_d = fs / d
    # time of each kept sample
    t = (i0 - i_on) / fs + np.arange(y.size) / fs_d
    # PREPARE POTENTIAL SUBTRACTIONS 
    # range where known things like harmonics get subtracted
    lo, hi = fc - half_bw - 2 * guard, fc + half_bw + 2 * guard
    # which harmonics are in that range
    ks = np.arange(int(np.ceil(lo / f0)), int(np.floor(hi / f0)) + 1)  
    # which hum lines are in the range
    hs = np.arange(int(np.ceil(lo / f_hum)), int(np.floor(hi / f_hum)) + 1)
    hs = hs[(hs != 0) & (np.abs(hs) <= HUM_H_MAX)]
    # create the "tone version" which volumes may change
    H = line_cols(t, ks * f0 - fc, p_harm(T))
    # each harmonic k follows the fundamentals pitch drift
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
    """How loud the noise is in the band, as one number (same units as the exact score).

    The spectrum of the residual is a noise carpet with a few tall line spikes. Sort it,
    drop the top 20 % (the spikes), average the rest, and scale up by 1/TRIM_BIAS,
    because dropping the top also removed the naturally high parts of the noise.
    The Hann window keeps each line's energy near the line. One number for the whole
    band: a local floor would be pushed up by the scan's side-bumps next to lines.
    """
    t = band.t - band.t[0]
    w = sg.windows.hann(t.size)
    nfft = int(2 ** np.ceil(np.log2(4 * t.size)))
    P = np.abs(np.fft.fft(r * w, nfft)) ** 2 / np.sum(w ** 2)
    f = np.fft.fftfreq(nfft, 1 / band.fs_d)
    P = np.sort(P[np.abs(f) <= band.half_bw])
    return float(P[:int(TRIM * P.size)].mean() / TRIM_BIAS)


def _comb_energy(band, f_rel, S, delta):
    """Summed spectral energy of a comb's lines (nearest grid cell)."""
    # frequency spacing 
    df = f_rel[1] - f_rel[0]
    # get all lines in comb 
    _, fs_ = comb_lines(band, delta)
    # keep only lines in search region
    fs_ = fs_[np.abs(fs_ - band.fc) <= band.half_bw]
    # for each line find nearest point in spectrum 
    j = np.clip(np.round((fs_ - band.fc - f_rel[0]) / df).astype(int), 0, f_rel.size - 1)
    # sum energy
    return S[j].sum()


def pursue(band):
    """Find combs one at a time until nothing stands MIN_SNR_DB over the floor.

    Strongest first, so a sidelobe never outranks its line. A comb that keeps coming back
    after its envelope has been loosened is left as it is and its lines are masked out
    of the scan (CLEAN-style), so one imperfect removal cannot stall the search.
    """
    # list of regions to ignore
    masked = []                 # (f_lo, f_hi) absolute Hz                      
    for _ in range(MAX_COMBS):
        # subtract everything known (hum, harmonics ...)
        Q = nuisance(band)
        r = band.y - Q @ (Q.conj().T @ band.y)
        # take spectrum of what is left and blanks out ignored regions
        f_rel, S = _scan(band, r)
        s = S.copy()
        f_abs = band.fc + f_rel
        for a, b in masked:
            s[(f_abs > a) & (f_abs < b)] = 0.0
        # measure noise level
        band.floor = floor = noise_floor(band, r)
        # get highest remaining peak
        j = int(np.argmax(s))
        Sj, _, _ = exact(band, Q, r, f_abs[j])
        # check if remaining highest peak is loud above the noise
        if 10 * np.log10(max(Sj, 1e-300) / floor) < MIN_SNR_DB:
            break
        # delta from the whole comb: maximise its summed energy
        df = f_rel[1] - f_rel[0]
        d0 = (f_abs[j] / band.f0) % 1.0
        cand = d0 + np.arange(-3, 4) * df / band.f0
        E = np.array([_comb_energy(band, f_rel, S, d) for d in cand])
        # keep estimation with highest energy
        delta = float(cand[int(np.argmax(E))] % 1.0)
        # Within ~2.5/T Hz of a comb already found, a "new" comb is that comb's leftover
        # skirt (its sideband envelopes are not a clean exponential where the pickup is
        # strongly non-linear), and it would replicate on every k. Loosen the old one.
        tol = max(DUP_CELLS / (band.T * band.f0), 0.002)
        dup = [c for c in band.combs if _dist(c["delta"], delta) < tol]
        # new comb record, then check every tooth like a checklist
        comb = {"delta": delta, "floor": floor, "lines": {}, "P": P_COMB, "ks": []}
        w = max(1, int(round(0.25 / (band.T * df))))          # +- 1/(4T) Hz
        # where the teeth should be
        ks, fs_ = comb_lines(band, delta)
        for k, fk in zip(ks, fs_):
            # skip teeth outside the search region
            if abs(fk - band.fc) > band.half_bw:
                continue
            # go to the expected position and take the highest point nearby
            jj = int(round((fk - band.fc - f_rel[0]) / df))
            a, b = max(0, jj - w), min(s.size, jj + w + 1)
            if b <= a:
                continue
            m = a + int(np.argmax(s[a:b]))
            # measure the tooth and write it down
            Sm, Cm, Dm = exact(band, Q, r, f_abs[m])
            snr = float(10 * np.log10(max(Sm, 1e-300) / floor))
            comb["lines"][int(k)] = (f_abs[m], float(abs(Cm)), snr, float(Dm))
            # strong teeth (>= 9 dB over noise) get subtracted from the next round on
            if snr >= MIN_SNR_DB - 3:
                comb["ks"].append(int(k))
        if dup:
            # The same comb again. Either lines that were under the (then higher) floor
            # have surfaced give them columns or its envelope was too stiff.
            old = dup[0]
            new = set(comb["ks"]) - set(old["ks"])
            # new strong teeth appeared: add them to the old comb
            if new:
                old["ks"] = sorted(set(old["ks"]) | new)
                for k in new:
                    old["lines"][k] = comb["lines"][k]
            # none new: let the old comb's volume curve bend more
            elif old["P"] < P_COMB + 4:
                old["P"] += 2
            # still coming back: give up and ignore its teeth and this peak
            else:
                hw = 1.0 / band.T
                masked += [((k + old["delta"]) * band.f0 - hw, (k + old["delta"]) * band.f0 + hw)
                           for k in old["ks"]]
                masked.append((f_abs[j] - hw, f_abs[j] + hw))
            continue
        # no strong teeth: probably junk, ignore this peak from now on
        if not comb["ks"]:
            masked.append((f_abs[j] - 1.0 / band.T, f_abs[j] + 1.0 / band.T))
            continue
        # a real new comb: gets subtracted in the next round
        band.combs.append(comb)
    return band


RESOLVED = 0.5            # a line whose atom keeps less than half its norm outside the
                          # nuisance span is mostly harmonic: its amplitude is not measurable


def refine_mode(band, ci, n, J=3, n_f=61, n_lam=41):
    """Exact frequency, decay and loudness of the mode tooth n of comb ci.

    The only place decay is measured. Guessing game: for many (frequency, decay) pairs,
    build a test tone that fades exactly like that and see how well it matches the
    recording; the best pair wins. The mode tooth and its visible neighbours (up to J
    each side) are matched together, since all teeth of one mode share one offset and
    one decay. Grid instead of an optimiser: for short-lived lines the score can have
    several bumps along the decay axis. Coarse grid (f +- 1/T, lam 0..LAM_MAX), then a
    fine grid around the best cell.
    Returns f (Hz), lam (1/s), loudness of the tooth at window start, teeth used.
    """
    comb = band.combs[ci]
    # teeth to match: the mode tooth plus its clearly visible neighbours
    prof = comb_profile(band, comb)
    js = [0] + [j for j in range(-J, J + 1) if j and n + j in prof and prof[n + j][1]]
    # subtract everything except this comb, so y holds the comb plus noise
    saved = comb["ks"]
    comb["ks"] = []
    Q = nuisance(band)
    comb["ks"] = saved
    t = band.t - band.t[0]
    y = band.y - Q @ (Q.conj().T @ band.y)

    def grid(fg, lg):
        """Score every (frequency in fg, decay in lg) pair; return the best one."""
        total = np.zeros((lg.size, fg.size))
        carrier = np.zeros((lg.size, fg.size), complex)
        for j in js:
            # test tones at every guessed frequency, shifted to tooth j
            E = np.exp(2j * np.pi * np.outer(t, fg + j * band.f0 - band.fc))
            for il, lam in enumerate(lg):
                # let them fade at the guessed decay
                A = E * np.exp(-lam * t)[:, None]
                # drop the part the subtraction would take anyway (as in exact)
                A = A - Q @ (Q.conj().T @ A)
                na = np.sum(np.abs(A) ** 2, 0)
                num = A.conj().T @ y
                # how well each test tone matches, added up over the teeth
                total[il] += np.abs(num) ** 2 / na
                if j == 0:
                    carrier[il] = num / na
        il, jf = np.unravel_index(int(np.argmax(total)), total.shape)
        return fg[jf], lg[il], carrier[il, jf]

    # coarse pass: +- 1/T around the tooth, decay 0..LAM_MAX
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
    for k, (f, amp, snr, frac) in comb["lines"].items():
        if frac < RESOLVED:
            continue
        # fix higher freq = louder by dividing through the f
        lev = 20 * np.log10(amp / f)
        # only keep lines above the minimum snr
        if snr >= MIN_SNR_DB:
            prof[k] = (lev, True)
        else:
            prof[k] = (lev - snr + MIN_SNR_DB, False)     # what MIN_SNR_DB would have needed
    return prof


SYM_W = (1.0, 0.5, 0.25)  # weight of pair j = 1, 2, 3: the nearest sidebands are the loudest
SYM_CLIP_DB = 12.0        # one pair spoilt by a leaking harmonic must not decide alone


def symmetry(prof, n):
    """Weighted mean |L(n+j) - L(n-j)| (dB) over the pairs where one line is visible.

    Two lines under the floor carry no information and are skipped; one visible and one
    under the floor counts only if the visible one is the louder (else they could be equal).
    """
    d, w = [], []
    # compare the pairs n+1 vs n-1, n+2 vs n-2, n+3 vs n-3
    for j, wj in enumerate(SYM_W, start=1):
        # skip the pair if one tooth is missing (untrusted or outside the band)
        if n + j not in prof or n - j not in prof:
            continue
        # a = right tooth, b = left tooth; va / vb = visible (measured) or only "at most"
        (a, va), (b, vb) = prof[n + j], prof[n - j]
        # both measured: their gap
        if va and vb:
            dj = abs(a - b)
        # one measured and louder than the other's limit: gap is at least this
        elif va and a > b:
            dj = a - b
        elif vb and b > a:
            dj = b - a
        # otherwise they could be equal: no information
        else:
            continue
        # cap each pair so one spoilt pair cannot decide alone
        d.append(min(dj, SYM_CLIP_DB))
        w.append(wj)
    # weighted mean gap (0 = symmetric) and pairs used; no pairs also gives 0.0,
    # so check the count: 0 pairs means "no information", not "perfectly symmetric"
    return (float(np.average(d, weights=w)) if d else 0.0), len(d)


def choose_carrier(band, comb, mu, sigma):
    """Candidate k with (k + delta) in mu +- 3 sigma: the one about which the comb is most
    symmetric, among those that are visible themselves. Returns dict or None."""
    prof = comb_profile(band, comb)
    # candidates: visible teeth inside Gabrielli's expected range mu +- 3 sigma
    cands = [k for k in prof if prof[k][1] and abs(k + comb["delta"] - mu) <= 3 * sigma]
    # no candidate: this comb is not the mode
    if not cands:
        return None
    vis = {k: v[0] for k, v in prof.items() if v[1]}
    rows = []
    for n in cands:
        # how lopsided the comb is around this tooth
        asym, npairs = symmetry(prof, n)
        # how much louder than its loudest visible neighbour (not used for the ranking)
        others = [vis[k] for k in vis if 0 < abs(k - n) <= len(SYM_W)]
        margin = vis[n] - max(others) if others else np.inf
        # distance from Gabrielli's mean in sigma
        z = (n + comb["delta"] - mu) / sigma
        rows.append(dict(k=n, asym=asym, npairs=npairs, margin=margin, z=z))
    # penalty = asymmetry + 0.5 z^2, lowest wins; with 0 pairs asym is 0 for all,
    # so the tooth closest to Gabrielli's mean wins
    rows.sort(key=lambda r: (r["asym"] + 0.5 * r["z"] ** 2))
    best = rows[0]
    # penalties of winner and runner-up: a big gap means a clear decision
    best["runner_up"] = rows[1]["asym"] + 0.5 * rows[1]["z"] ** 2 if len(rows) > 1 else np.inf
    best["score"] = best["asym"] + 0.5 * best["z"] ** 2
    return best


# %%
# ---------- one recording ----------

HALF_SIDE = 3.2           # band reaches this many f0 past the prior window, for sidebands
MAX_ASYM_DB = 6.0         # a mode's comb is symmetric about it; worse than this is not a comb
MIN_MARGIN_DB = -10.0     # carrier vs loudest other line of its comb. Where the tine swings
                          # across the pole the carrier drops below its sidebands (~ -3 dB in
                          # the synthetic bass), but not by tens of dB: that is leakage.


def analyze_recording(x, fs, f0_guess, modes=MODES, name=("", "")):
    """All modes of one recording. Returns (list of dicts, context dict)."""
    # when the hammer hits, and how long to look (0.3-1.5 s)
    i_on = onset(x)
    T = window_len(f0_guess)
    i5 = i_on + int(0.005 * fs)
    # get exact f0
    f0 = refine_f0(x[i5:i5 + int(T * fs)], fs, f0_guess)
    # determine hum freq
    f_hum = hum_freq(x, fs, f0)
    # pitch drift of the fundamental
    track = PhaseTrack(x, fs, f0, i_on, 0.005, T + 0.12)

    out = []
    # one row per mode: found with its numbers, or not found with a reason ("why")
    for name_m, (mu, sigma) in modes.items():
        row = dict(note=name[0], dyn=name[1], mode=name_m, mu=mu, f0=f0, found=False, T=T)
        # Skip only if the expected range itself is above F_MAX (Gabrielli: nothing above
        # ~10 kHz). The sideband margin on top is trimmed to what the sample rate allows.
        if (mu - 3 * sigma) * f0 > F_MAX:
            row["why"] = "above F_MAX"
            out.append(row)
            continue
        # slice where the mode is expected (mu +- 3 sigma), plus room for its neighbour teeth
        guard = max(1.5 * f0, 30.0)
        lo_b = (mu - 3 * sigma - HALF_SIDE) * f0
        hi_b = min((mu + 3 * sigma + HALF_SIDE) * f0, 0.45 * fs - 2 * guard)
        # find all combs in that slice
        band = pursue(make_band(x, fs, f0, 0.5 * (lo_b + hi_b), 0.5 * (hi_b - lo_b), i_on, T, f_hum,
                                track))
        row["n_combs"] = len(band.combs)
        # keep only plausible combs
        picks = []
        for ci, comb in enumerate(band.combs):
            # best tooth of this comb; None = no tooth in the expected range
            ch = choose_carrier(band, comb, mu, sigma)
            if ch is None:
                continue
            # too lopsided, or the tooth is much quieter than its neighbours
            if ch["asym"] > MAX_ASYM_DB or ch["margin"] < MIN_MARGIN_DB:
                continue
            # inside a harmonic's envelope bandwidth a line cannot be told from the harmonic
            if min(comb["delta"], 1 - comb["delta"]) * f0 < (p_harm(T) + 1) / (2 * T):
                continue
            # compare combs by carrier level, not SNR: the floor at detection drops as
            # combs are removed, so later (weaker) combs would look better
            f_k, amp_k, snr = comb["lines"][ch["k"]][:3]
            picks.append((20 * np.log10(amp_k / f_k), snr, ci, ch))
        # nothing plausible left: not found
        if not picks:
            row["why"] = "no comb centred in the prior window"
            out.append(row)
            continue
        # the loudest survivor is the mode
        _, snr, ci, ch = max(picks, key=lambda p: p[0])
        # measure it exactly: frequency, decay, loudness
        f, lam, amp, js = refine_mode(band, ci, ch["k"])
        # dies away almost at once: the hammer, not a mode
        if lam > 0.95 * LAM_MAX:
            row["why"] = "decay at the limit: an impact transient"
            out.append(row)
            continue
        # fundamental at the window start, from the phase track (fitted with harmonics 2
        # and 3 over the whole window, so they do not leak into it)
        a1 = 2 * abs(track.envelope(band.t[:1])[0])
        # write down the result: frequency, decay, loudness re the fundamental, confidence
        row.update(found=True, ratio=f / f0, f=f, lam=lam, lam_db=8.686 * lam,
                   amp_db=20 * np.log10(2 * amp / a1), snr_db=snr, delta=band.combs[ci]["delta"],
                   k=ch["k"], asym_db=ch["asym"], npairs=ch["npairs"], margin_db=ch["margin"],
                   score=ch["score"], runner_up=ch["runner_up"],
                   t_start=float(band.t[0]), n_candidates=len(picks), n_lines=len(js))
        out.append(row)
    # in-between values, for inspection
    ctx = dict(f0=f0, f_hum=f_hum, T=T, track=track)
    return out, ctx


# %%
# ---------- the whole keyboard ----------

DYNS = ("p", "mp", "mf", "f")


def _job(args):
    note, dyn, f0_guess = args
    import warnings
    warnings.filterwarnings("ignore")
    x, fs = load(note, dyn)
    rows, _ = analyze_recording(x, fs, f0_guess, name=(note, dyn))
    return rows


def run_all(workers=None):
    from concurrent.futures import ProcessPoolExecutor
    d = np.load("fundamentals.npz", allow_pickle=True)
    notes, dyns, table = list(d["notes"]), list(d["dyns"]), d["table"]
    jobs = [(n, dy, float(table[i, dyns.index(dy)]["f"])) for i, n in enumerate(notes) for dy in DYNS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(_job, jobs)):
            rows += res
            print(f"{i + 1}/{len(jobs)} {jobs[i][0]}-{jobs[i][1]}", flush=True)
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
                       dyns="")
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
                           spread_hz=float(np.ptp(fs_)), dyns=",".join(g["dyn"] for g in best))
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


# Runs the analysis only if there is no table yet; delete tine_modes.npz to redo it.
if __name__ == "__main__":
    import os
    if not os.path.exists(TABLE_FILE):
        save_table(summarise(run_all()))


# %%
# ---------- keyboard plot, in the layout of Gabrielli et al. 2020, Fig. 20 ----------


def plot_keyboard(path=TABLE_FILE):
    import matplotlib.pyplot as plt

    d = np.load(path, allow_pickle=True)
    f_e0 = f0_nominal("E0")
    key = np.array([round(12 * np.log2(f0_nominal(n) / f_e0)) + 1 for n in d["notes"]])
    ratio, conf = d["ratio"], d["confirmed"].astype(bool)
    single = np.isfinite(ratio) & ~conf

    fig, ax = plt.subplots(figsize=(10, 8))
    for mu, _ in MODES.values():
        ax.axhline(mu, ls="--", color="0.4", lw=1.2, zorder=1)
    k = np.linspace(1, 73, 400)
    f0_k = f_e0 * 2.0 ** ((k - 1) / 12)
    ax.plot(k, 10e3 / f0_k, ":", color="k", lw=2, label="10 kHz (as in the paper)")
    ax.plot(k, F_MAX / f0_k, ":", color="0.4", lw=1, label=f"{F_MAX / 1e3:.0f} kHz (our search limit)")
    kk = np.broadcast_to(key[:, None], ratio.shape)
    ax.scatter(kk[single], ratio[single], s=45, facecolors="none", edgecolors="0.55", lw=1.2,
               zorder=2, label="one dynamic only")
    ax.scatter(kk[conf], ratio[conf], s=45, color="#3b6fd4", zorder=3,
               label="confirmed ($\\geq$2 dynamics)")

    ax.set_xlim(0, 74)
    ax.set_ylim(0, 100)
    ax.set_xlabel("key")
    ax.set_ylabel("f0 ratio")
    ax.grid(axis="x", color="0.92")
    top = ax.secondary_xaxis("top")
    top.set_xticks(np.arange(1, 74, 12), [f"E{i}" for i in range(7)])
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Tine modes over the keyboard (our recordings), "
                 "in the layout of Gabrielli et al. 2020, Fig. 20")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_keyboard()

# %%
