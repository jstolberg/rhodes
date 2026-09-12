# %% [markdown]
# # Fitting the model
# The model splits into three parts with seperate parameters: 
# The tine, the pickup and the hammer. Assuming an $M$ keys piano
# and $N$ modes per key we
# have the following parameters.
# 
# | Group | Symbol | Parameter | Scope | No. |
# | --- | :---: | --- | --- | --- |
# | Tine   | $f_n$ | modal frequency | per mode | $N \times M$ |
# |  -     | $\sigma_n$ | modal decays | per mode | $N \times M$ |
# |  -     | $c_n$ | modal excitation coefficients | per mode | $N \times M$ |
# | Pickup | m | slope steepness | Global | 1 |
# | -      | a | half peak width  | Global | 1 |
# | -      | $p_d$ | pickup distance  | per key | $M$ |
# | -      | $p_o$ | pickup offset | per key | $M$ |
# | Hammer | $\tau_0$ | contact time at max velocity | per key | $M$ |
# | - | $\beta$ | material compression factor | Global | 1 |
# 
# Except for the fundamental frequency $f_0$ all of these parameters are free
# and have to be fittet. As several parameters contribute to the same measured 
# features of the the output signal (e.g. amplitude of fundamental depends on 
# the pickup as well as $c_n$, $\beta$ and $\tau_0$), it is necessary to isolate 
# parameters and optimize step-wise. We proceed with further details in the
# following sections, but in short we suggest these steps:
# 
# 1. Identify exact fundamental frequency $f_0$ for all keys.
# 2. Fit pickup parameters $p_d$, $p_0$, $m$ and $a$, together with a free
# oscilation amplitude $A_0$ and decay $\sigma_0$ for the fundamental of each key.
# Note that $A_0$ is not a model parameter, but an intermediate step. 
# 3. Fit $\tau_0$ and $\beta$ with fixed pickup and fundamental, using 
# signal amplitudes (the $A_0$ of previous step) at different velocities, 
# considering their ratios in order to cancel out $c_0$.
# 4. Fit $c_0$ with fixed $\tau_0$ and $\beta$.
# 5. Identify inharmonic modes $f_n$.
# 6. Fit $c_n$ and $\sigma_n$ for inharmonic modes, with the rest of the model fixed. 
# 
# As different time and frequency segments of the sampled signal is containing
# information relevant to different parts of this optimization process, we split 
# an imagined spectogram of the sample into named areas below for further reference.
# The horizontal axis denotes time, and the vertical is the frequency spectrum.
# By H0 we denote the fundamental $f_0$ and let $H1$, $H2$, etc. denote its harmonics
# $2f_0$, $3f_0$, etc. Some areas are left unnamed, as these carry no relevant information. 
# 
# | | Contact | Inharmonic decay | Harmonic decay | Fundamental decay |
# | :---: | :---: | :---: | :---: | :---: | 
# | $>H6$ | S1 | S2 | - | - |
# | $H0-H6$ | S3 | S4 | S5 | - | 
# | $\leq H0$ | S6 | S7 | S8 | S9 | 
# | Approx. duration: | 1-2ms | $<1s$ | $<4s$ | $>4s$ |
# 
# %% [markdown]
# ## 1. Fitting $f_0$
# See mode_measurement.py on branch niklas/dev.
# 
# ## 2. Fitting pickup and fundamental
# The pickup produces only harmonic overtones, whose decay depends on the 
# decay of the fundamental. As it has been seen in previous reseach that all 
# all inharmonic modes are either above $H7$ or below $H0$, considering only
# S4 and S5 can give us a clear view of the amplitude, decay and non-linearity 
# applied to the fundamental in near isolation. This means that we can fit the
# pickup and fundamental parameters, without knowing anything about the inharmonic
# modes and the hammer excitation. 
# 
# We suggest using a MR STFT loss on the forward pass in this area against a real
# sample. Note that we are **not** fitting any frequencies, only amplitude and decay,
# so we expect gradient should behave well. Furthermore, we train against all 
# samples simultaneously, in order to seperate otherwise entangled parameters, 
# like $A_0$ and $p_d$, which both affect output amplitude, but differ in scope
# ($A_0$ depends on velocity, $p_d$ does not).
# 
# ## 3. Fitting $\tau_0$ and $\beta$
# From step 2, we achieve from each key and velocity an amplitude $A_0(v)$, which
# signify amplitude the idealised free oscilation at $t=0$. As this parameter is 
# directly handed over by the hammer model, when transitioning to free decay, we 
# can create forward passes and measure this amplitude against the fitted amplitude
# of data samples. Sadly, $A_0$ depends both on $c_0$ and $\tau_0$, and thus fitting
# against data samples could lead to arbitrary results for these parameters.
# 
# However, as $c_0$ is a multiplicative factor equal at all velocities, while 
# contact time is velocity dependent, we can fit $\tau(v)$ and $\beta$ by
# by evaluating output amplitudes at different velocities, and taking their ratio:
# Denote by $V_0(v, \beta, \tau_0, c_0)$ the amplitude of the fundamental
# of the output signal $\epsilon(t)$ at a fixed $t$. Defining
# $$R(\beta, \tau_0) = \frac{V_0\big|_{v=v_1}}{V_0\big|_{v=v_2}}$$
# Since the multiplicative factor $c_0$ is unchanced, it cancels out and 
# the ratio $R$ no longer depends on it. Simple RMS loss can be taken on these
# ratios between synthsized and data samples, in order to fit $\beta$ and $\tau_0$
# with backpropagation. 
# 
# ## 4. Fitting $c_0$
# With pickup and hammer excitation parameters fixed in step 2 and 3, $c_0$ 
# remains the only free, unfitted parameter of relevance to the fundamental, 
# and can be fitted by direct comparison of output amplitudes, e.g. RMS loss 
# on amplitude of exponential decay fits of the fundamental mode.
# 
# ## 5. Identify inharmonic modes $f_n$
# TBD
# 
# Suggestion: Same modal analysis as in step 1.  
# 
# ## 6. Fit $c_n$ and $\sigma_n$ for inharmonic modes
# TBD 
# 
# Suggestion: Since the full model is now fitted except for these parameters, 
# we can fit them using MR STFT on regions S2 and S7.  

# %% [markdown]
# # Evaluatation measures
# For each step above we first evaluate the fitting procedure and choice of loss,
# by first attempting to fit synthetic samples produced by the same model with 
# known parameters; It the optimization fails at this, we will know that the loss
# or gradient is unsuited for optimizing the relevant parameters, and we will need
# to reevaluate. This can also be used to evaluate hyperparameters, e.g. the 
# segmentation of the sample spectrogram, as well as evaluate if optimizing, e.g. 
# step 2 is stabile even in the presence of unknown inharmonic overtones and 
# subfundamentals. 

# %%
