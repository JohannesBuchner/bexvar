#!/usr/bin/env python3

"""
Continuous poisson log-autoregressive model for eROSITA light curves.

The model assumes a autoregressive model for the instantaneous log-count rate
in each time bin. For the background, each time bin is estimated independently.

The autoregressive model can take care of lightcurve gaps. Its assumption
is that the power spectrum transitions from white noise with some amplitude (sigma)
to a powerlaw with index -2, at a characteristic dampening time-scale (tau).

sigma, tau and the mean count rate (c) are the most important parameters of this model.

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import log, pi, cos, exp
import scipy.fftpack
import sys
import tqdm
from astropy.table import Table
import stan_utility


model = stan_utility.compile_model_code("""
data {
  int N;

  // duration of time bin
  real dt;
  // number of time step between end of previous time bin and end of current time bin
  // this may be different from one when there are gaps
  real<lower=0> tsteps[N-1];
  int<lower=0> z_obs[N];
  int<lower=0> B_obs[N];
  vector[N] Barearatio;
  vector[N] countrate_conversion;
  vector[N] Bcountrate_conversion;

  real prior_logBc_mean;
  real prior_logBc_std;
  real prior_logc_mean;
  real prior_lognoise_mean;
  real prior_logc_std;
  real prior_lognoise_std;
  real prior_logtau_mean;
  real prior_logtau_std;
}
transformed data {
  vector[N] logBarearatio = log(Barearatio);
  vector[N] logcountrate_conversion = log(countrate_conversion);
  vector[N] logBcountrate_conversion = log(Bcountrate_conversion);
  real logminduration = log(dt);
  real T = sum(tsteps);
  real logT = log(T);
}
parameters {
  // regression time-scale
  real logtau;
  // mean
  real logc;
  // sigma
  real lognoise;
  // unit normal noise deviations
  vector[N] W;

  // intrinsic background count rate at each time bin
  vector[N] logBy;
}
transformed parameters {
  // linear transformations of the log parameters
  real phi;
  real tau;
  real noise;
  real c;
  // intrinsic source count rate at each time bin
  vector[N] logy;

  // transform parameters to linear space
  tau = exp(logtau);
  noise = exp(lognoise);
  c = exp(logc);
  phi = exp(-dt / tau);

  // apply centering to real units, AR(1) formulas:
  logy[1] = logc + W[1] * noise;
  for (i in 2:N) {
    logy[i] = logc + exp(-tsteps[i-1] / tau) * (logy[i-1] - logc) + W[i] * noise * tsteps[i-1] / dt;
  }
}
model {
  // correct for observing efficiency
  vector[N] logysrcarea = logy + logcountrate_conversion;
  vector[N] logybkgarea = logBy + logBarearatio + logBcountrate_conversion;
  vector[N] logytot;// = logysrcarea;
  // sum source and background together
  for (i in 1:N) {
    logytot[i] = log_sum_exp(logysrcarea[i], logybkgarea[i]);
  }

  logBy ~ normal(prior_logBc_mean, prior_logBc_std);
  B_obs ~ poisson_log(logBy + logBcountrate_conversion);

  // source AR process priors:
  logtau ~ student_t(2, prior_logtau_mean, prior_logtau_std);
  logc ~ normal(prior_logc_mean, prior_logc_std);
  lognoise ~ normal(prior_lognoise_mean, prior_lognoise_std);
  logy ~ normal(0, 5); // stay within a reasonable range
  W ~ std_normal();
  // comparison to total source region counts
  z_obs ~ poisson_log(logytot);
}
""")


np.random.seed(1)

filename = sys.argv[1]


lc_all = Table.read(filename, hdu='RATE', format='fits')
nbands = lc_all['COUNTS'].shape[1]
band = 0
if len(sys.argv) > 2:
    band = int(sys.argv[2])

print("band %d" % band)
lc = lc_all[lc_all['FRACEXP'][:,band] > 0.1]
bc = lc['BACK_COUNTS'][:,band]
c = lc['COUNTS'][:,band]
bgarea = 1. / lc['BACKRATIO']
# length of the time bin
dt = lc['TIMEDEL']
assert dt.max() == dt.min()
dt = dt.min()
# TIME is the mid point of the light curve bin
# here we want the starting point:
x_start = lc['TIME'] - lc['TIME'][0] - lc['TIMEDEL'] / 2.0
x_end   = lc['TIME'] - lc['TIME'][0] + lc['TIMEDEL'] / 2.0
x = lc['TIME'] - lc['TIME'][0]
tsteps = (x_end[1:] - x_end[:-1])
assert (tsteps > 0).all(), np.unique(tsteps)
assert (bc.astype(int) == bc).all(), bc
prefix = sys.argv[1] + '-%d-cplar1b' % band
fe = lc['FRACEXP'][:,band]
rate_conversion = fe * dt

N = len(x_start)
data = dict(
    # provide observations
    N=N, z_obs=c, B_obs=bc.astype(int),
    # additional information about the sampling:
    dt=dt, Barearatio=1. / bgarea, tsteps=tsteps,
    # source count rate is modulated by fracexp
    countrate_conversion=rate_conversion,
    # background count rate is modulated by fracexp, except in the hard band,
    # where it is assumed constant (particle background dominated)
    Bcountrate_conversion=rate_conversion*0 + 1 if band == 2 else rate_conversion,
    # background count rates expected:
    prior_logBc_mean=0, prior_logBc_std=np.log(10),
    # source count rates expected:
    prior_logc_mean=0, prior_logc_std=5,
    # expected noise level is 10% +- 2 dex
    prior_lognoise_mean=np.log(0.1), prior_lognoise_std=np.log(100),
    #prior_logtau_mean=np.log(x.max()), prior_logtau_std=np.log(x.max() / (dt)),
    prior_logtau_mean=np.log(dt), prior_logtau_std=np.log(100),
)

def init_function(chain=None):
    # guess good parameters for chain to start
    guess = dict(
        # start with short time-scales: all bins independent
        logtau=np.log(dt / 2),
        # background count rates; estimated from background counts
        logBy=np.array(np.log((bc + 0.1) / data['Bcountrate_conversion'])),
        # average count rate; estimated from counts
        logc=np.log((c + 0.1) / data['countrate_conversion']).mean(),
        # minimal noise
        lognoise=np.log(1e-10),
        W=np.random.normal(size=N),
    )
    print("initial guess:", guess)
    return guess

# print(c, bc.astype(int), bgarea, rate_conversion, Nsteps.astype(int), dt)

# Continuous Poisson Log-Auto-Regressive 1 with Background
results = stan_utility.sample_model(model, data, outprefix=prefix,
    control=dict(adapt_delta=0.99, max_treedepth=12),
    #warmup=5000, iter=10000,
    init=init_function,
    seed=1)
  #control=dict(max_treedepth=14))

la = stan_utility.get_flat_posterior(results)
samples = []
paramnames = []
badlist = ['lp__', 'phi', 'Bphi']
#badlist += ['log' + k for k in la.keys()]
# remove linear parameters, only show log:
badlist += [k.replace('log', '') for k in la.keys() if 'log' in k and k.replace('log', '') in la.keys()]

from ultranest.plot import PredictionBand
plt.figure(figsize=(10, 5))
plt.plot(x, c / rate_conversion, 'o ')
plt.plot(x, bc / bgarea / rate_conversion, 'o ')
pband = PredictionBand(x)
for ysample in tqdm.tqdm(np.exp(results.posterior.logy.values.reshape((-1, N)))):
    pband.add(ysample)

pband.line(color='k')
pband.shade(color='k', alpha=0.5)
pband.shade(q=0.495, color='k', alpha=0.1)

pband = PredictionBand(x)
for ysample in tqdm.tqdm(np.exp(results.posterior.logBy.values.reshape((-1, N)))):
    pband.add(ysample / bgarea)

pband.line(color='orange')
pband.shade(color='orange', alpha=0.5)
pband.shade(q=0.495, color='orange', alpha=0.1)

plt.yscale('log')
plt.xlabel('Time')
plt.ylabel('Count rate [cts/s]')
plt.ylim(min(((bc + 0.1) / bgarea / rate_conversion).min(), ((c + 0.1) / rate_conversion).min()) / 10, None)
plt.savefig(prefix + '_t.pdf', bbox_inches='tight')
plt.close()


print("priors:")
for k, v in sorted(data.items()):
    if k.startswith('prior'):
        print("%20s: " % k, v / log(10) if k.startswith('log') else v)

print("posteriors:")
for k in sorted(la.keys()):
    print('%20s: %.4f +- %.4f' % (k, la[k].mean(), la[k].std()))
    if k not in badlist and la[k].ndim == 1:
        samples.append(la[k] / log(10) if k.startswith('log') else la[k])
        paramnames.append(k)
    elif la[k].ndim > 1:
        plt.figure()
        plt.hist(la[k].mean(axis=1), histtype='step', bins=40)
        plt.hist(la[k].mean(axis=0), histtype='step', bins=40)
        plt.yscale('log')
        plt.xlabel(k)
        plt.savefig(prefix + "_hist_%s.pdf" % k, bbox_inches='tight')
        plt.close()
        
samples = np.transpose(samples)
print(paramnames)
import corner
corner.corner(samples, labels=paramnames)
plt.savefig(prefix + "_corner_log.pdf", bbox_inches='tight')
plt.close()
corner.corner(np.exp(samples), labels=[k.replace('log', '') for k in paramnames])
plt.savefig(prefix + "_corner.pdf", bbox_inches='tight')
plt.close()

#stan_utility.plot_corner(results, outprefix="plar1b")

T = (x.max() - x.min()) * 100
#xf = np.linspace(0, 1.0 / (2.0 * T), 10000)
#omega = 2 * pi * xf * T
omega = np.linspace(0, 10. / dt, 10000)
# longest duration
omega1 = 1. / x.max()
# Nyquist frequency: twice the bin duration
omega0 = 1. / (2 * dt)

pband2 = PredictionBand(omega)

for tausample, sigma in zip(tqdm.tqdm(results.posterior.tau.values.flatten()), results.posterior.noise.values.flatten()):
    phi = exp(-1. / (tausample / dt))
    gamma = dt / tausample
    specdens = (2 * pi)**0.5 * sigma**2 / (1 - phi**2) * gamma / (pi * (gamma**2 + omega**2))
    pband2.add(specdens)

pband2.line(color='r')
pband2.shade(color='r', alpha=0.5)
pband2.shade(q=0.95/2, color='r', alpha=0.1)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Spectral power density')
plt.xscale('log')
plt.yscale('log')
ylo, yhi = plt.ylim()
# mark observing window:
plt.vlines([omega0, omega1], ylo, yhi, ls='--', color='k', alpha=0.5)
plt.ylim(ylo, yhi)
#plt.xlim(2 / (N * T), 1000 * 2 / (N * T))
plt.savefig(prefix + '_F.pdf', bbox_inches='tight')
plt.close()
