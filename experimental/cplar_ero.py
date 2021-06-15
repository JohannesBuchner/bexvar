#!/usr/bin/env python3
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
  int<lower=1> Nsteps[N-1];
  int<lower=0> z_obs[N];
  int<lower=0> B_obs[N];
  vector[N] Barearatio;
  vector[N] countrate_conversion;
  vector[N] Bcountrate_conversion;

  real prior_logBc_mean;
  real prior_logBnoise_mean;
  real prior_logBc_std;
  real prior_logBnoise_std;
  real prior_logc_mean;
  real prior_lognoise_mean;
  real prior_logc_std;
  real prior_lognoise_std;

}
transformed data {
  vector[N] logBarearatio = log(Barearatio);
  vector[N] logcountrate_conversion = log(countrate_conversion);
  vector[N] logBcountrate_conversion = log(Bcountrate_conversion);
  real logminduration = log(dt);
  real logT = log(sum(Nsteps) * dt);
  real T = sum(Nsteps) * dt;
}
parameters {
  // regression time-scale
  real<lower=log(dt / 10),upper=log(10*T)> logtau;
  // mean
  real logc;
  // sigma
  real lognoise;
  // unit normal noise deviations
  vector[N] W;

  // regression time-scale
  real<lower=log(dt / 10),upper=log(10*T)> logBtau;
  // mean
  real logBc;
  // sigma
  real logBnoise;
  // unit normal noise deviations
  vector[N] BW;
}
transformed parameters {
  // linear transformations of the log parameters
  real phi;
  real tau;
  real noise;
  real c;
  // intrinsic source count rate at each time bin
  vector[N] logy;

  //real Bphi;
  real Btau;
  real Bnoise;
  real Bc;
  // intrinsic background count rate at each time bin
  vector[N] logBy;

  // transform parameters to linear space
  tau = exp(logtau);
  Btau = exp(logBtau);
  noise = exp(lognoise);
  Bnoise = exp(logBnoise);
  c = exp(logc);
  Bc = exp(logBc);
  phi = exp(-dt / tau);
  //Bphi = exp(-dt / Btau);

  // apply centering to real units, AR(1) formulas:
  logy[1] = logc + W[1] * noise;
  logBy[1] = logBc + BW[1] * Bnoise;
  for (i in 2:N) {
    // mean, dampened random walk term, and noise term
    logBy[i] = logBc + exp(-dt * Nsteps[i-1] / Btau) * (logBy[i-1] - logBc) + BW[i] * exp(logBnoise) * Nsteps[i-1];
    //logBy[i] = logBc + BW[i] * Bnoise;
    logy[i] = logc + exp(-dt * Nsteps[i-1] / tau) * (logy[i-1] - logc) + W[i] * exp(lognoise) * Nsteps[i-1];
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
  
  // background AR process priors:
  // logBtau ~ normal(logT, 10 * logT / logminduration);
  //Btau ~ uniform(dt, 10 * T);
  logBc ~ normal(prior_logBc_mean, prior_logBc_std);
  logBnoise ~ normal(prior_logBnoise_mean, prior_logBnoise_std);
  BW ~ std_normal();
  // comparison to background counts
  B_obs ~ poisson_log(logBy + logBcountrate_conversion);

  // source AR process priors:
  // logtau ~ normal(logT, 10 * logT / logminduration);
  // tau ~ uniform(dt, 10 * T);
  logc ~ normal(prior_logc_mean, prior_logc_std);
  lognoise ~ normal(prior_lognoise_mean, prior_lognoise_std);
  W ~ std_normal();
  // comparison to total source region counts
  z_obs ~ poisson_log(logytot);
}
""")


np.random.seed(1)

filename = sys.argv[1]


lc_all = Table.read(filename, hdu='RATE', format='fits')
nbands = lc_all['COUNTS'].shape[1]
for band in range(nbands):
    print("band %d" % band)
    lc = lc_all[lc_all['FRACEXP'][:,band] > 0.1]
    x = lc['TIME'] - lc['TIME'][0]
    bc = lc['BACK_COUNTS'][:,band]
    c = lc['COUNTS'][:,band]
    bgarea = 1. / lc['BACKRATIO']
    fe = lc['FRACEXP'][:,band]
    rate_conversion = fe * lc['TIMEDEL']
    dt = np.unique(lc['TIMEDEL'])
    assert len(dt) == 1
    dt = dt[0]
    Nsteps = (x[1:] - x[:-1]) / dt
    assert (Nsteps.astype(int) == Nsteps).all()
    assert (bc.astype(int) == bc).all(), bc
    prefix = sys.argv[1] + '-%d-plar1b' % band

    N = len(x)
    data = dict(
        # provide observations
        N=N, z_obs=c, B_obs=bc.astype(int),
        # additional information about the sampling: 
        dt=dt, Barearatio=1. / bgarea, Nsteps=Nsteps.astype(int),
        # source count rate is modulated by fracexp
        countrate_conversion=rate_conversion,
        # background count rate is modulated by fracexp, except in the hard band,
        # where it is assumed constant (particle background dominated)
        Bcountrate_conversion=rate_conversion*0 + 1 if band == 2 else rate_conversion,
        prior_logBc_mean=0, prior_logBnoise_mean=0,
        prior_logBc_std=10, prior_logBnoise_std=3,
        prior_logc_mean=0, prior_lognoise_mean=0,
        prior_logc_std=10, prior_lognoise_std=3,
    )
    # print(c, bc.astype(int), bgarea, rate_conversion, Nsteps.astype(int), dt)

    # Continuous Poisson Log-Auto-Regressive 1 with Background
    results = stan_utility.sample_model(model, data, outprefix=prefix, control=dict(max_treedepth=16))

    la = stan_utility.get_flat_posterior(results)
    samples = []
    paramnames = []
    badlist = ['lp__', 'phi', 'Bphi']
    #badlist += ['log' + k for k in la.keys()]
    # remove linear parameters, only show log:
    badlist += [k.replace('log', '') for k in la.keys() if 'log' in k and k.replace('log', '') in la.keys()]

    for k in sorted(la.keys()):
        print('%20s: %.4f +- %.4f' % (k, la[k].mean(), la[k].std()))
        if k not in badlist and la[k].ndim == 1:
            samples.append(la[k] / log(10) if k.startswith('log') else la[k])
            paramnames.append(k)
    samples = np.transpose(samples)
    print(paramnames)
    import corner
    corner.corner(samples, labels=paramnames)
    plt.savefig(prefix + "_corner.pdf", bbox_inches='tight')
    plt.close()

    #stan_utility.plot_corner(results, outprefix="plar1b")

    from ultranest.plot import PredictionBand
    plt.figure(figsize=(10, 5))
    plt.plot(x, c / rate_conversion, 'o ')
    plt.plot(x, bc / bgarea / rate_conversion, 'o ')
    pband = PredictionBand(x)
    for ysample in tqdm.tqdm(np.exp(results.posterior.logy.values.reshape((-1, N)))):
        pband.add(ysample)

    pband.line(color='k')
    pband.shade(color='k', alpha=0.5)
    pband.shade(q=0.49, color='k', alpha=0.1)

    pband = PredictionBand(x)
    for ysample in tqdm.tqdm(np.exp(results.posterior.logBy.values.reshape((-1, N)))):
        pband.add(ysample / bgarea)

    pband.line(color='orange')
    pband.shade(color='orange', alpha=0.5)
    pband.shade(q=0.49, color='orange', alpha=0.1)

    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Count rate [cts/s]')
    #plt.ylim(1e-4, None)
    plt.savefig(prefix + '.pdf', bbox_inches='tight')
    plt.close()


    T = (x.max() - x.min()) * 100
    #xf = np.linspace(0, 1.0 / (2.0 * T), 10000)
    #omega = 2 * pi * xf * T
    omega = np.linspace(0, 10. / dt, 10000)
    omega1 = 1. / x.max()
    omega0 = 1. / dt

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
    plt.savefig(prefix + 'plar1b_F.pdf', bbox_inches='tight')
    plt.close()
    break
