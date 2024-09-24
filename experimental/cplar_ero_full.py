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
from numpy import log, pi, exp
import sys
import tqdm
import h5py
import datetime
from brokenaxes import brokenaxes
import astropy.time
from astropy.table import Table
import cmdstancache
import corner
from ultranest.plot import PredictionBand


model_code = """
data {
  int N;

  // duration of time bin
  real dt;
  // number of time step between end of previous time bin and end of current time bin
  // this may be different from dt when there are gaps
  real<lower=0> tstep;
  array[N] int<lower=0> z_obs;
  array[N] int<lower=0> B_obs;
  vector[N] Barearatio;
  vector[N] countrate_conversion;
  vector[N] Bcountrate_conversion;
  int N_good;
  array[N] int<lower=0> mask_good;

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
  real T = tstep * N;
  real logT = log(T);

  array[N_good] int<lower=0> B_obs_filtered;
  array[N_good] int<lower=0> z_obs_filtered;
  {
    int j = 0;
    for (i in 1:N) {
      if (mask_good[i] == 1) {
        j += 1;
        B_obs_filtered[j] = B_obs[i];
        z_obs_filtered[j] = z_obs[i];
      }
    }
  }
}
parameters {
  // auto-regression time-scale
  real logtau;
  // mean
  real logc;
  // sigma
  real lognoise;
  // unit normal noise deviations
  vector[N] W;

  // intrinsic background count rate at each time bin
  vector[N] logBy;

  // hyper-parameter for background count rates
  real prior_logBc_mean;
  real<lower=0> prior_logBc_std;
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
    logy[i] = logc + exp(-tstep / tau) * (logy[i-1] - logc) + W[i] * noise * tstep / dt;
    // print(i, ": ", logc, " ", tstep, " ", tau, " ", dt, " ", logy[i-1], " ", logy[i], " ", W[i], " ", noise);
  }
}
model {
  // correct for observing efficiency
  vector[N] logysrcarea = logy + logcountrate_conversion;
  vector[N] logybkgarea = logBy + logBarearatio + logBcountrate_conversion;
  vector[N_good] logytot_filtered;
  vector[N_good] logBy_filtered;
  // sum source and background together
  {
    int j = 0; 
    for (i in 1:N) {
      if (mask_good[i] == 1) {
        j += 1;
        logytot_filtered[j] = log_sum_exp(logysrcarea[i], logybkgarea[i]);
        logBy_filtered[j] = logBy[i] + logBcountrate_conversion[i];
        // print(i, ": ", logytot_filtered[j], " ", logy[i], " ", logcountrate_conversion[i], logysrcarea[i], " ", logybkgarea[i]);
      }
    }
  }

  prior_logBc_std ~ uniform(0, 5);
  logBy ~ normal(prior_logBc_mean, prior_logBc_std);
  B_obs_filtered ~ poisson_log(logBy_filtered);

  // source AR process priors:
  logtau ~ normal(prior_logtau_mean, prior_logtau_std);
  logc ~ normal(prior_logc_mean, prior_logc_std);
  lognoise ~ normal(prior_lognoise_mean, prior_lognoise_std);
  //logy ~ normal(0, 5); // stay within a reasonable range
  W ~ std_normal();
  // comparison to total source region counts
  z_obs_filtered ~ poisson_log(logytot_filtered);
}
"""



filename = sys.argv[1]

seed = 1

lc_all = Table.read(filename, hdu='RATE', format='fits')
nbands = lc_all['COUNTS'].shape[1]
if len(sys.argv) > 2:
    band = int(sys.argv[2])
else:
    band = 0

fracexp = lc_all['FRACEXP'][:,band]
indices_good_fracexp, = np.where(fracexp > fracexp.max() * 0.1)
print("band %d" % band)

print(fracexp.min(), fracexp.max())
# only model part where we have data
# skip last, because TIMEDEL is different
lc = lc_all[indices_good_fracexp.min() : min(len(fracexp) - 1, indices_good_fracexp.max() + 1)]
bc = lc['BACK_COUNTS'][:,band].value
c = lc['COUNTS'][:,band].value
bgarea = np.array(1. / lc['BACKRATIO'].value)
# length of the time bin
dt = lc['TIMEDEL']
assert dt.max() == dt.min(), (dt.max(), dt.min())
dt = dt.min()
print("dt:", dt)
# TIME is the mid point of the light curve bin
# here we want the starting point:
t0 = lc['TIME'][0]
x_start = lc['TIME'] - t0 - lc['TIMEDEL'] / 2.0
x_end   = lc['TIME'] - t0 + lc['TIMEDEL'] / 2.0
x = (lc['TIME'] - t0)
tsteps = (x_end[1:] - x_end[:-1]).value
assert (tsteps == dt).all(), np.unique(tsteps)
assert (bc.astype(int) == bc).all(), bc
prefix = sys.argv[1] + '-%d-cplar1full' % band
fe = lc['FRACEXP'][:,band].value
rate_conversion = fe * dt
mask_good = fe > fe.max() * 0.1
assert (fe[mask_good] > 0).all()

"""
fig, axs = plt.subplots(2, 1, figsize=(25, 10), sharex=True)
bax = axs[0]
bax.plot(lc_all['TIME'], fracexp, 'o ', ms=2)
bax.plot(lc['TIME'], fe, 's-', ms=2)
bax.set_xlabel('Time [s]')
bax.set_ylabel('FRACEXP')
bax.set_xlim(lc_all['TIME'].min(), lc_all['TIME'].max())
bax = axs[1]
bax.plot(lc_all['TIME'], lc_all['COUNTS'][:,band].value, 'o ', ms=2)
bax.plot(lc['TIME'], c, 's-', ms=2)
bax.set_xlabel('Time [s]')
bax.set_ylabel('COUNTS')
bax.set_ylim(1, None)
bax.set_yscale('log')
bax.set_xlim(lc_all['TIME'].min(), lc_all['TIME'].max())
fig.savefig(prefix + '_fracexp.pdf')
plt.close()
"""

#print("tsteps:", tsteps.sum())

N = len(x_start)
data = dict(
    # provide observations
    N=N, z_obs=c, B_obs=bc.astype(int),
    # additional information about the sampling:
    dt=dt, Barearatio=1. / bgarea, tstep=dt,
    # source count rate is modulated by fracexp
    countrate_conversion=rate_conversion,
    # background count rate is modulated by fracexp, except in the hard band,
    # where it is assumed constant (particle background dominated)
    Bcountrate_conversion=rate_conversion*0 + 1 if band == 2 else rate_conversion,
    N_good = mask_good.sum(), mask_good=mask_good * 1,
    # source count rates expected:
    prior_logc_mean=0, prior_logc_std=5,
    # expected noise level is 10% +- 2 dex
    prior_lognoise_mean=np.log(0.1), prior_lognoise_std=np.log(100),
    prior_logtau_mean=np.log(x.max()), prior_logtau_std=3,
    # prefer long correlation time-scales; the data have to convince us that the
    # data points scatter.
    # prior_logtau_mean=np.log(x.max()), prior_logtau_std=10 * np.log(x.max() / dt),
    # prefer short correlation time-scales; the data have to convince us that the
    # data points are similar.
    #prior_logtau_mean=np.log(dt), prior_logtau_std=np.log(1000),
)

def init_function(seed):
    # guess good parameters for chain to start
    rng = np.random.RandomState(seed)

    guess = dict(
        # start with short time-scales: all bins independent
        logtau=np.log(dt / 2),
        # background count rates; estimated from background counts
        logBy=np.array(np.log((bc + 0.1) / (0.0001 + data['Bcountrate_conversion']))),
        # background count rates expected:
        prior_logBc_mean=np.median(np.log((bc + 0.1) / (0.0001 + data['Bcountrate_conversion']))),
        prior_logBc_std=np.log(10),
        # average count rate; estimated from counts
        logc=np.median(np.log(((c + 0.1) / (0.0001 + data['countrate_conversion'])))),
        # minimal noise
        lognoise=np.log(1e-10),
        W=rng.normal(size=N),

    )
    # print("initial guess:", guess)
    return guess

# Continuous Poisson Log-Auto-Regressive 1 with Background
stan_variables, method_variables = cmdstancache.run_stan(
    model_code, data=data,
    # adapt_delta=0.98, max_treedepth=12,
    #show_console=True,
    refresh=10,
    inits=init_function(seed),
    # iter_warmup=2000, iter_sampling=1000,
    seed=seed)

paramnames = []
badlist = ['lp__', 'phi', 'Bphi']
#badlist += ['log' + k for k in la.keys()]
# remove linear parameters, only show log:
badlist += [k.replace('log', '') for k in stan_variables.keys()
    if 'log' in k and k.replace('log', '') in stan_variables.keys()]

typical_step = max(np.median(tsteps), dt * 5)

def tomjd(t_seconds):
    return (astropy.time.Time(51543.875, format='mjd') + (t_seconds + t0) * astropy.units.s).mjd - mjd0

mjd0 = 58828
x_mjd = tomjd(x.value)
print(x_mjd, x.value)

for broken in False, True:
    print("plotting time series...")
    fig = plt.figure(figsize=(15, 5))
    if broken:
        i, = np.where((x_end[mask_good][1:] - x_end[mask_good][:-1]).value > typical_step * 20)
        xlims = list(
            zip([tomjd(x_start.value[mask_good][0] - 3 * typical_step)] + list(tomjd(x_start.value[mask_good][i+1] - 3 * typical_step)),
            list(tomjd(x_end.value[mask_good][i] + 3 * typical_step)) + [tomjd(x_end.value[mask_good][-1] + 3 * typical_step)]))
        print(xlims)
        bax = brokenaxes(xlims=xlims, hspace=0.05)
    else:
        bax = plt.gca()

    bax.plot(x_mjd[mask_good], ((c + 0.1) / rate_conversion)[mask_good], 'x ', ms=4, label='source count rate', color='tab:blue')
    bax.plot(x_mjd[mask_good], ((bc + 0.1) / bgarea / rate_conversion)[mask_good], '+ ', ms=4, label='background count rate', color='gray')
    y = np.exp(stan_variables['logy'].reshape((-1, N)))
    bax.plot(x_mjd, np.median(y, axis=0), color='navy')
    bax.fill_between(
        x_mjd,
        np.quantile(y, 0.005, axis=0),
        np.quantile(y, 0.995, axis=0),
        color='navy', alpha=0.4, label='source log-AR(1) model',
    )
    By = np.exp(stan_variables['logBy'].reshape((-1, N))) / bgarea
    bax.plot(x_mjd[mask_good], np.median(By, axis=0)[mask_good], color='gray')
    bax.fill_between(
        np.where(mask_good, x_mjd, np.nan),
        np.where(mask_good, np.quantile(By, 0.005, axis=0), np.nan),
        np.where(mask_good, np.quantile(By, 0.995, axis=0), np.nan),
        color='gray', alpha=0.4, label='background LogNormal model',
    )
    bax.set_yscale('log')
    bax.set_xlabel('Time [MJD - %d]' % mjd0)
    bax.set_ylabel('Count rate [cts/s]')
    bax.set_title(
        'ID=%s <source_counts>:%.2f #=%d' % (
            filename.replace('.fits', '').replace('020_LightCurve_', ''),
            c.mean(), len(c)))

    if broken:
        bax.set_ylim(
          ((c + 0.1) / rate_conversion)[mask_good].min(),
          ((c + 0.1) / rate_conversion)[mask_good].max(),
        )
        ymax = ((c + 0.1) / rate_conversion)[mask_good].max()
    else:
        bax.set_xlim(x_mjd.min(), x_mjd.max())
        bax.set_ylim(bax.get_ylim())
        ymax = bax.set_ylim()[1]
    
    for year in np.arange(2020, 2023, 0.5):
        year_mjd = (astropy.time.Time(datetime.datetime(int(year), int(year * 12) % 12 + 1, 1), format='datetime')).mjd - mjd0
        if x_mjd.min() < float(year_mjd) < x_mjd.max():
            try:
                bax.text(
                    year_mjd,
                    ymax * 0.99,
                    '%s ' % (year), rotation=90, size=6, ha='center', va='top',
                )
            except ValueError:
                pass
    bax.legend()
    if broken:
        fig.savefig(prefix + '_t_s.pdf')
    else:
        fig.savefig(prefix + '_t.pdf')
    plt.close()

print("priors:")
for k, v in sorted(data.items()):
    if k.startswith('prior'):
        # convert to base 10 for easier reading
        print("%20s: " % k, v / log(10) if k.startswith('log') else v)

samples = []

with h5py.File(prefix + "_post.h5", 'w') as fout:
    for k, v in stan_variables.items():
        if k not in badlist:
            print("storing", k)
            fout.create_dataset(k, data=v, shuffle=True, compression='gzip')

print("posteriors:")
for k in sorted(stan_variables.keys()):
    print('%20s: %.4f +- %.4f' % (k, stan_variables[k].mean(), stan_variables[k].std()))
    if k not in badlist and stan_variables[k].ndim == 1:
        # convert to base 10 for easier reading
        samples.append(stan_variables[k] / log(10) if k.startswith('log') else stan_variables[k])
        paramnames.append(k)
    elif stan_variables[k].ndim > 1 and False:
        plt.figure()
        plt.hist(stan_variables[k].mean(axis=1), histtype='step', bins=40, label='over bins')
        plt.hist(stan_variables[k].mean(axis=0), histtype='step', bins=40, label='over realisations')
        plt.yscale('log')
        plt.xlabel(k)
        plt.legend(title='average')
        plt.savefig(prefix + "_hist_%s.pdf" % k, bbox_inches='tight')
        plt.close()

samples = np.transpose(samples)
if False:
    print('making corner plots ...')
    corner.corner(samples, labels=paramnames)
    plt.savefig(prefix + "_corner_log.pdf", bbox_inches='tight')
    plt.close()
    corner.corner(10**(samples), labels=[k.replace('log', '') for k in paramnames])
    plt.savefig(prefix + "_corner.pdf", bbox_inches='tight')
    plt.close()

print("saving samples...")
np.savetxt(
  prefix + 'LC_netrate.txt.gz',
  np.transpose(np.vstack((lc['TIME'].reshape((1, -1)), x.reshape((1, -1)), y[::40]))),
  fmt='%.5f', #delimiter=',',
)

print("plotting fourier transforms ...")
# switch to units of seconds here
omega = np.linspace(0, 10. / dt, 10000)
# longest duration: entire observation
omega1 = 1. / x.max()
# Nyquist frequency: twice the bin duration
omega0 = 1. / (2 * dt)

pband2 = PredictionBand(omega)

for tausample, sigma in zip(tqdm.tqdm(stan_variables['tau'].flatten()), stan_variables['noise'].flatten()):
    DT = 1 # unit: seconds
    phi = exp(-DT / tausample)
    gamma = DT / tausample
    specdens = (2 * pi)**0.5 * sigma**2 / (1 - phi**2) * gamma / (pi * (gamma**2 + omega**2))
    pband2.add(specdens / x.max() / dt)
print('factors:', x.max(), dt, len(x))
pband2.line(color='r')
pband2.shade(color='r', alpha=0.5)
pband2.shade(q=0.95/2, color='r', alpha=0.1)

# handle inferred damped random walk realisations here:
#from astropy.timeseries import LombScargle
def fourier_periodogram(t, y):
    N = len(t)
    frequency = np.fft.fftfreq(N, t[1] - t[0])
    y_fft = np.fft.fft(y)
    positive = (frequency > 0)
    return frequency[positive], (1. / N) * abs(y_fft[positive]) ** 2

#from ducc0.fft import genuine_fht
#from scipy.ndimage import gaussian_filter
#f = 1. / np.array(x)[1:][::-1]
#pbandf = PredictionBand(f)
#pbandf = PredictionBand(f[len(f) // 2:])
#pbandf = PredictionBand(f[1::2])
#pbandf = PredictionBand(1. / dt / (1 + np.arange(len(f[1::2])))[::-1])
pbandf = None
t = np.array(x)

for y_realisation in tqdm.tqdm(y):
    # take the fourier transform of y_realisation
    #fourier_spectrum_twice = (genuine_fht(y_realisation))**2 * len(x)
    #fourier_spectrum = fourier_spectrum_twice[: len(fourier_spectrum_twice) // 2]
    # make a bit smoother
    #smooth_fourier_spectrum = gaussian_filter(
    #    fourier_spectrum[1:],
    #    sigma=2, truncate=100, mode='nearest')
    #print(smooth_fourier_spectrum.shape)
    #pbandf.add(smooth_fourier_spectrum)
    #frequency, power = LombScargle(t, y_realisation, normalization='psd').autopower()
    frequency, power = fourier_periodogram(t, y_realisation)
    if pbandf is None:
        pbandf = PredictionBand(frequency)
    pbandf.add(power)

#plt.figure()
pbandf.line(color='navy')
pbandf.shade(color='navy', alpha=0.5)
pbandf.shade(q=0.95/2, color='navy', alpha=0.1)

plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectral density (PSD)')
plt.xscale('log')
plt.yscale('log')
ylo, yhi = plt.ylim()
# mark observing window:
plt.vlines([omega0, omega1], ylo, yhi, ls='--', color='k', alpha=0.5)
plt.ylim(ylo, yhi)
#plt.xlim(2 / (N * T), 1000 * 2 / (N * T))
plt.savefig(prefix + '_F.pdf', bbox_inches='tight')
plt.close()

#
