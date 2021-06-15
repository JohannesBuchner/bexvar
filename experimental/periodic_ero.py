#!/usr/bin/env python3
"""
Given an eROSITA SRCTOOL light curve, computes a Bayesian periodogram.

The model is:

   rate = B + A * ((1 + sin((t/P + p) * 2 * pi)) / 2)**shape

   B: base rate
   A: variable rate
   P: period
   p: phase
   shape: signal shape parameter (30 for QPEs, 1 for sine)

Example:

$ python periodic_ero.py 020_LightCurve_00001.fits

It will make a few visualisations and a file containing the resulting parameters.

"""

import sys
import argparse

import joblib
import matplotlib.pyplot as plt
from numpy import log, log10, pi, sin
import numpy as np
import scipy.stats, scipy.optimize
from astropy.table import Table
import tqdm.auto as tqdm
#from getdist import MCSamples, plots

mem = joblib.Memory('.', verbose=False)

# 1-sigma quantiles and median
quantiles = scipy.stats.norm().cdf([-1, 0, 1])

N = 1000
M = 1000

@mem.cache
def estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion):
    """ Compute the PDF at positions in log(source count rate)s grid log_src_crs_grid 
    for observing src_counts counts in the source region of size src_area,
    and bkg_counts counts in the background region of size bkg_area.
    
    """
    # background counts give background cr deterministically
    u = np.linspace(0, 1, N)[1:-1]
    def prob(log_src_cr):
        src_cr = 10**log_src_cr * rate_conversion
        bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) / bkg_area
        like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
        return like
    
    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    if weights.sum() == 0:
        print(np.log10(src_counts.max() / rate_conversion))
    weights /= weights.sum()
    
    return weights

def model(time, base, ampl, period, phase, shape):
    return base + ampl * ((1 + sin((time / period + phase) * 2 * pi)) / 2)**shape

class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write('error: %s\n' % message)
		self.print_help()
		sys.exit(2)

parser = HelpfulParser(description=__doc__,
	epilog="""Johannes Buchner (C) 2020 <johannes.buchner.acad@gmx.com>""",
	formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('lightcurve', type=str, help="eROSITA light curve fits file")
parser.add_argument("--band", type=int, default=0, help="Energy band")
parser.add_argument("--fracexp_min", type=float, default=0.1, help="Smallest fractional exposure to consider")
parser.add_argument("--shape_mean", type=float, default=1, help="Expected shape. Use 1 for sine, 30 for QPE.")
parser.add_argument("--shape_std", type=float, default=0.5, help="Expected shape diversity in dex.")

args = parser.parse_args()

filename = args.lightcurve
band = args.band

lc_all = Table.read(filename, hdu='RATE', format='fits')
if lc_all['COUNTS'].ndim == 1:
    lc = lc_all[lc_all['FRACEXP'] > args.fracexp_min]
    bc = lc['BACK_COUNTS'].astype(int)
    c = lc['COUNTS'].astype(int)
    rate_conversion = lc['FRACEXP'] * lc['TIMEDEL']
    rate = lc['RATE']
    rate_err=lc['RATE_ERR']
else:
    nbands = lc_all['COUNTS'].shape[1]
    print("band %d" % band)
    lc = lc_all[lc_all['FRACEXP'][:,band] > args.fracexp_min]
    bc = lc['BACK_COUNTS'][:,band].astype(int)
    c = lc['COUNTS'][:,band]
    rate_conversion = lc['FRACEXP'][:,band] * lc['TIMEDEL']
    rate = lc['RATE'][:,band]
    rate_err=lc['RATE_ERR'][:,band]

bgarea = 1. / lc['BACKRATIO']
t0 = lc['TIME'][0]
x = lc['TIME'] - t0

if np.log10(c / rate_conversion + 0.001).max() > 2:
    log_src_crs_grid = np.linspace(-2, np.log10(c / rate_conversion).max() + 0.5, M)
else:
    log_src_crs_grid = np.linspace(-2, 2, M)

print("preparing time bin posteriors...")
src_posteriors_list = []
for xi, ci, bci, bgareai, rate_conversioni in zip(tqdm.tqdm(x), c, bc, bgarea, rate_conversion):
    # print(xi, ci, bci, bgareai, rate_conversioni)
    pdf = estimate_source_cr_marginalised(log_src_crs_grid, ci, bci, bgareai, rate_conversioni)
    src_posteriors_list.append(pdf)
src_posteriors = np.array(src_posteriors_list)

print("running qpe...")
outprefix = '%s-band%d-fracexp%s-expsine' % (filename, band, args.fracexp_min)
time = x

parameter_names = ['jitter', 'logbase', 'logampl', 'logperiod', 'phase', 'shape']
rv_gamma = scipy.stats.norm(np.log10(args.shape_mean), args.shape_std)
rv_base = scipy.stats.norm(0, 2)
rv_ampl = scipy.stats.norm(0, 2)
logTmax = log10((time[-1] - time[0]) * 5)
logTmin = log10(np.min(time[1:] - time[:-1]))

def transform(cube):
    params = cube.copy()
    params[0] = 10**(cube[0]*2 - 2)
    params[1] = rv_base.ppf(cube[1])
    params[2] = rv_ampl.ppf(cube[2])
    params[3] = cube[3]*(logTmax - logTmin) + logTmin
    params[4] = cube[4]
    params[5] = 10**rv_gamma.ppf(cube[5])
    return params

def loglike(params):
    jitter, log_base, log_amplfrac, log_period, phase, shape = params
    # predict model:
    model_logmean = log10(model(time, 10**log_base, 10**(log_base + log_amplfrac), 10**log_period, phase, shape))

    # compute for each grid log-countrate its probability, according to log_mean, log_sigma
    variance_pdf = np.exp(-0.5 * ((log_src_crs_grid.reshape((1, -1)) - model_logmean.reshape((-1, 1))) / jitter)**2) / (2 * pi * jitter**2)**0.5
    # multiply that probability with the precomputed probabilities (pdfs)
    like = log((variance_pdf * src_posteriors).mean(axis=1) + 1e-100).sum()
    if not np.isfinite(like):
        like = -1e300
    return like

from ultranest import ReactiveNestedSampler
import ultranest.stepsampler
sampler = ReactiveNestedSampler(parameter_names, loglike, transform=transform, 
    log_dir=outprefix, resume=True)
sampler.stepsampler = ultranest.stepsampler.RegionSliceSampler(nsteps=40, max_nsteps=40, adaptive_nsteps='move-distance')
samples = sampler.run(frac_remain=0.5)['samples']
print("running qpe... done")

print("plotting ...")

sampler.plot()

plt.figure(figsize=(12, 4))
from brokenaxes import brokenaxes

x1 = x + lc['TIMEDEL'] * 2
x0 = x - lc['TIMEDEL'] * 3
mask_breaks = x0[1:] > x1[:-1]
xlims = list(zip([x0.min()] + list(x0[1:][mask_breaks]), list(x1[:-1][mask_breaks]) + [x1.max()]))
xm = np.hstack([np.linspace(lo, hi, 40) for lo, hi in xlims])
xm[::40] = np.nan

print(xlims)
from ultranest.plot import PredictionBand

bax = brokenaxes(xlims=xlims, hspace=.05)

for jitter, log_base, log_amplfrac, log_period, phase, shape in tqdm.tqdm(samples[:100]):
    model_logmean = model(xm, 10**log_base, 10**(log_base + log_amplfrac), 10**log_period, phase, shape)
    bax.plot(xm, model_logmean, color='orange', alpha=0.5)

bax.errorbar(x, y=rate, yerr=rate_err, marker='x', ls=' ')
bax.set_xlabel('Time [s] - %s' % t0)
bax.set_ylabel('Count rate [cts/s]')
plt.yscale('log')
plt.savefig(outprefix + '.pdf', bbox_inches='tight')
plt.close()

del xm, band
xm = np.linspace(x0.min() - (x1.max() + x0.min()), x1.max() + (x1.max() + x0.min()), 4000)
plt.figure(figsize=(12, 4))
band = PredictionBand(xm)
for jitter, log_base, log_amplfrac, log_period, phase, shape in tqdm.tqdm(samples[:4000]):
    #if 10**log_period < 10000: continue
    band.add(model(xm, 10**log_base, 10**(log_base + log_amplfrac), 10**log_period, phase, shape))
#    plt.plot(xm, model_logmean, color='orange', alpha=0.5)

band.line(color='orange')
band.shade(q=0.45, color='orange', alpha=0.2)
band.shade(color='orange', alpha=0.2)
plt.errorbar(x, y=rate, yerr=rate_err, marker='x', ls=' ')
plt.xlabel('Time [s] - %s' % t0)
plt.ylabel('Count rate [cts/s]')
plt.yscale('log')
plt.savefig(outprefix + '-full.pdf', bbox_inches='tight')
plt.close()
