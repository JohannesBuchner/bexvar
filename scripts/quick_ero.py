#!/usr/bin/env python3
"""
Given an eROSITA SRCTOOL light curve, computes various variability statistics.

Run as:

$ python quick_ero.py 020_LightCurve_00001.fits

"""

from numpy import log
import numpy as np
import sys
from astropy.table import Table
from astropy.stats import bayesian_blocks

__version__ = '1.1.0'
__author__ = 'Johannes Buchner'

def calc_Fvar(flux, flux_err):
    assert flux.shape == flux_err.shape
    assert np.isfinite(flux).all()
    assert np.isfinite(flux_err).all()
    flux_mean = np.nanmean(flux)
    n_points = len(flux)

    s_square = np.nansum((flux - flux_mean) ** 2) / (n_points - 1)
    sig_square = np.nansum(flux_err ** 2) / n_points

    nev = (s_square - sig_square) / flux_mean**2
    if not nev > 0.001:
        nev = 0.001
    fvar = np.sqrt(nev)

    sigxserr_a = np.sqrt(2 / n_points) * (sig_square / flux_mean ** 2)
    sigxserr_b = np.sqrt(sig_square / n_points) * (2 * fvar / flux_mean)
    sigxserr = np.sqrt(sigxserr_a**2 + sigxserr_b**2)
    fvar_err = sigxserr / (2 * fvar)

    return nev, sigxserr, fvar, fvar_err, flux_mean, n_points


def calc_MAD(flux, flux_err):
    assert flux.shape == flux_err.shape
    i = flux.argmin()
    j = flux.argmax()
    up = flux[j] - flux_err[j]
    lo = flux[i] + flux_err[i]
    
    diff = up - lo
    mad_sig = diff / (flux_err[i]**2 + flux_err[j]**2)**0.5
    
    return mad_sig, diff

# 1-sigma quantiles and median
#quantiles = scipy.stats.norm().cdf([-1, 0, 1])

first = True
for filename in sys.argv[1:]:
    print(filename, end="\r")
    lc_all = Table.read(filename, hdu='RATE', format='fits')
    nbands = lc_all['COUNTS'].shape[1]
    for band in range(nbands):
        lc = lc_all[lc_all['FRACEXP'][:,band] > 0.1]
        if len(lc['TIME']) == 0:
            continue
        x = lc['TIME'] - lc['TIME'][0]
        bc = lc['BACK_COUNTS'][:,band]
        c = lc['COUNTS'][:,band]
        bgarea = 1. / lc['BACKRATIO']
        rate_conversion = lc['FRACEXP'][:,band] * lc['TIMEDEL']

        # compute rate and rate_err ourselves, because SRCTOOL has nans
        rate = (c - bc / bgarea) / rate_conversion
        sigma_src = (c + 0.75)**0.5 + 1
        sigma_bkg = (bc + 0.75)**0.5 + 1
        rate_err = (sigma_src**2 + sigma_bkg**2 / bgarea)**0.5 / rate_conversion
        assert np.isfinite(rate).all(), rate
        assert np.isfinite(rate_err).all(), rate_err

        nev, nev_err, fvar, fvar_err, cr_mean, n_points = calc_Fvar(rate, rate_err)
        mad_sig, mad = calc_MAD(rate, rate_err)

        cr = rate.mean()
        cr_err = np.mean(rate_err**2)**0.5

        # parameters from eronrta email "regarding BBlocks Alerts" from 18.12.2019
        fp_rate = 0.01
        ncp_prior = 4 - log(73.53 * fp_rate * len(rate)**-0.478)
        edges = bayesian_blocks(t=x, x=rate, sigma=rate_err,
            fitness='measures', ncp_prior=ncp_prior)
        nbblocks=len(edges)-1

        with open('quickstats_%d.txt' % band, 'w' if first else 'a') as out:
            out.write(filename + "\t")
            out.write(filename.replace('020_LightCurve_', '').replace('.fits', '').replace('.gz', '') + "\t")
            out.write("%d\t" % n_points)
            out.write("%f\t" % cr)
            out.write("%f\t" % cr_err)
            out.write("%f\t" % nev)
            out.write("%f\t" % nev_err)
            out.write("%f\t" % (nev/nev_err))
            out.write("%f\t" % fvar)
            out.write("%f\t" % fvar_err)
            out.write("%f\t" % (fvar/fvar_err))
            out.write("%f\t" % mad)
            out.write("%f\t" % mad_sig)
            out.write("%d\n" % nbblocks)
    first = False
print()
