bexvar
==================

Bayesian excess variance for Poisson data time series with backgrounds.
Excess variance is over-dispersion beyond the observational poisson noise,
caused by an astrophysical source.

* `Introduction <#introduction>`_
* `Method <#method>`_
* `Tutorial <#tutorial>`_
* `Output plot <#visualising-the-results>`_ and files

Introduction
-------------------

In high-energy astrophysics, the analysis of photon count time series
is common. Examples include the detection of gamma-ray bursts,
periodicity searches in pulsars, or the characterisation of
damped random walk-like accretion in the X-ray emission of
active galactic nuclei.

Methods
--------------

paper: https://arxiv.org/abs/2106.14529

This repository provides new statistical analysis methods for light curves.
They can deal with

* very low count statistics (0 or a few counts per time bin)
* (potentially variable) instrument sensitivity
* (potentially variable) backgrounds, measured simultaneously in an 'off' region.

The tools can read eROSITA light curves. Contributions that can read other
file formats are welcome.

The `bexvar_ero.py` tool computes posterior distributions on the Bayesian excess variance,
and source count rate.

`quick_ero.py` computes simpler statistics, including Bayesian blocks,
fraction variance, the normalised excess variance, and 
the amplitude maximum deviation statistics.

Licence
--------
AGPLv3 (see COPYING file). Contact me if you need a different licence.

Install
--------

.. image:: https://img.shields.io/pypi/v/bexvar.svg
    :target: https://pypi.python.org/pypi/bexvar

.. image:: https://github.com/JohannesBuchner/bexvar/actions/workflows/test.yml/badge.svg
    :target: https://github.com/JohannesBuchner/bexvar/actions/workflows/test.yml

.. image:: https://img.shields.io/badge/astroph.HE-arXiv%3A2106.14529-B31B1B.svg
    :target: https://arxiv.org/abs/2106.14529
    :alt: Publication



Install as usual::

	$ pip3 install bexvar

This also installs the required `ultranest <https://johannesbuchner.github.io/UltraNest/>`_
python package.


Example
----------

Run with::

	$ bexvar_ero.py 020_LightCurve_00001.fits

Run simpler variability analyses with::

	$ quick_ero.py 020_LightCurve_*.fits.gz


Contributing
--------------

Contributions are welcome. Please open pull requests
with code contributions, or issues for bugs and questions.

Contributors include:

* Johannes Buchner
* David Bogensberger

If you use this software, please cite this paper: https://arxiv.org/abs/2106.14529
