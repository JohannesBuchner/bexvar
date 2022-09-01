#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

try:
    from setuptools import setup
except:
    from distutils.core import setup

try:
    with open('README.rst') as readme_file:
        readme = readme_file.read()

    with open('HISTORY.rst') as history_file:
        history = history_file.read()
except IOError:
    with open(os.path.join(os.path.dirname(__file__), 'README.rst')) as readme_file:
        readme = readme_file.read()

    with open(os.path.join(os.path.dirname(__file__), 'HISTORY.rst')) as history_file:
        history = history_file.read()

requirements = ['numpy', 'scipy', 'ultranest', 'matplotlib', 'astropy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest>=3', ]

setup(
    author="Johannes Buchner",
    author_email='johannes.buchner.acad@gmx.com',
    python_requires='>=3.5.*',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Bayesian excess variance for Poisson data time series with backgrounds.",
    install_requires=requirements,
    license="Affero GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    keywords='bexvar',
    name='bexvar',
    scripts=['scripts/bexvar_ero.py', 'scripts/quick_ero.py'],
    setup_requires=setup_requirements,
    url='https://github.com/JohannesBuchner/bexvar',
    version='1.1.0',
)
