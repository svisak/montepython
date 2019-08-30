#!/usr/bin/env python3

from setuptools import setup
import os

with open('README') as f:
    readme = f.read()

os.system('git describe > VERSION')
version_file = open('VERSION')
version = version_file.read().strip()

setup(
   name='montepython',
   version=version,
   description='Markov chain Monte Carlo algorithms',
   author='Isak Svensson',
   author_email='isak.svensson@chalmers.se',
   packages=['montepython'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)
