#!/usr/bin/env python3

from setuptools import setup
import subprocess

with open('README') as f:
    readme = f.read()

#cmd = 'git describe --dirty'
cmd = ['git', 'describe', '--tags', '--abbrev=0']
version = subprocess.run(cmd, stdout=subprocess.PIPE)
version = version.stdout.decode('utf-8').strip()

setup(
    name='montepython',
    version=version,
    description='Markov chain Monte Carlo algorithms',
    author='Isak Svensson',
    author_email='isak.svensson@chalmers.se',
    packages=['montepython'],  #same as name
    install_requires=['numpy', 'scipy'], #external packages as dependencies
)
