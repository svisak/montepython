#!/usr/bin/env python3

from setuptools import setup
import subprocess

with open('README') as f:
    readme = f.read()

#cmd = 'git describe --abbrev=0'
#version = subprocess.check_output(cmd, shell=True).decode('utf-8')
cmd = 'git describe --dirty'
dirty_version = subprocess.check_output(cmd, shell=True).decode('utf-8')

setup(
    name='montepython',
    version=dirty_version,
    description='Markov chain Monte Carlo algorithms',
    author='Isak Svensson',
    author_email='isak.svensson@chalmers.se',
    packages=['montepython'],  #same as name
    install_requires=['numpy', 'scipy'], #external packages as dependencies
)
