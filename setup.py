from setuptools import setup

with open('README') as f:
    readme = f.read()

setup(
   name='montepython',
   version='0.1',
   description='Markov chain Monte Carlo algorithms',
   author='Isak Svensson',
   author_email='isak.svensson@chalmers.se',
   packages=['montepython'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)
