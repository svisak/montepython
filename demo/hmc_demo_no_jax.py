import montepython
import prettyplease

import h5py
import numpy as np

class MultivariateGaussian(montepython.SimpleBayes):
    '''I no longer use the montepython.Bayes class, SimpleBayes is more convenient.'''

    def __init__(self, mu, cov):
        '''Defining a constructor is optional.'''
        super().__init__()
        self.mu = mu
        self.cov = cov
        self.cov_inv = np.linalg.inv(cov)

    def lnprior(self, position):
        return 0

    def lnlikelihood(self, position):
        diff = position - self.mu
        return -0.5 * diff.T @ self.cov_inv @ diff

    def lnposterior(self, position):
        '''Define posterior.'''
        return self.lnlikelihood(position) + self.lnprior(position)

    def lnposterior_gradient(self, position):
        '''Gradient of the log posterior.'''
        diff = position - self.mu
        return -self.cov_inv @ diff

    def evaluate(self, position):
        '''
        This is the only MANDATORY method!
        You must set posterior value and gradient here.
        Preferably with the set methods as shown.
        How you get hold of value and grad is up to you ...
        '''
        value = self.lnposterior(position)
        grad = self.lnposterior_gradient(position)
        self.set_lnposterior_value(value)
        self.set_lnposterior_gradient(grad)


print('Setting up posterior')
# Random number generator
seed = 665 + 1
rng = np.random.default_rng(seed)

# Define posterior
ndim = 4
mu = np.zeros(ndim)
cov = np.eye(ndim)
bayes = MultivariateGaussian(mu, cov)

# Tune mass matrix using BFGS
print('\nTuning mass matrix using BFGS')
startpos = rng.uniform(low=-1, high=1, size=ndim)
mass_matrix = montepython.tune_mass_matrix(bayes, startpos)

# HMC parameters
startpos = rng.uniform(low=-1, high=1, size=ndim)
n_timesteps = 20 # alias for 'leapfrog_ell'
dt = 0.1 # alias for 'leapfrog_epsilon'

# Create HMC object
hmc = montepython.HMC(bayes, startpos, n_timesteps=n_timesteps, dt=dt, mass_matrix=mass_matrix)

######################### A SHORT RUN ###################################
# Do a short test run to check acceptance rate
# This is NOT burn-in; these are real samples that will be auto-saved soon

print('\nRunning 100 samples')
hmc.run(100)
acc_rate = hmc.acceptance_rate()
print(f'Acceptance rate: {acc_rate:.2f}')

######################### A TIMED RUN ###################################
# In addition to run(), montepython has two other methods for sampling.
# The first is run_for(t_limit, unit='hours') which runs for t_limit units.
# These samples will be appended to the existing 100, exactly as if they originated from the same call.

t_limit = 0.2
unit = 'minutes'
print(f'\nRunning for {t_limit} {unit}')
hmc.run_for(t_limit, unit=unit)
print(f'Finished running.')

######################### A BATCHED TIMED RUN WITH AUTOSAVE ###################################
# The second, which I use, is batched_run_for(t_limit, n_batches, unit='hours', *args, **kwargs)
# which runs for t_limit units n_batches times _with an autosave after each batch_.
# As before, these samples will be appended to the existing ones.

n_batches = 3
path = 'chains' # Optional. Defaults to 'h5'. Will be created if it does not exist.
filename = 'hmc_no_jax.h5' # Optional. Defaults to a timestamp. I recommend using the default.
# dataset_name = 'some_name' # I recommend AGAINST specifying this. But you can do it.
additional_info = {}
additional_info['seed'] = seed
additional_info['parameter_names'] = [f'a{i}' for i in range(ndim)]
print(f'\nRunning for {t_limit} {unit} {n_batches} times with autosaving')
hmc.batched_run_for(t_limit, n_batches, unit=unit, path=path, filename=filename, **additional_info)
print(f'Finished {n_batches} batches.')
print(f'Number of samples: {len(hmc.chain())}')

# See the docstring/code/README for more options/information. 

# Alternatively, you can use run() or run_for() and call hmc.to_disk() manually.
# Or you can get the chain itself with hmc.chain() and save it yourself however you like.

######################### PLOT RESULT ##################################
print('Plotting results.')
f = h5py.File(f'{path}/{filename}', 'r')
dset_list = list(f.keys())
print('Datasets in .h5 file:')
print(dset_list)
dset_name = dset_list[-1]
print(f'Plotting dataset {dset_name}')
dset = f.get(dset_name)
chain = dset[...]
labels = dset.attrs.get('parameter_names')
fig = prettyplease.corner(chain, labels=labels, title_loc='center')
fig.savefig('corner.pdf', bbox_inches='tight')
