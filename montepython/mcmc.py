from abc import ABC, abstractmethod
import numpy as np
import os
import socket
import sys
import time

from .state import State
from .metachain import MetaChain
from .utils import mcmc_to_disk, convert_to_seconds
import montepython


class MCMC(ABC):
    """
    Abstract MCMC base class. All actual sampling classes (such as HMC)
    inherits the methods in this class, and implement the abstract methods.

    All pertinent information about the sampler, including the Markov chain
    itself, can be accessed with the methods provided here.

    :param initial_state:
        The starting State of the Markov chain.

    :param *args:
        Not used.

    :param **kwargs:
        Implementation-specific parameters. Please see the documentation for
        the specific sampler you want to use to learn about parameters
        specific to that sampler.

    """

    def __init__(self, bayes, startpos, **kwargs):
        # MAKE SURE STARTPOS IS AN NDARRAY
        if not isinstance(startpos, np.ndarray):
            msg = 'startpos is not an ndarray'
            raise ValueError(msg)

        # SET BAYES
        self._bayes = bayes

        # SET COMMON INITIAL STATE PARAMETERS
        self._bayes.evaluate(startpos)
        tmp = {}
        tmp['position'] = startpos
        tmp['lnposterior'] = self._bayes.get_lnposterior_value()
        initial_state = State(**tmp)

        # CREATE METACHAIN
        self._metachain = MetaChain(initial_state)

        # POP OPTIONAL PARAMETERS
        self._temperature = kwargs.pop('temperature', 1.0)

        # SAVE TOTAL SAMPLING TIME
        self._total_runtime = 0

        # NUMPY SEED
        self._numpy_seed = None

    @property
    def numpy_seed(self):
        return self._numpy_seed

    @numpy_seed.setter
    def numpy_seed(self, seed):
        self._numpy_seed = seed
        np.random.seed(seed)

    def acceptance_rate(self):
        """Return the acceptance rate of the samples so far."""
        return self._metachain.acceptance_rate()

    def chain(self, warmup=0):
        """
        Return the Markov chain resulting from the sampling.
        Discard the warmup first samples.
        This is an ndarray.
        """
        chain = self._metachain.chain()
        return chain[warmup:, :]

    def chain_with_startpos(self):
        """Return the Markov chain including the start position."""
        return self._metachain.chain_with_startpos()

    @property
    def metachain(self):
        return self._metachain

    def ndim(self):
        return self._metachain.ndim()

    @property
    def total_runtime(self):
        return self._total_runtime

    def to_disk(self, path=None, filename=None, dataset_name=None, **kwargs):
        """Save the MCMC chain with metadata in an HDF5 file."""

        if path is not None:
            kwargs['path'] = path
        if filename is not None:
            kwargs['filename'] = filename
        if dataset_name is not None:
            kwargs['dataset_name'] = dataset_name
        kwargs['acceptance_rate'] = self.acceptance_rate()
        kwargs['ndim'] = self.ndim()
        kwargs['startpos'] = self._metachain.startpos()
        kwargs['mcmc_type'] = self.mcmc_type()
        kwargs['mcmc_type_uppercase'] = self.mcmc_type(uppercase=True)
        kwargs['mcmc_type_expand'] = self.mcmc_type(expand=True)
        kwargs['montepython_version'] = montepython.__version__
        kwargs['n_samples'] = len(self.chain())
        kwargs['total_runtime'] = self.total_runtime
        kwargs['hostname'] = socket.gethostname()

        # CPU hours
        try:
            kwargs['n_cores'] = int(os.environ['SLURM_NTASKS'])
        except KeyError:
            kwargs['n_cores'] = os.cpu_count()
        kwargs['cpu_hours'] = self.total_runtime * kwargs['n_cores'] / 3600

        for key, value in kwargs.items():
            kwargs[key] = value
        return mcmc_to_disk(self, **kwargs)

    def replace_bayes(self, bayes):
        self._bayes = bayes

    def run(self, n_samples):
        """Run the MCMC sampler for n_samples samples."""
        t = time.time()
        for i in range(n_samples):
            self.sample()
        self._total_runtime += time.time() - t

    def run_for(self, t_limit, unit='hours'):
        """
        Run the MCMC sampler for the specified length of time.
        Valid units are 'minutes', 'hours', and 'days'.
        """
        t_limit = convert_to_seconds(t_limit, unit)
        t_start = time.time()
        t_elapsed = 0
        while t_elapsed < t_limit:
            self.sample()
            t_elapsed = time.time() - t_start
        self._total_runtime += t_elapsed

    def batched_run_for(self, t_limit, n_batches, unit='hours', path=None, filename=None, dataset_name=None, acceptance_rate_limit=0.2, **metadata):
        for i in range(n_batches):
            t = time.time()
            self.run_for(t_limit)
            print(f'Finished batch in {time.time()-t:.0f} s')
            print(f'Acceptance rate: {self.acceptance_rate()}')
            path, filename, dataset_name = self.to_disk(path=path, filename=filename, dataset_name=dataset_name, mode='a', **metadata)
        if self.acceptance_rate() < acceptance_rate_limit:
            print(f'Low acceptance rate ({self.acceptance_rate()}), terminating')
            sys.exit(1)

    def sample(self):
        # PROPOSE NEW STATE
        current_state = self._metachain.head()
        proposed_state = self.propose_state(current_state)

        # ACCEPTANCE PROBABILITY
        current_lnprob = self.joint_lnprob(current_state)
        proposed_lnprob = self.joint_lnprob(proposed_state)
        diff = proposed_lnprob - current_lnprob
        metropolis_ratio = 1
        if 0 > diff:
            metropolis_ratio = np.exp(diff / self._temperature)
        acceptance_probability = metropolis_ratio

        # SAVE USER-SUPPLIED INFO ABOUT THE PROPOSED STATE
        for key, value in self._bayes.state_info().items():
            if key not in proposed_state.dict():
                proposed_state.set(key, value)

        # ACCEPT / REJECT
        if np.random.rand() < acceptance_probability:
            self._metachain.accept(proposed_state)
        else:
            self._metachain.reject()

    @abstractmethod
    def propose_state(self, current_state):
        """Propose new state."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def joint_lnprob(self, state):
        """Return the log probability of the (joint) state."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_ugly_string(self):
        """Return a description, suitable for filenames, of the MCMC object."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_pretty_string(self):
        """
        Return a description, suitable for titles etc, of the MCMC object.
        """
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def mcmc_type(self, uppercase=False, expand=False):
        """Return a string with the name of the MCMC algorithm."""
        raise NotImplementedError("Unimplemented abstract method!")
