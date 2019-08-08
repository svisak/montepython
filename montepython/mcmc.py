#!/usr/bin/env python

import sys
from abc import ABC, abstractmethod
import numpy as np

class MCMC(ABC):
    """
    Abstract MCMC base class. All actual sampling classes (such as HMC)
    inherits the methods in this class, and implement the abstract methods.

    All pertinent information about the sampler, including the Markov chain
    itself, can be accessed with the methods provided here.

    :param **kwargs:
        All parameters for the MCMC sampler are supplied as keyword arguments.
        See below for a list of common (and mandatory) parameters. Please see
        the documentation for the specific sampler you want to use to learn
        about parameters specific to that sampler.

        Common
        ------
        dim:
            The number of dimensions in parameter space.

        startpos:
            The starting position of the Markov chain. This should be a
            vector of length dim.

        lnprior:
            A function that takes a vector in the parameter space as input and
            returns the natural logarithm of the prior probability for that
            position.

        lnlikelihood:
            A function that takes a vector in the parameter space as input and
            returns the natural logarithm of the prior probability for that
            position.

        save_to_disk:
            Whether to save the Markov chain to disk. Default is False.

        batch_size:
            If set, the sampling will be performed in bathes of size batch_size.
            The chain will be saved to disk after every batch if save_to_disk
            is True.

        batch_filename:
            The file to which the chain will be backed up if save_to_disk is True.
            A default, "mcmc_chain_backup", is provided.

    """

    def __init__(self,  args=[], **kwargs):
        dim = kwargs.pop('dim')
        startpos = kwargs.pop('startpos')
        self.lnprior = kwargs.pop('lnprior')
        self.lnlikelihood = kwargs.pop('lnlikelihood')
        self._save_to_disk = kwargs.pop('save_to_disk', False)
        self._batch_size = kwargs.pop('batch_size', sys.maxsize)
        self._batch_filename = kwargs.pop('batch_filename', 'mcmc_chain_backup')
        self._metachain = MetaChain(dim, startpos)

    def set_seed(self, seed):
        """Set the numpy random seed."""
        np.random.seed(seed)

    def acceptance_fraction(self):
        """Return the acceptance fraction of the samples so far."""
        return self._metachain.acceptance_fraction()

    def chain(self):
        """Return the Markov chain resulting from the sampling. This is an ndarray."""
        return self._metachain.chain()

    def lnposterior(self, position):
        # CONVENIENCE
        lnprior_val = self.lnprior(position)
        lnlikelihood_val = self.lnlikelihood(position)

        # CHECK INF/NAN
        if np.isinf(lnprior_val):
            return lnprior_val
        elif np.isinf(lnlikelihood_val):
            return lnlikelihood_val
        if np.isnan(lnprior_val) or np.isnan(lnlikelihood_val):
            raise ValueError('NaN encountered in lnposterior')

        # OK, RETURN LN OF POSTERIOR
        return lnprior_val + lnlikelihood_val

    def save_batch(self):
        if self._save_to_disk:
            print('Backing up chain to file {}'.format(self._batch_filename))
            np.save(self._batch_filename, chain())
        else:
            pass

    def run(self, n_samples):
        """
        Run the MCMC algorithm.

        :param n_samples: Number of MCMC steps to perform
        :returns: Nothing
        """
        self._metachain.extend(n_samples)
        while n_samples > self._batch_size:
            self.run_batch(self._batch_size)
            n_samples -= self._batch_size
            self.save_batch()
        self.run_batch(n_samples)
        self.save_batch()

    def run_batch(self, batch_size):
        for i in range(batch_size):
            self.sample()

    @abstractmethod
    def sample(self):
        """Obtain a new sample from the distribution."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_ugly_string(self):
        """Return a description, suitable for filenames, of the MCMC object."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def to_pretty_string(self):
        """Return a description, suitable for titles etc, of the MCMC object."""
        raise NotImplementedError("Unimplemented abstract method!")

    @abstractmethod
    def get_mcmc_type(self):
        """Return a string with the name of the MCMC algorithm."""
        raise NotImplementedError("Unimplemented abstract method!")


class MetaChain():
    """
    This class is used to store relevant data about the sample process, such as
    the resulting Markov chain as a numpy ndarray and the acceptance fraction.
    It also handles updating of the chain after each sample.

    Users should not interact with this class directly.

    """

    def __init__(self, dim, startpos=None):
        self._dim = dim
        self._chain = np.empty((0, self._dim))
        self._n_accepted = 0
        self._index = -1
        if startpos is not None:
            self.extend(1)
            self.accept(startpos)

    def reset(self, startpos=None):
        if startpos is None:
            self.__init__(self._dim, self.startpos())
        else:
            self.__init__(self._dim, startpos)

    def accept(self, position):
        """Add position to the chain and increment the index and accepted samples."""
        self._chain[self._index+1, :] = position
        self._n_accepted += 1
        self._index += 1

    def reject(self):
        """Copy the previous head to the new head, and increment index."""
        self._chain[self._index+1, :] = self.head()
        self._index += 1

    def head(self):
        """Return the latest sample in the chain."""
        return self._chain[self._index, :]

    def startpos(self):
        """Return the first sample in the chain."""
        return self._chain[0, :]

    def chain(self):
        """Return the Markov chain."""
        return self._chain

    def extend(self, n):
        """Extend _chain to accommodate upcoming samples."""
        self._chain = np.concatenate((self._chain, np.zeros((n, self._dim))))

    def acceptance_fraction(self):
        """Return the current acceptance fraction."""
        return self._n_accepted / (self._index+1)

    def dimensionality(self):
        """
        Return the dimensionality of the problem, i.e.
        the number of sampled parameters.

        """
        return self._dim

    def n_samples_taken(self):
        """
        Return the number of samples in the chain at this moment.
        This can be useful for monitoring progress, and for unit testing.
        """
        return self._index+1
