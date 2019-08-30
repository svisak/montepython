from abc import ABC, abstractmethod
import numpy as np
import h5py
import sys

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
        ndim:
            The number of dimensions in parameter space.

        startpos:
            The starting position of the Markov chain. This should be a
            vector of length ndim.

        lnprior:
            A function that takes a vector in the parameter space as input and
            returns the natural logarithm of the prior probability for that
            position.

        lnlikelihood:
            A function that takes a vector in the parameter space as input and
            returns the natural logarithm of the prior probability for that
            position.

    """

    def __init__(self,  args=[], **kwargs):
        ndim = kwargs.pop('ndim')
        startpos = kwargs.pop('startpos')
        self.lnprior = kwargs.pop('lnprior')
        self.lnlikelihood = kwargs.pop('lnlikelihood')
        self._metachain = MetaChain(ndim, startpos)

        # USED INTERNALLY TO REMEMBER PREVIOUSLY COMPUTED POSTERIOR VALUE
        self._current_value = None

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

    def _remember_value(self, value):
        self._current_value = value

    def _recall_value(self):
        return self._current_value

    def run(self, n_samples):
        """Run the MCMC sampler for n_samples samples."""
        self._metachain.extend(n_samples)
        for i in range(n_samples):
            self.sample()

    def to_disk(self, filename=None, dataset_name=None, *args, **kwargs):
        """Save the MCMC chain with metadata in an HDF5 file."""
        if filename is None:
            filename = 'out.hdf5'
        if dataset_name is None:
            dataset_nate = self.to_ugly_string()
        f = h5py.File(filename)
        dset = f[dataset_name]
        dset[...] = self.chain()
        dset.attrs['acceptance_fraction'] = self.acceptance_fraction()
        dset.attrs['ndim'] = self._metachain.dimensionality()
        dset.attrs['startpos'] = self._metachain.startpos()
        dset.attrs['mcmc_type'] = self.get_mcmc_type()
        dset.attrs['montepython_version'] = montepython.__version__
        for key, value in kwargs.items():
            dset.attrs[key] = value

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

    def __init__(self, ndim, startpos):
        self._ndim = ndim
        self._chain = np.empty((0, self._ndim))
        self._n_accepted = 0
        self._index = -1

        # ADD START POSITION
        self.extend(1)
        self.accept(startpos)

    def reset(self):
        self.__init__(self._ndim, self.startpos())

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
        self._chain = np.concatenate((self._chain, np.empty((n, self._ndim))))

    def acceptance_fraction(self):
        """Return the current acceptance fraction."""
        return self._n_accepted / (self._index+1)

    def dimensionality(self):
        """
        Return the dimensionality of the problem, i.e.
        the number of sampled parameters.

        """
        return self._ndim

    def chain_length(self):
        """
        Return the length of the ndarray. This may include elements
        that are not yet filled.
        """
        return self.chain().shape[0]

    def n_samples_taken(self):
        """
        Return the number of samples in the chain at this moment.
        This can be useful for monitoring progress, and for unit testing.
        """
        return self._index+1
