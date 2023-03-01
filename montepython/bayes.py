from abc import ABC, abstractmethod
import numpy as np

class Bayes(ABC):
    """Abstract Bayes class. As a user, you should implement your likelihood
    and prior here. If you're using the HMC algorithm you must also implement
    the gradient of the negative log posterior.

    Do this by extending the class and implementing the abstract 'evaluate'
    method, wherein you set the current values of the log likelihood and the
    log prior as well as the gradient of the negative log posterior.
    These are set with the "set_{}_value" methods.

    Feel free to add any methods you like to your inheriting class.
    """

    def __init__(self):
        self._lnlikelihood_value = None
        self._lnprior_value = None
        self._nlp_gradient_value = None

    def get_lnposterior_value(self):
        lnlikelihood_value = self.get_lnlikelihood_value()
        lnprior_value = self.get_lnprior_value()
        if np.isinf(lnprior_value):
            return lnprior_value
        elif np.isinf(lnlikelihood_value):
            return lnlikelihood_value
        lnposterior_value = lnprior_value + lnlikelihood_value
        if np.isnan(lnposterior_value):
            msg = 'NaN encountered in Bayes.get_lnposterior_value()'
            raise FloatingPointError(msg)
        return lnposterior_value

    def get_lnlikelihood_value(self):
        return self._lnlikelihood_value

    def get_lnprior_value(self):
        return self._lnprior_value

    def get_nlp_gradient_value(self):
        return self._nlp_gradient_value

    def set_lnlikelihood_value(self, val):
        self._lnlikelihood_value = val

    def set_lnprior_value(self, val):
        self._lnprior_value = val

    def set_nlp_gradient_value(self, val):
        self._nlp_gradient_value = val

    def state_info(self):
        """
        If this method returns a dictionary, each item in the dictionary will
        be saved in the State object that stores the current MCMC sample.
        Use it to save information that varies depending on where in the
        parameter space the sample is taken.
        """
        return {}

    @abstractmethod
    def evaluate(self, position):
        """
        As a user, this is the method you must implement. Calculate the
        log prior and log likelihood values as well as the gradient of the
        negative log posterior (nlp), and set them with the provided set
        methods. The gradient is only needed for HMC, leave it unset for RWM.
        """
        raise NotImplementedError("Unimplemented abstract method!")


class SimpleBayes(ABC):
    """Abstract Bayes class. As a user, you should implement your posterior
    here. If you're using the HMC algorithm you must also implement
    the gradient of either the log posterior or the negative log posterior.

    Do this by extending the class and implementing the abstract 'evaluate'
    method, wherein you set the current values of the log likelihood and the
    log prior as well as the gradient of the negative log posterior.
    These are set with the "set_{}_value" methods.

    Feel free to add any methods you like to your inheriting class.
    """

    def __init__(self):
        self._lnposterior_value = None
        self._nlp_gradient_value = None

    def get_lnposterior_value(self):
        return self._lnposterior_value

    def get_nlp_gradient_value(self):
        return self._nlp_gradient_value

    def set_lnposterior_value(self, val):
        self._lnposterior_value = val

    def set_nlp_gradient_value(self, val):
        self._nlp_gradient_value = val

    def set_lnposterior_gradient(self, val):
        self.set_nlp_gradient_value(-val)

    def state_info(self):
        """
        If this method returns a dictionary, each item in the dictionary will
        be saved in the State object that stores the current MCMC sample.
        Use it to save information that varies depending on where in the
        parameter space the sample is taken.
        """
        return {}

    @abstractmethod
    def evaluate(self, position):
        """
        As a user, this is the method you must implement. Calculate the
        log prior and log likelihood values as well as the gradient of the
        negative log posterior (nlp), and set them with the provided set
        methods. The gradient is only needed for HMC, leave it unset for RWM.
        """
        raise NotImplementedError("Unimplemented abstract method!")
