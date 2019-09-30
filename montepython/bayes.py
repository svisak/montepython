from abc import ABC, abstractmethod
import numpy as np

class BayesBase(ABC):

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
            msg = 'NaN encountered in BayesBase.get_lnposterior_value()'
            raise FloatingPointError(msg)
        return lnposterior_value

    def get_nlp_value(self):
        return -self.get_lnposterior_value()

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

    @abstractmethod
    def evaluate(self, position):
        raise NotImplementedError("Unimplemented abstract method!")
