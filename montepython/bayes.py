class BayesBase(ABC):

    def __init__(self):
        self._lnlikelihood = None
        self._lnprior = None
        self._gradient = None

    def lnposterior(self):
        lnlikelihood_value = self.lnlikelihood()
        lnprior_value = self.lnprior()
        if np.isinf(lnprior_value):
            return lnprior_value
        elif np.isinf(lnlikelihood_value):
            return lnlikelihood_value
        lnposterior_value = lnprior_value + lnlikelihood_value
        if np.isnan(lnposterior_value):
            msg = 'NaN encountered in MCMC.lnposterior()'
            raise FloatingPointError(msg)
        return lnposterior_value

    def lnlikelihood(self):
        return self._lnlikelihood

    def lnprior(self):
        return self._lnprior

    def gradient(self):
        return self._gradient

    def lngradient(self):
        return self.gradient() / self.lnposterior()

    def set_lnlikelihood(self, val):
        self._lnlikelihood = val

    def set_lnprior(self, val):
        self._lnprior = val

    def set_gradient(self, val):
        self._gradient = val

    @abstractmethod
    def evaluate_pdf(self, position):
        raise NotImplementedError("Unimplemented abstract method!")
