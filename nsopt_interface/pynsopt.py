#!/usr/bin/python

import ctypes as c
import ctypes.util
import sys

class PyNSOPT():

    def __init__(inifile, *args, **kwargs):
        # FIND libnsopt.so
        libnsopt_so = c.util.find_library("nsopt")
        if libnsopt_so is None:
            print('libnsopt.so not found, make sure it is in your LD_LIBRARY_PATH', file=sys.stderr)
        sys.exit(1)

        # Will hold number of residuals
        self._n_residuals = None # Ordinary python int
        # Will hold number of parameters that are optimized
        self._n_parameters = None
        # Number of extra parameters, i.e. so called normalization constants (in this case none)
        self._n_extra_parameters = None

        # residual_list is a derived fortran type so I can not use it directly in python.
        # That is why I created the global variable residual_list so that I have some space to use.
        # res will contain all needed information about the residuals (observables)
        self._res = None

        self._lec_vector = None

        # LOAD THE LIBRARY
        self.nsopt = c.CDLL(libnsopt_so, c.RTLD_GLOBAL)
        self._nsopt_init(inifile)

    def _nsopt_init(self, inifile):
        # This is a DEFINE constant in the C/fortran-code so I can't access it, so define it here
        ininow = c.c_int(2147483647)

        # STEP A -- Initialize nsopt
        # Do some basic initializations
        self.nsopt.program_startup_()

        # STEP B -- Load configuration files
        #nsopt.chp_ini_read_file_python_wrapper(nsopt.cfast_ini_get, b".", b"ns-input.ini", ininow)
        self.nsopt.chp_ini_read_file_python_wrapper(nsopt.cfast_ini_get, b".", inifile, ininow)
        # It is also possible to add a single configuration option at a time, if needed
        #nsopt.chp_ini_read_line(nsopt.cfast_ini_get(), b".", b"run_file = foobar.run", ininow)

        # Step C -- Prepare nsopt for calculations
        # Initialize all variables and pre-compute stuff
        self.nsopt.program_initialization_()

        # Step C' -- Prepare python for calculations
        # Get the number of residuals, parameters, and extra parameters from the program
        res_nr = c.c_int(0)
        par_nr = c.c_int(0)
        nsopt.get_par_extra_nr_.restype = c.c_int
        self._n_extra_parameters = c.c_int(self.nsopt.get_par_extra_nr_())
        self.nsopt.get_nresiduals_(c.byref(res_nr))
        self.nsopt.pounders_param_get_nr_(c.byref(par_nr))
        self._n_residuals = res_nr.value
        self._n_parameters = par_nr.value

        # Allocate res
        self._res = self.nsopt.residual_mp_residual_list_
        self.nsopt.residual_mp_allocate_residual_list_type_(c.byref(res_nr), c.byref(par_extra_nr), self._res)

        # Create a vector of parameter values
        par_vec_t = c.c_double * self._n_parameters
        par_vec = [0] * self._n_parameters
        self._lec_vector = par_vec_t(*par_vec)

    def tear_down(self):
        self.nsopt.residual_mp_deallocate_residual_list_type_(self._res)
        self.nsopt.program_termination_()

    def read_lec_vector_from_inifile(self):
        tmp = c.c_double(0)
        for i in range(self._n_parameters):
            nsopt.pounders_param_get_value_(c.byref(c.c_int(i)), c.byref(tmp))
            self._lec_vector[i] = tmp.value
    
    def set_lec_vector(self, lec_numpy_array):
        for i in range(self._n_parameters):
            self._lec_vector[i] = lec_numpy_array[i]

    def get_lec_vector(self):
        # TODO Check if self._lec_vector is an ndarray :)
        return self._lec_vector

    def get_lec_name(self, lec):
        str_buf = c.create_string_buffer(str_buf_size)
        buf_size = 20
        self.nsopt.pounders_param_get_name_(c.byref(c.c_int(lec)), str_buf, buf_size)
        lec_name = str_buf.value.decode('ASCII').strip()
        return lec_name

    def calculate(self):
        # This function calculates all included observables and amplitudes
        # using the LECs in self._lec_vector
        # The resulting residual data is in self._res
        par_nr = c.c_int(self._n_parameters)
        self.nsopt.calc_chi_squared_master_(self._lec_vector, self._res, c.byref(par_nr))

    def get_residual_name(self, residual):
        str_buf = c.create_string_buffer(str_buf_size)
        buf_size = 20
        self.nsopt.residual_mp_get_residual_list_obs_(self._res, c.byref(c.c_int(residual+1)), str_buf)
        residual_name = str_buf.value.decode('ASCII').strip()
        return residual_name

    def get_theo_val(self, residual):
        """Return the theoretical (i.e. calculated) value."""
        return _get_double_val(self.nsopt.residual_mp_get_residual_list_theo_val_)

    def get_expe(self, residual):
        """Return experimental value."""
        return _get_double_val(self.nsopt.residual_mp_get_residual_list_expe_)

    def get_exp_err_val(self, residual):
        """Return experimental error."""
        return _get_double_val(self.nsopt.residual_mp_get_residual_list_exp_err_val_)

    def get_res_val(self, residual):
        """Return residual value."""
        return _get_double_val(self.nsopt.residual_mp_get_residual_list_res_val_)

    def get_theo_deriv(self, residual, lec):
        tmp = c.c_double(0)
        self.nsopt.residual_mp_get_residual_list_theo_deriv_(self._res, c.byref(c.c_int(residual+1)), c.byref(c.c_int(lec+1)), c.byref(tmp))
        return tmp.value
        
    def _get_double_val(self, res_getter_func, residual):
        tmp = c.c_double(0)
        res_getter_func(self._res, c.byref(c.c_int(residual+1)), c.byref(tmp))
        return tmp.value
