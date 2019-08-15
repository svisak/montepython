#!/usr/bin/env python

import ctypes as c
import ctypes.util
import sys

class PyNSOPT():
    """
    This class provides an interface between Python and NSOPT.
    Example usage:

        pynsopt = PyNSOPT("ns-input.ini")
        # Either
        pynsopt.read_lec_vector_from_inifile()
        # Or pynsopt.set_lec_vector(lec_ndarray)
        pynsopt.calculate()
        print(pynsopt.get_theo_val(residual_number=0))
        pynsopt.terminate()
    """

    def __init__(self, *args, **kwargs):
        inifile = kwargs.pop('inifile', "ns-input.ini")
        inifile_path = kwargs.pop('inifile_path', ".")
        # FIND libnsopt.so
        libnsopt_so = c.util.find_library("nsopt")
        if libnsopt_so is None:
            print('libnsopt.so not found, make sure it is in your LD_LIBRARY_PATH', file=sys.stderr)
            sys.exit(1)

        # This is a DEFINE constant in the C/fortran-code so I can't access it, so define it here
        self._ininow = c.c_int(2147483647)

        # Will hold number of residuals
        self._n_residuals = None # Ordinary python int
        # Will hold number of parameters that are optimized
        self._n_parameters = None
        # Number of extra parameters, i.e. so called normalization constants (in this case none)
        self._n_extra_parameters = None

        # LOAD THE LIBRARY
        # residual_list is a derived fortran type so I can not use it directly in python.
        # That is why I created the global variable residual_list so that I have some space to use.
        # self._res will contain all needed information about the residuals (observables)
        self._nsopt = c.CDLL(libnsopt_so, c.RTLD_GLOBAL)
        self._res = self._nsopt.residual_mp_residual_list_
        self._nsopt_init(inifile, inifile_path)
        self._allocate_res()
        self._lec_vector = self._create_lec_vector()

    def _nsopt_init(self, inifile, inifile_path):
        # Initialize nsopt
        self._nsopt.program_startup_()

        # Load configuration
        self._nsopt.chp_ini_read_file_python_wrapper(self._nsopt.cfast_ini_get, str.encode(inifile_path), str.encode(inifile), self._ininow)

        # Prepare nsopt for calculations
        self._nsopt.program_initialization_()

    def load_configuration_option(self, option):
        """Add a single configuration option. Untested, use with caution."""
        self._nsopt.chp_ini_read_line(self._nsopt.cfast_ini_get, b".", str.encode(option), self._ininow)

    def _allocate_res(self):
        # Get the number of residuals, parameters, and extra parameters from the program
        res_nr = c.c_int(0)
        self._nsopt.get_nresiduals_(c.byref(res_nr))
        self._n_residuals = res_nr.value
        # Get number of parameters (LECs)
        par_nr = c.c_int(0)
        self._nsopt.pounders_param_get_nr_(c.byref(par_nr))
        self._n_parameters = par_nr.value
        # Get number of extra parameters
        self._nsopt.get_par_extra_nr_.restype = c.c_int
        self._n_extra_parameters = c.c_int(self._nsopt.get_par_extra_nr_())

        # Allocate res
        self._nsopt.residual_mp_allocate_residual_list_type_(c.byref(res_nr), c.byref(self._n_extra_parameters), self._res)

    def _create_lec_vector(self):
        par_vec_t = c.c_double * self._n_parameters
        par_vec = [0] * self._n_parameters
        lec_vector = par_vec_t(*par_vec)
        return lec_vector

    def terminate(self):
        self._nsopt.residual_mp_deallocate_residual_list_type_(self._res)
        self._nsopt.program_termination_()

    def read_lec_vector_from_inifile(self):
        tmp = c.c_double(0)
        for i in range(self._n_parameters):
            self._nsopt.pounders_param_get_value_(c.byref(c.c_int(i)), c.byref(tmp))
            self._lec_vector[i] = tmp.value
    
    def set_lec_vector(self, lec_ndarray):
        for i in range(self._n_parameters):
            self._lec_vector[i] = lec_ndarray[i]

    def get_lec_vector(self):
        # TODO Check if self._lec_vector is an ndarray :)
        return self._lec_vector

    def get_lec_name(self, lec):
        str_buf = c.create_string_buffer(str_buf_size)
        buf_size = 20
        self._nsopt.pounders_param_get_name_(c.byref(c.c_int(lec)), str_buf, buf_size)
        lec_name = str_buf.value.decode('ASCII').strip()
        return lec_name

    def calculate(self, lec_ndarray=None):
        """
        Calculate all included observables and amplitudes using the LECs in
        lec_ndarray (if provided), otherwise self._lec_vector (which can be set
        with set_lec_vector().) The resulting residual data is stored in
        self._res and should be accessed with the methods in this class.
        """
        if lec_ndarray is not None:
            self.set_lec_vector(lec_ndarray)
        par_nr = c.c_int(self._n_parameters)
        self._nsopt.calc_chi_squared_master_(self._lec_vector, self._res, c.byref(par_nr))

    def get_number_of_residuals(self):
        return self._n_residuals

    def get_number_of_parameters(self):
        return self._n_parameters

    def get_number_of_extra_parameters(self):
        return self._n_extra_parameters

    def get_residual_name(self, residual):
        str_buf = c.create_string_buffer(str_buf_size)
        buf_size = 20
        self._nsopt.residual_mp_get_residual_list_obs_(self._res, c.byref(c.c_int(residual+1)), str_buf)
        residual_name = str_buf.value.decode('ASCII').strip()
        return residual_name

    def get_theo_val(self, residual_number):
        """Return the theoretical (i.e. calculated) value."""
        return self._get_double_val(self._nsopt.residual_mp_get_residual_list_theo_val_, residual_number)

    def get_expe(self, residual):
        """Return experimental value."""
        return self._get_double_val(self._nsopt.residual_mp_get_residual_list_expe_, residual_number)

    def get_exp_err_val(self, residual):
        """Return experimental error."""
        return self._get_double_val(self._nsopt.residual_mp_get_residual_list_exp_err_val_, residual_number)

    def get_res_val(self, residual):
        """Return residual value."""
        return self._get_double_val(self._nsopt.residual_mp_get_residual_list_res_val_, residual_number)

    def get_theo_deriv(self, residual_number, lec_number):
        tmp = c.c_double(0)
        self._nsopt.residual_mp_get_residual_list_theo_deriv_(self._res, c.byref(c.c_int(residual_number+1)), c.byref(c.c_int(lec_number+1)), c.byref(tmp))
        return tmp.value
        
    def _get_double_val(self, res_getter_func, residual_number):
        tmp = c.c_double(0)
        res_getter_func(self._res, c.byref(c.c_int(residual_number+1)), c.byref(tmp))
        return tmp.value
