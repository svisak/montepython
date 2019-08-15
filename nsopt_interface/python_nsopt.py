#!/usr/bin/python

# This module is used to interface with the program nsopt
import ctypes as c
from ctypes.util import find_library

libnsopt_path = find_library("nsopt")

# Load the library. Without RTLD_GLOBAL I get errors later when nsopt tries to use mkl-stuff
#nsopt = c.CDLL(b"/net/home/svisak/repos/nsopt/nucleon-scattering/libnsopt.so", c.RTLD_GLOBAL)
nsopt = c.CDLL("{}".format(libnsopt_path), c.RTLD_GLOBAL)

# This is a DEFINE constant in the C/fortran-code so I can't access it, so define it here
ininow = c.c_int(2147483647)

# STEP A -- Initialize nsopt

# Do some basic initializations
nsopt.program_startup_()

# STEP B -- Load configuration files

# The program nsopt is controlled by ini files, containing various configuration options
# The syntax of the ini files is self explanatory

# This line reads a configuration file in the directory "." (current directory)
# with the name "machlearn.ini".
nsopt.chp_ini_read_file_python_wrapper(nsopt.cfast_ini_get, b".", b"ns-input.ini", ininow)

# It is also possible to add a single configuration option at a time, if needed
#nsopt.chp_ini_read_line(nsopt.cfast_ini_get(), b".", b"run_file = foobar.run", ininow)

# Step C -- Prepare nsopt for calculations

# Initialize all variables and pre-compute stuff
nsopt.program_initialization_()

# Step C' -- Prepare python for calculations

# ctypes use special variables to represent integers and doubles used in the C/Fortran code
# c.c_int is a int, and c.c_double is a double

# Will hold number of residuals (i.e. number of observables calculated, in this case one)
res_nr = c.c_int(0)

# Will hold number of parameters that are optimized. In the current case nine.
par_nr = c.c_int(0)

# Number of extra parameters, i.e. so called normalization constants (in this case none)
par_extra_nr = c.c_int(0)

# Get the number of residuals from the program
nsopt.get_nresiduals_(c.byref(res_nr))
print('PYTHON res_nr = ', res_nr)

# Get the number of parameters
nsopt.pounders_param_get_nr_(c.byref(par_nr))

# Get the number of extra parameters
nsopt.get_par_extra_nr_.restype = c.c_int
par_extra_nr = c.c_int(nsopt.get_par_extra_nr_())

# residual_list is a derived fortran type so I can not use it directly in python.
# That is why I created the global variable residual_list so that I have some space to use.
# res will contain all needed information about the residuals (observables)
res = nsopt.residual_mp_residual_list_

# Allocate res
nsopt.residual_mp_allocate_residual_list_type_(c.byref(res_nr), c.byref(par_extra_nr), res)

# Create a vector of parameter values
par_vec_t = c.c_double * par_nr.value
par_vec = [0] * par_nr.value
par_vec_c = par_vec_t(*par_vec)
tmp = c.c_double(0)

# For inspecting parameter names
str_buf_size = 40
tmp_str_buf = c.create_string_buffer(str_buf_size)

print("PY: Number of parameters: {}".format(par_nr.value))
print("PY: Number of residuals: {}".format(res_nr.value))

# Step D -- Do calculations

# par_vec_c contains the numerical values to use for the coupling constants when doing
# calculations. Here, I set them to the values specified in the ini files:
for n in range(par_nr.value):
    nsopt.pounders_param_get_value_(c.byref(c.c_int(n)), c.byref(tmp))
    par_vec_c[n] = tmp.value

# This function calculates all included observables and amplitudes using the pars par_vec_c
# The resulting residual data is in res
for n in range(1):
    nsopt.calc_chi_squared_master_(par_vec_c, res, c.byref(par_nr))

# Print some relevant data
for n in range(res_nr.value):
    #print("PY: Residual %4d:" %(n))
    nsopt.residual_mp_get_residual_list_obs_(res, c.byref(c.c_int(n+1)), tmp_str_buf)
    residual_name = tmp_str_buf.value.decode('ASCII').strip()
    print("PY: Residual %4d, %s:" %(n, residual_name))

    # The theoretically calculated value
    nsopt.residual_mp_get_residual_list_theo_val_(res, c.byref(c.c_int(n+1)), c.byref(tmp))
    print("PY: \tTheoretical : %+12.5e" %(tmp.value))

    nsopt.residual_mp_get_residual_list_expe_(res, c.byref(c.c_int(n+1)), c.byref(tmp))
    print("PY: \tExperimental: %+12.5e" %(tmp.value))

    nsopt.residual_mp_get_residual_list_exp_err_val_(res, c.byref(c.c_int(n+1)), c.byref(tmp))
    print("PY: \tExp. error  : %+12.5e" %(tmp.value))

    nsopt.residual_mp_get_residual_list_res_val_(res, c.byref(c.c_int(n+1)), c.byref(tmp))
    print("PY: \tResidual    : %+12.5e" %(tmp.value))

    #nsopt.residual_mp_get_residual_list_theo_deriv_(res, c.byref(c.c_int(n+1)), c.byref(tmp))
    #print("PY: \ttheo_deriv    : %+12.5e" % (tmp.value))

    theo_deriv = []
    for i in range(par_nr.value):
        nsopt.residual_mp_get_residual_list_theo_deriv_(res, c.byref(c.c_int(n+1)), c.byref(c.c_int(i+1)), c.byref(tmp))
        theo_deriv.append(tmp.value)
    print("PY: \ttheo_deriv  ", theo_deriv)

print('\n===== LECs in this run =====')
for n in range(par_nr.value):
    nsopt.pounders_param_get_name_(c.byref(c.c_int(n)), tmp_str_buf, str_buf_size)
    tmp_str = tmp_str_buf.value.decode('ASCII').strip()
    print(tmp_str)
print('')

# Step E -- cleanup

# Clean up. Everything is not cleaned up. Since our program normally terminates after this, we
# have not cared so much about freeing everything and so on...
nsopt.residual_mp_deallocate_residual_list_type_(res)
nsopt.program_termination_()
