import tensorflow as tf
import numpy as np
import time

from sklearn import preprocessing

# SETUP
num_neuron = 128
learn_rate = 0.001
path_to_data = "existing/elastnet/" # from data-science-project root

# For the mixed problem variant.
# v (nu; Poisson) = 0.5, Assumption of incompressible
# m (assumed to be youngs modulus field) = 
x_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_coord')
y_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_data')
x_elas = np.loadtxt('data_incompressible/m_rose_nu_05/strain_coord')
y_elas = np.loadtxt('data_incompressible/m_rose_nu_05/m_data')
