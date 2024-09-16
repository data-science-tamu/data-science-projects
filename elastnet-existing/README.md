# Physics-informed deep learning for elasticity: forward, inverse, and mixed problems
# Chun-Teh Chen and Grace Gu
# ggu@berkeley.edu

This package includes the following files:

1. elastnet.py - The code of ElastNet using TensorFlow library

2. data_incompressible - The 13 incompressible models studied in the paper
The folders are named as m_X_nu_05, where X is the pattern of the Young's modulus field
z1, z2, ..., z12 = the 12 Chinese zodiac patterns

3. data_compressible - The 146 compressible models studied in the paper
The folders are named as m_X_nu_Y, where X is the pattern of the Young's modulus field,
and Y is the pattern of the Poisson's ratio field
TML = the Mona Lisa pattern
TSN = the Starry Night pattern

##########################################################################################

# System requirements

1. Python
2. TensorFlow
3. numpy
4. sklearn

##########################################################################################

# Installation guide

No installation is required.

##########################################################################################

# Demo

To reproduce the results shown in Supplementary Figure 3:

1. Run elastnet.py

* The final prediction of the Young's modulus field will be written in y_pred_m_final
* The final prediction of the lateral displacement field will be written in y_pred_v_final
* The expected run time using a Nvidia Tesla v100 is about 2.5 hours
