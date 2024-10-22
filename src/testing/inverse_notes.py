"""
From the provided elastnet.py

# For the strain:
def conv2d(x, W):
    return tf.nn.conv2d(
        tensor, convolution_matrix, strides=[1,1,1,1],padding='VALID'
    )
strain is calculated using conv 2d;
conv_x = [
    [-0.5, -0.5],
    [0.5, 0.5]]
conv_y = [
    [0.5, -0.5],
    [0.5, -0.5]]
u_mat = axial displacement? (u_x)
v_mat = lateral displacement? (u_y)

# equation (1)
e_xx = conv2d(u_mat, conv_x) # epsilon_xx
e_yy = conv2d(v_mat, conv_y) # epsilon_yy
r_xy = conv2d(u_mat, conv_y) + conv2d(v_mat, conv_x) # gamma_xy

Values are then adjusted
e_xx, e_yy, r_xy = 100 * reshape to 1D ([-1])
# Will likely just use the given strain data (no lateral? data given)
# for the inverse problem

# # For the elasticity
# # Elastic Constitutive Relation, equation (2)
# What the strain {} is multiplied by
ecr_matrix = ( 1 / (1 - v^2)) * [[ 1,  v,     0   ],
                                 [ v,  1,     0   ],
                                   0,  0, (1-v)/2 ]]
# Right side {e_xx, e_yy, r_xy}
strain_stack = stack([e_xx, e_yy, e_xy], axis = 1) # Or just the strain file

# The E value
y_mod_stack = stack([pred_m, pred_m, pred_m], axis = 1) # pred_m is the youngs modulus

# What to do with v, poisson's (especially the square) (guess below)
v_stack = stack([v_pred, v_pred, v_pred], axis = 1)
for the outside one, can do .multiply(v_stack, v_stack)

# The fraction in front.
=> nn.divide(y_mod_stack, 1 - nn.multiply(v_stack, v_stack))

# How to sub v into the matrix.
# Could try
ecr_matrices = [
    [[ 1,  v,     0   ],
     [ v,  1,     0   ],
       0,  0, (1-v)/2 ]], for v in v_stack[:, 0]
]
# ?
# need to test value results (with v = 0.5)

## Once stress is obtained:
stress_xx = stress[:, 0]
stress_yy = stress[:, 1]
stress_xy = stress[:, 2]
# All 3 are reshaped to 256 by 256

"""
