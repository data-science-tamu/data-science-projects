import tensorflow as tf
import numpy as np
import time

from sklearn import preprocessing

# Setup

num_neuron = 128
learn_rate = 0.001

x_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_coord')
y_disp = np.loadtxt('data_incompressible/m_rose_nu_05/disp_data')
x_elas = np.loadtxt('data_incompressible/m_rose_nu_05/strain_coord')
y_elas = np.loadtxt('data_incompressible/m_rose_nu_05/m_data')

ss_x = preprocessing.StandardScaler()
x_disp = ss_x.fit_transform(x_disp.reshape(-1, 2))
x_elas = ss_x.fit_transform(x_elas.reshape(-1, 2))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

logs_path = 'TensorBoard/'

# Define elasticity network

xs_disp = tf.placeholder(tf.float32, [None, 2])
ys_disp = tf.placeholder(tf.float32, [None, 2]) # u & v
xs_elas = tf.placeholder(tf.float32, [None, 2])

W_fc1a = weight_variable([2, num_neuron])
b_fc1a = bias_variable([num_neuron])
h_fc1a = tf.nn.relu(tf.matmul(xs_elas, W_fc1a) + b_fc1a)

W_fc2a = weight_variable([num_neuron, num_neuron])
b_fc2a = bias_variable([num_neuron])
h_fc2a = tf.nn.relu(tf.matmul(h_fc1a, W_fc2a) + b_fc2a)

W_fc3a = weight_variable([num_neuron, num_neuron])
b_fc3a = bias_variable([num_neuron])
h_fc3a = tf.nn.relu(tf.matmul(h_fc2a, W_fc3a) + b_fc3a)

W_fc4a = weight_variable([num_neuron, num_neuron])
b_fc4a = bias_variable([num_neuron])
h_fc4a = tf.nn.relu(tf.matmul(h_fc3a, W_fc4a) + b_fc4a)

W_fc5a = weight_variable([num_neuron, num_neuron])
b_fc5a = bias_variable([num_neuron])
h_fc5a = tf.nn.relu(tf.matmul(h_fc4a, W_fc5a) + b_fc5a)

W_fc6a = weight_variable([num_neuron, num_neuron])
b_fc6a = bias_variable([num_neuron])
h_fc6a = tf.nn.relu(tf.matmul(h_fc5a, W_fc6a) + b_fc6a)

W_fc7a = weight_variable([num_neuron, num_neuron])
b_fc7a = bias_variable([num_neuron])
h_fc7a = tf.nn.relu(tf.matmul(h_fc6a, W_fc7a) + b_fc7a)

W_fc8a = weight_variable([num_neuron, num_neuron])
b_fc8a = bias_variable([num_neuron])
h_fc8a = tf.nn.relu(tf.matmul(h_fc7a, W_fc8a) + b_fc8a)

W_fc9a = weight_variable([num_neuron, num_neuron])
b_fc9a = bias_variable([num_neuron])
h_fc9a = tf.nn.relu(tf.matmul(h_fc8a, W_fc9a) + b_fc9a)

W_fc10a = weight_variable([num_neuron, num_neuron])
b_fc10a = bias_variable([num_neuron])
h_fc10a = tf.nn.relu(tf.matmul(h_fc9a, W_fc10a) + b_fc10a)

W_fc11a = weight_variable([num_neuron, num_neuron])
b_fc11a = bias_variable([num_neuron])
h_fc11a = tf.nn.relu(tf.matmul(h_fc10a, W_fc11a) + b_fc11a)

W_fc12a = weight_variable([num_neuron, num_neuron])
b_fc12a = bias_variable([num_neuron])
h_fc12a = tf.nn.relu(tf.matmul(h_fc11a, W_fc12a) + b_fc12a)

W_fc13a = weight_variable([num_neuron, num_neuron])
b_fc13a = bias_variable([num_neuron])
h_fc13a = tf.nn.relu(tf.matmul(h_fc12a, W_fc13a) + b_fc13a)

W_fc14a = weight_variable([num_neuron, num_neuron])
b_fc14a = bias_variable([num_neuron])
h_fc14a = tf.nn.relu(tf.matmul(h_fc13a, W_fc14a) + b_fc14a)

W_fc15a = weight_variable([num_neuron, num_neuron])
b_fc15a = bias_variable([num_neuron])
h_fc15a = tf.nn.relu(tf.matmul(h_fc14a, W_fc15a) + b_fc15a)

W_fc16a = weight_variable([num_neuron, num_neuron])
b_fc16a = bias_variable([num_neuron])
h_fc16a = tf.nn.relu(tf.matmul(h_fc15a, W_fc16a) + b_fc16a)

W_fc17a = weight_variable([num_neuron, 1])
b_fc17a = bias_variable([1])
y_preda = tf.matmul(h_fc16a, W_fc17a) + b_fc17a
y_pred_m = y_preda[:, 0]

# Define displacement network

W_fc1b = weight_variable([2, num_neuron])
b_fc1b = bias_variable([num_neuron])
h_fc1b = tf.nn.swish(tf.matmul(xs_disp, W_fc1b) + b_fc1b)

W_fc2b = weight_variable([num_neuron, num_neuron])
b_fc2b = bias_variable([num_neuron])
h_fc2b = tf.nn.swish(tf.matmul(h_fc1b, W_fc2b) + b_fc2b)

W_fc3b = weight_variable([num_neuron, num_neuron])
b_fc3b = bias_variable([num_neuron])
h_fc3b = tf.nn.swish(tf.matmul(h_fc2b, W_fc3b) + b_fc3b)

W_fc4b = weight_variable([num_neuron, num_neuron])
b_fc4b = bias_variable([num_neuron])
h_fc4b = tf.nn.swish(tf.matmul(h_fc3b, W_fc4b) + b_fc4b)

W_fc5b = weight_variable([num_neuron, num_neuron])
b_fc5b = bias_variable([num_neuron])
h_fc5b = tf.nn.swish(tf.matmul(h_fc4b, W_fc5b) + b_fc5b)

W_fc6b = weight_variable([num_neuron, num_neuron])
b_fc6b = bias_variable([num_neuron])
h_fc6b = tf.nn.swish(tf.matmul(h_fc5b, W_fc6b) + b_fc6b)

W_fc7b = weight_variable([num_neuron, num_neuron])
b_fc7b = bias_variable([num_neuron])
h_fc7b = tf.nn.swish(tf.matmul(h_fc6b, W_fc7b) + b_fc7b)

W_fc8b = weight_variable([num_neuron, num_neuron])
b_fc8b = bias_variable([num_neuron])
h_fc8b = tf.nn.swish(tf.matmul(h_fc7b, W_fc8b) + b_fc8b)

W_fc9b = weight_variable([num_neuron, num_neuron])
b_fc9b = bias_variable([num_neuron])
h_fc9b = tf.nn.swish(tf.matmul(h_fc8b, W_fc9b) + b_fc9b)

W_fc10b = weight_variable([num_neuron, num_neuron])
b_fc10b = bias_variable([num_neuron])
h_fc10b = tf.nn.swish(tf.matmul(h_fc9b, W_fc10b) + b_fc10b)

W_fc11b = weight_variable([num_neuron, num_neuron])
b_fc11b = bias_variable([num_neuron])
h_fc11b = tf.nn.swish(tf.matmul(h_fc10b, W_fc11b) + b_fc11b)

W_fc12b = weight_variable([num_neuron, num_neuron])
b_fc12b = bias_variable([num_neuron])
h_fc12b = tf.nn.swish(tf.matmul(h_fc11b, W_fc12b) + b_fc12b)

W_fc13b = weight_variable([num_neuron, num_neuron])
b_fc13b = bias_variable([num_neuron])
h_fc13b = tf.nn.swish(tf.matmul(h_fc12b, W_fc13b) + b_fc13b)

W_fc14b = weight_variable([num_neuron, num_neuron])
b_fc14b = bias_variable([num_neuron])
h_fc14b = tf.nn.swish(tf.matmul(h_fc13b, W_fc14b) + b_fc14b)

W_fc15b = weight_variable([num_neuron, num_neuron])
b_fc15b = bias_variable([num_neuron])
h_fc15b = tf.nn.swish(tf.matmul(h_fc14b, W_fc15b) + b_fc15b)

W_fc16b = weight_variable([num_neuron, num_neuron])
b_fc16b = bias_variable([num_neuron])
h_fc16b = tf.nn.swish(tf.matmul(h_fc15b, W_fc16b) + b_fc16b)

W_fc17b = weight_variable([num_neuron, 1])
b_fc17b = bias_variable([1])
y_predb = tf.matmul(h_fc16b, W_fc17b) + b_fc17b
y_pred_v = y_predb[:, 0]

# Read displacements

y_u = ys_disp[:, 0]
y_v = y_pred_v

# Calculate strains

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'VALID')

u_matrix = tf.reshape(y_u, [257, 257])
v_matrix = tf.reshape(y_v, [257, 257])
u_matrix_4d = tf.reshape(u_matrix, [-1, 257, 257, 1])
v_matrix_4d = tf.reshape(v_matrix, [-1, 257, 257, 1])
conv_x = np.array([[[[-0.5]], [[-0.5]]], [[[0.5]], [[0.5]]], ])
conv_y = np.array([[[[0.5]], [[-0.5]]], [[[0.5]], [[-0.5]]], ])

conv_x = tf.constant(conv_x, dtype = tf.float32)
conv_y = tf.constant(conv_y, dtype = tf.float32)
y_e_xx = conv2d(u_matrix_4d, conv_x)
y_e_yy = conv2d(v_matrix_4d, conv_y)
y_r_xy = conv2d(u_matrix_4d, conv_y) + conv2d(v_matrix_4d, conv_x)
y_e_xx = 100*tf.reshape(y_e_xx, [-1])
y_e_yy = 100*tf.reshape(y_e_yy, [-1])
y_r_xy = 100*tf.reshape(y_r_xy, [-1])

# Define elastic tensors

c_matrix = (1/(1-(1/2.0)**2))*np.array([[1, 1/2.0, 0], [1/2.0, 1, 0], [0, 0, (1-1/2.0)/2.0]])
c_matrix = tf.constant(c_matrix, dtype = tf.float32)

# Calculate stresses

strain    = tf.stack([y_e_xx, y_e_yy, y_r_xy], axis = 1)
modulus   = tf.stack([y_pred_m, y_pred_m, y_pred_m], axis = 1)
stress    = tf.multiply(tf.matmul(strain, c_matrix), modulus)
stress_xx = stress[:, 0]
stress_yy = stress[:, 1]
stress_xy = stress[:, 2]

# Calculate sum of stresses

stress_xx_matrix = tf.reshape(stress_xx, [256, 256])
stress_yy_matrix = tf.reshape(stress_yy, [256, 256])
stress_xy_matrix = tf.reshape(stress_xy, [256, 256])

# Calculate sum of sub stresses

sum_conv = np.array([[[[1.0]], [[1.0]], [[1.0]]], [[[1.0]], [[1.0]], [[1.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])
y_pred_m_matrix = tf.reshape(y_pred_m, [256, 256])
y_pred_m_matrix_4d = tf.reshape(y_pred_m_matrix, [-1, 256, 256, 1])
y_pred_m_conv = conv2d(y_pred_m_matrix_4d, sum_conv)
stress_xx_matrix_4d = tf.reshape(stress_xx_matrix, [-1, 256, 256, 1])
stress_yy_matrix_4d = tf.reshape(stress_yy_matrix, [-1, 256, 256, 1])
stress_xy_matrix_4d = tf.reshape(stress_xy_matrix, [-1, 256, 256, 1])
wx_conv_xx = np.array([[[[-1.0]], [[-1.0]], [[-1.0]]], [[[0.0]], [[0.0]], [[0.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])
wx_conv_xy = np.array([[[[1.0]], [[0.0]], [[-1.0]]], [[[1.0]], [[0.0]], [[-1.0]]],
[[[1.0]], [[0.0]], [[-1.0]]], ])
wy_conv_yy = np.array([[[[1.0]], [[0.0]], [[-1.0]]], [[[1.0]], [[0.0]], [[-1.0]]],
[[[1.0]], [[0.0]], [[-1.0]]], ])
wy_conv_xy = np.array([[[[-1.0]], [[-1.0]], [[-1.0]]], [[[0.0]], [[0.0]], [[0.0]]],
[[[1.0]], [[1.0]], [[1.0]]], ])

wx_conv_xx = tf.constant(wx_conv_xx, dtype = tf.float32)
wx_conv_xy = tf.constant(wx_conv_xy, dtype = tf.float32)
wy_conv_yy = tf.constant(wy_conv_yy, dtype = tf.float32)
wy_conv_xy = tf.constant(wy_conv_xy, dtype = tf.float32)
fx_conv_xx = conv2d(stress_xx_matrix_4d, wx_conv_xx)
fx_conv_xy = conv2d(stress_xy_matrix_4d, wx_conv_xy)
fx_conv_sum = fx_conv_xx + fx_conv_xy
fy_conv_yy = conv2d(stress_yy_matrix_4d, wy_conv_yy)
fy_conv_xy = conv2d(stress_xy_matrix_4d, wy_conv_xy)
fy_conv_sum = fy_conv_yy + fy_conv_xy
fx_conv_sum_norm = tf.divide(fx_conv_sum, y_pred_m_conv)
fy_conv_sum_norm = tf.divide(fy_conv_sum, y_pred_m_conv)

# Calculate loss

mean_modu = tf.constant(np.mean(y_elas), dtype = tf.float32)
loss_x = tf.reduce_mean(tf.abs(fx_conv_sum_norm))
loss_y = tf.reduce_mean(tf.abs(fy_conv_sum_norm))
loss_m = tf.abs(tf.reduce_mean(y_pred_m) - mean_modu)
loss_v = tf.abs(tf.reduce_mean(y_pred_v))
loss = loss_x + loss_y + loss_m/100 + loss_v/100
err = tf.reduce_sum(tf.abs(y_elas - y_pred_m))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training process

start_time  = time.time()
for i in range(200001):
    sess.run(train_step, feed_dict = {xs_elas: x_elas, xs_disp: x_disp, ys_disp: y_disp})
    if i % 100 == 0:
        err_vale = np.array(sess.run(err, feed_dict = {xs_elas: x_elas}))
        loss_vale = np.array(sess.run(loss, feed_dict = {xs_elas: x_elas, xs_disp: x_disp, ys_disp: y_disp}))
        print(i, loss_vale, err_vale)
y_pred_m_value = sess.run(y_pred_m, feed_dict = {xs_elas: x_elas})
y_pred_v_value = sess.run(y_pred_v, feed_dict = {xs_disp: x_disp})
np.savetxt('y_pred_m_final', y_pred_m_value)
np.savetxt('y_pred_v_final', y_pred_v_value)
print("--- %s Elapsed time ---" % (time.time() - start_time))
