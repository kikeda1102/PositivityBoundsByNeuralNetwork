# Positivity bounds calculation by NN

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# M1/M2 Mac上での警告回避
from tensorflow.compat.v1.keras.optimizers import Adam as LegacyAdam

tf.config.run_functions_eagerly(True) # eager executionを有効に
tf.data.experimental.enable_debug_mode() # なんか警告でたのでデバッグモードを有効に


# params

# energy
x_max = 100 # xの最大値
x_list = np.linspace(0, x_max, 10) # xの値域
M2 = 1 # EFTのcutoff scale
m2_list = M2 * (1 + x_list) 
# spin
J_max = 40 # Jの最大値
J_list = np.arange(0, J_max+1, 2) # Jの値域

# spacetime dimension
d = 10

# input data
from itertools import product
input_data = np.array(list(product(m2_list, J_list)))



# g_k関数, null constraint

def g2(d, m2, J):
    return 1 / (m2 ** 2)

def mathcalJ2(d, J):
    return J * (J + d - 3)

def g3(d, m2, J):
    return ( 3 - (4/(d-2)) * mathcalJ2(d, J) ) / ( m2 ** 3 )

def n4(d, m2, J):
    return ( mathcalJ2(d, J) * ( 2*mathcalJ2(d, J)- 5*d + 4 ) ) / ( m2 ** 4 )



# outputs
def g_n_vector(d, m2, J):
    return np.array([g2(d, m2, J), g3(d, m2, J), n4(d, m2, J)])

# coefficients
def coeffs_vector(A, c4):
    return np.array([-A, 1, c4])

# constraint
def constraint(d, m2, J, A, c4):
    return g_n_vector(d, m2, J) @ coeffs_vector(A, c4)

# step function
def step(d, m2, J, A, c4):
    return np.where( constraint(d, m2, J, A, c4) < 0, 1, 0) # constraintが負なら1, 正なら0

# loss function
def custom_loss(A, c4):
    loss_list = [1/A]
    for m2 in m2_list:
        for J in J_list:
            loss_list.append(tf.reduce_sum(tf.where(constraint(d, m2, J, A, c4) < 0, 1, 0)))
    return tf.reduce_sum(loss_list)


# model
n_node = 4
model = Sequential([
    Dense(n_node, activation='relu', input_dim=2),
    Dense(1, activation='relu')
])

model.compile(optimizer=LegacyAdam(), loss=custom_loss)



# 学習
n_epochs = 300
for i in range(1, n_epochs+1): # 反復回数
    model.fit(input_data, tf.zeros(input_data.shape), epochs=n_epochs, verbose=0)