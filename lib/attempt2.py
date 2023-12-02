import numpy as np
import math
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
J_list = np.arange(0, J_max, 2) # Jの値域
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




# heavy averageの定義

# normalization constant
def nJd(d, J):
    return ( (4*np.pi)**(d/2) * (d+2*J-3) * (np.vectorize(math.gamma)(d+J-3)) ) / ( np.pi * np.vectorize(math.gamma)((d-2)/2)*np.vectorize(math.gamma)(J+1) )


def integrand(rho, func, J, m2):
    return (m2 ** ((2-d)/2)) * rho * func(d, m2, J)


def summand(rho, func, J, m2_list):
    # 台形近似で数値積分
    return np.trapz( integrand(rho, func, J, m2=m2_list), m2_list)

# heavy average
def heavy_average(rho, func):
    return  np.sum(nJd(d, J=J_list) * summand(rho, func, J=J_list, m2_list=m2_list))



# loss function
def custom_loss(y_true, y_pred):
    a2 = 1
    a3 = 1
    w4 = 10
    return a2 * heavy_average(rho=y_pred, func=g2) + a3 * heavy_average(rho=y_pred, func=g3) + w4 * tf.square( heavy_average(rho=y_pred, func=n4) )


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
    model.fit(input_data, tf.zeros(input_data.shape[0]), epochs=n_epochs, verbose=0)