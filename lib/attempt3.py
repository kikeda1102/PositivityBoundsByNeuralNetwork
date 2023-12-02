import numpy as np
import math
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# M1/m2 Mac上での警告回避
from tensorflow.compat.v1.keras.optimizers import Adam as LegacyAdam

tf.config.run_functions_eagerly(True) # eager executionを有効に
tf.data.experimental.enable_debug_mode() # なんか警告でたのでデバッグモードを有効に


# params
# energy
x_max = 10 # xの最大値
x_list = np.linspace(0, x_max, 11) # xの値域
m2 = 1 # EFTのcutoff scale
m2_list = m2 * (1 + x_list) 
# spin
J_max = 40 # Jの最大値
J_list = np.arange(0, J_max, 2) # Jの値域
# spacetime dimension
d = 10

# g_k関数, null constraint
def g2(m2, J):
    return 1 / (m2 ** 2)

def mathcalJ2(J):
    return J * (J + d - 3)

def g3(m2, J):
    return ( 3 - (4/(d-2)) * mathcalJ2(J) ) / ( m2 ** 3 )

def n4(m2, J):
    return ( mathcalJ2(J) * ( 2*mathcalJ2(J)- 5*d + 4 ) ) / ( m2 ** 4 )

# normalization constant
def nJd(J):
    return ( (4*np.pi)**(d/2) * (d+2*J-3) * (np.vectorize(math.gamma)(d+J-3)) ) / ( np.pi * np.vectorize(math.gamma)((d-2)/2)*np.vectorize(math.gamma)(J+1) )



# DataFrameを作成
df = pd.DataFrame(list(product(m2_list, J_list)), columns=['m2', 'J'])

# 各関数を DataFrame に格納
df['g2'] = g2(df['m2'], df['J'])
df['g3'] = g3(df['m2'], df['J'])
df['n4'] = n4(df['m2'], df['J'])

# for test 
df['rho'] = 1

# heavy average
def integrand(func, grouped_dfs):
    output_list = []
    for grouped_df in grouped_dfs:
        grouped_df['m2^((2-d)/2)'] = grouped_df['m2'] ** ((2-d)/2)
        output_list.append( grouped_df['m2^((2-d)/2)'] * grouped_df['rho'] * grouped_df[func] )
    return output_list

def summand(func, grouped_dfs):
    # integrandを、m2について台形近似で数値積分
    output_list = []
    for grouped_df in grouped_dfs:
        output_list.append( np.trapz( integrand(func, grouped_dfs), m2_list) )
    return output_list

def heavy_average(func, grouped_dfs):
    output_list = []
    for i in range(len(J_list)):
        output_list.append( nJd(J_list[i]) * summand(func, grouped_dfs)[i] )
    return tf.reduce_sum(output_list)


# model
n_node = 4
model = Sequential([
    Dense(n_node, activation='relu', input_dim=2),
    Dense(1, activation='relu')
])

# input data
X_input = df[['m2', 'J']].values
y_input = tf.zeros(len(df))

# loss function by GPT
def custom_loss(y_true, y_pred):
    a2 = 1
    a3 = 1
    w4 = 10

    df['rho'] = tf.convert_to_tensor( model(X_input) )
    grouped_dfs = [grouped_df for _, grouped_df in df.groupby('J')]
    loss = a2 * heavy_average(func='g2', grouped_dfs=grouped_dfs) + \
            a3 * heavy_average(func='g3', grouped_dfs=grouped_dfs) + \
            w4 * tf.square(heavy_average(func='n4', grouped_dfs=grouped_dfs))

    return loss


# compile
model.compile(optimizer=LegacyAdam(), loss=custom_loss)

# 学習
n_epochs = 300
for i in range(1, n_epochs+1): # 反復回数
    model.fit(X_input, y_input, epochs=n_epochs, verbose=0)