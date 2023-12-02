# without dataframe
import numpy as np
import math
from itertools import product
from scipy.sparse import diags
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
x_max = 9 # xの最大値
x_step = 1  # 刻み幅を指定
x_list = np.arange(0, x_max + x_step, x_step)
M2 = 1 # EFTのcutoff scale
m2_list = M2 * (1 + x_list) 
m2_step = M2 * x_step
# spin
J_max = 40 # Jの最大値
J_list = np.arange(0, J_max, 2) # Jの値域
# spacetime dimension
d = 10
# input data
X_input = np.array(list(product(J_list, m2_list)), dtype=np.float64)


# g_k関数, null constraint, 各係数
# m2^{(2-d)/2} heavy averageの係数
def m2coeff(J, m2):
    return ( m2**((2-d)/2) )

def mathcalJ2(J):
    return J * (J + d - 3)

def g2(J, m2):
    return 1 / (m2 ** 2)

def g3(J, m2):
    return ( 3 - (4/(d-2)) * mathcalJ2(J) ) / ( m2 ** 3 )

def n4(J, m2):
    return ( mathcalJ2(J) * ( 2*mathcalJ2(J)- 5*d + 4 ) ) / ( m2 ** 4 )



# 各関数に対応するvectorを作成
def make_vector(func):
    vector = tf.convert_to_tensor(func(X_input[:, 0], X_input[:, 1]), dtype=tf.double)
    vector = tf.expand_dims(vector, axis=1)
    return vector

m2coeff_vector = make_vector(m2coeff)
g2_vector = make_vector(g2)
g3_vector = make_vector(g3)
n4_vector = make_vector(n4)


# heavy average
# 積分
def approx_integral(func_vector, rho_vector):
    integrand = m2coeff_vector * ( func_vector * rho_vector )
    # 1がm2_max個ならんだ小ブロックを作成
    block = np.ones(len(m2_list))
    # ブロック対角行列を作成
    integral_matrix = np.kron(np.eye(len(J_list)), block)

    return (1/m2_step) * tf.matmul( integral_matrix, integrand )



# normalization constant
def nJd(J):
    return ( (4*np.pi)**(d/2) * (d+2*J-3) * (np.vectorize(math.gamma)(d+J-3)) ) / ( np.pi * np.vectorize(math.gamma)((d-2)/2)*np.vectorize(math.gamma)(J+1) )
# 総和
def heavy_average(func_vector, rho_vector):
    return tf. reduce_sum( nJd(J=J_list) * approx_integral(func_vector, rho_vector) )



# model
n_node = 4
model = Sequential([
    Dense(n_node, activation='relu', input_dim=2, dtype=tf.double),
    Dense(1, activation='relu', dtype=tf.double)
])

# loss function
def custom_loss(y_true, y_pred):
        a2 = 1 
        a3 = 1
        w4 = 1
        
        rho_vecotor = model(X_input)

        g2_term = a2 * heavy_average(func_vector=g2_vector, rho_vector=rho_vecotor)
        g3_term = a3 * heavy_average(func_vector=g3_vector, rho_vector=rho_vecotor)
        n4_term =  w4 * tf.abs(heavy_average(func_vector=n4_vector, rho_vector=rho_vecotor))
        loss =  g2_term + g3_term + n4_term
        # loss =  g2_term + g3_term
        return loss

# compile
model.compile(optimizer=LegacyAdam(), loss=custom_loss)

initial_factor = 15
y_input = initial_factor * tf.ones(len(X_input), dtype=np.float64)
# y_input = tf.zeros(len(X_input), dtype=np.float64)


# 学習
n_epochs = 10
for i in range(1, n_epochs+1): # 反復回数
    history = model.fit(X_input, y_input, epochs=n_epochs, verbose=0)
    if i % (2) == 0:
        print(f'{i}th epoch')
        print(f'loss: {history.history["loss"]}')



# 学習曲線
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
#保存
plt.savefig('../data/loss.png')


plt.clf()

# 結果の描画
x = heavy_average(func_vector=g2_vector, rho_vector=model(X_input)).numpy()
y = heavy_average(func_vector=g3_vector, rho_vector=model(X_input)).numpy()
plt.plot(x,y, 'o', label='data')
plt.xlabel('g2')
plt.ylabel('g3')
plt.legend()
#保存
plt.savefig('../data/g2g3.png')