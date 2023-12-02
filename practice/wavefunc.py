# 永井さんの記事を参考に 完成版

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

# Hamiltonianの定義
def make_T(M):
    T = np.zeros((M-1, M-1))
    for i in range(M-1):
        j = i - 1
        if 0 <= j < M-1:
            T[i, j] = 1
        j = i + 1
        if 0 <= j < M-1:
            T[i, j] = 1
        T[i, i] = -2
    return T

def make_V(M, ω, xs):
    V = np.zeros((M-1, M-1))
    for i in range(M-1):
        x = xs[i]
        V[i, i] = (1/2) * ω**2 * x**2
    return V

# params
M = 512 # メッシュ数
xmax = 5
h = 2 * xmax / M
xs = np.zeros(M-1)
for j in range(1, M):
    xs[j-1] = -xmax + h*j

T = make_T(M)
ω = 1
V = make_V(M, ω, xs)

# Hamiltonian
H = -T / (2 * h**2) + V

# 固有値を求める
e_eigen, v = np.linalg.eigh(H)
groundstate_energy = e_eigen[0] # 最小の固有値
print(f"ground state energy: {groundstate_energy}")

#  NNの定義
n1 = 4 # 隠れ層のノード数
model = Sequential([
    Dense(n1, activation='softplus', input_shape=(1,)),
    Dense(1, activation='softplus'),
])

# エネルギー期待値を求める関数 
def energy_EV(model, xs, H):
    psi = model(xs)
    psi_tf = tf.convert_to_tensor(psi, dtype=tf.float32)  # NumPyをTensorFlowのテンソルに変換
    H_tf = tf.convert_to_tensor(H, dtype=tf.float32)  # Hをfloat32に変換
    c = tf.reduce_sum(psi_tf**2)  # psiのノルムの二乗
    E = tf.reduce_sum(tf.matmul(tf.transpose(psi_tf), tf.matmul(H_tf, psi_tf))) / c
    return E

# カスタム損失関数
def custom_loss(y_val, y_pred):
    e = energy_EV(model = model, xs = xs, H = H)
    loss = e - groundstate_energy
    return loss


# 学習を行う関数
def main():
    opt = LegacyAdam(learning_rate=0.001)  # M1/M2 macに対する警告によりlegacy Adamを使用
    # コンパイル
    model.compile(optimizer=opt, loss=custom_loss)
    lossdata = []
    
    # 学習
    n_iteration = 300
    for i in range(1, n_iteration+1): # 反復回数
        model.fit(x = np.expand_dims(xs, axis=-1), y = np.zeros(M-1), epochs=1, verbose=0)

        # loss functionの値を計算
        e = energy_EV(model, xs, H)
        loss = e - groundstate_energy
        lossdata.append(loss)
        if i % 100 == 0:
            print(f"i = {i}, energy = {e}, loss = {loss}")

    # 結果の表示
    plt.plot(lossdata)
    plt.yscale('log')
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig("data/lossdata.png")

    # 波動関数の表示
    plt.clf()
    psi = model(xs)
    plt.plot(xs, psi)
    plt.xlabel("x")
    plt.ylabel("psi")
    plt.savefig("data/wavefunc.png")


# 実行
if __name__ == '__main__':
    main()
