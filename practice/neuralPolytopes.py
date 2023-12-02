import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1.keras.optimizers import Adam as LegacyAdam

# Tensorflow config
tf.config.run_functions_eagerly(True)  # eager executionを有効に
tf.data.experimental.enable_debug_mode()  # なんか警告でたのでデバッグモードを有効に

# matplotlib config
plt.rcParams['figure.figsize'] = [8, 4]

# params
d = 2  # dimension
p_list = [0.8, 1, 2, 8] # pの値
# p_list = [0.8] # pの値


# training dataの作成
# S^{d-1}上のランダムな点をN個生成
N = 2000
X = tf.random.normal((N, d))
X_train = X / tf.linalg.norm(X, axis=-1, keepdims=True)  # normalize
# 返り値は全て1
y_train = np.ones(N)

# カスタム活性化関数の定義
def custom_activation(x):
    return tf.pow(tf.reduce_sum(tf.abs(x) ** p, axis=-1, keepdims=True), 1/p)


def main(p, ax):
    # モデルの定義
    # ハイパーパラメータ
    batch_size = 1000
    epochs = 1000
    n_units = 2  # 各層のユニット数

    model = Sequential([
        Dense(n_units, input_dim=d, activation=custom_activation),
        Dense(1, activation=custom_activation)
    ])

    # モデルのコンパイル
    model.compile(optimizer=LegacyAdam(),
                loss='mean_squared_error')

    # モデルのトレーニング
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    # 結果の図示
    N_test = 10000
    # X_testは、R^d上のランダムな点
    X_test = tf.random.normal((N_test, d))

    y_pred = model.predict(X_test)

    # y_predが1に近い点を表示
    test_list = []
    for i in range(N_test):
        if 0.99 < y_pred[i] < 1.01:
            test_list.append(X_test[i])

    test_list = np.array(test_list)

    # subplot 1: 学習曲線
    ax[0].plot(history.history['loss'])
    ax[0].ylim = (0, max(history.history['loss']))
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].axis('equal')  # アスペクト比

    # subplot 2: Polytope
    ax[1].scatter(test_list[:, 0], test_list[:, 1], s=4)
    ax[1].legend([f'p = {p}'])
    ax[1].axis('equal')

    return ax

# 実行
if __name__ == '__main__':
    fig, axes = plt.subplots(len(p_list), 2, figsize=(8, 4 * len(p_list)))

    for i, p in enumerate(p_list):
        axes[i] = main(p, axes[i])

    # レイアウトの調整
    plt.tight_layout()

    # 保存
    plt.savefig('data/neuralPolytopes_combined.png', bbox_inches='tight')
