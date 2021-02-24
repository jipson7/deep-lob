import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Reshape, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

np.random.seed(1)
tf.random.set_seed(2)


def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    # build the last LSTM layer
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)

    # build the output layer
    out = Dense(3, activation='softmax')(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    HORIZON = 10
    horizons = {
        1: 0,
        2: 1,
        3: 2,
        5: 3,
        10: 4
    }
    horizon_idx = horizons[HORIZON]
    data_path = 'F:/Datasets/FI2010/deeplob/'

    dec_train = np.loadtxt(data_path + 'Train_Dst_NoAuction_DecPre_CF_6.txt')
    dec_test1 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_6.txt')
    dec_test2 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_7.txt')
    dec_test3 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_8.txt')
    dec_test4 = np.loadtxt(data_path + 'Test_Dst_NoAuction_DecPre_CF_9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3, dec_test4))

    # extract limit order book data from the FI-2010 dataset
    train_lob = prepare_x(dec_train)
    test_lob = prepare_x(dec_test)

    # extract label from the FI-2010 dataset
    train_label = get_label(dec_train)
    test_label = get_label(dec_test)

    # prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
    trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=100)
    trainY_CNN = trainY_CNN[:, horizon_idx] - 1
    trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

    # prepare test data.
    testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=100)
    testY_CNN = testY_CNN[:, horizon_idx] - 1
    testY_CNN = np_utils.to_categorical(testY_CNN, 3)

    deeplob = create_deeplob(100, 40, 64)

    deeplob.fit(trainX_CNN, trainY_CNN, epochs=200, batch_size=64, verbose=2, validation_data=(testX_CNN, testY_CNN))
