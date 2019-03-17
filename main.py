import pandas as pd
import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

random.seed(0)


def gen_sin_values():
    steps_per_cycle = 80
    number_of_cycles = 50
    random_factor = 0.05
    sin_values = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["time"])
    sin_values["value"] = sin_values.time.apply(
        lambda x: math.sin(
            x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor
        )
    )
    return sin_values


def generate_train_set(rawdata, n_prev=100):
    result_len = len(rawdata) - n_prev
    inputs = [rawdata.iloc[i : i + n_prev].values for i in range(result_len)]
    expected = [rawdata.iloc[i + n_prev].values for i in range(result_len)]
    return np.array(inputs), np.array(expected)


sin_values = gen_sin_values()
train_x, train_y = generate_train_set(sin_values[["value"]])

hidden_neurons = 300
model = Sequential()
model.add(
    LSTM(
        hidden_neurons,
        batch_input_shape=(None, train_x.shape[1], train_x.shape[2]),
        return_sequences=False,
    )
)

model.add(Dense(train_x.shape[2]))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(train_x, train_y, batch_size=100, nb_epoch=10, validation_split=0.05)

# predicted = model.predict(test_x)
# print(predicted)
