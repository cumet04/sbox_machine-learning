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


def generate_train_set(rawdata, n_prev=100):
    # rawdata as DataFrame
    result_len = len(rawdata) - n_prev
    inputs = [rawdata.iloc[i : i + n_prev].values for i in range(result_len)]
    expected = [rawdata.iloc[i + n_prev].values for i in range(result_len)]
    return np.array(inputs), np.array(expected)


def train_test_split(df, train_data_rate=0.9, n_prev=100):
    train_data_len = int(round(len(df) * train_data_rate))
    train = df.iloc[0:train_data_len]
    test = df.iloc[train_data_len:]
    return generate_train_set(train, n_prev), generate_train_set(test, n_prev)


length_of_sequences = 100
sin_values = gen_sin_values()
(train_x, train_y), (test_x, test_y) = train_test_split(
    sin_values[["value"]], n_prev=length_of_sequences
)

in_out_neurons = 1
hidden_neurons = 300

model = Sequential()
model.add(
    LSTM(
        hidden_neurons,
        batch_input_shape=(None, length_of_sequences, in_out_neurons),
        return_sequences=False,
    )
)
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(train_x, train_y, batch_size=600, nb_epoch=15, validation_split=0.05)

predicted = model.predict(test_x)
print(predicted)
