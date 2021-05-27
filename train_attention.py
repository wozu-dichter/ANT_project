import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, ReLU, LeakyReLU, Input, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from scipy.io import loadmat


class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        plt.ion()
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.logs = None
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        self.logs = logs
        self.validate()
        self.plot_figures()

    def validate(self):
        # valid_x = train_x
        # valid_y = train_y
        rt_predictions = self.model.predict(valid_x)
        indexes_prediction = np.array([rt_to_index(rt) for rt in rt_predictions])
        indexes_gt = np.array([rt_to_index(rt) for rt in valid_y])
        val_accuracy = np.sum(indexes_prediction == indexes_gt) / len(indexes_prediction)
        print("accuracy : {:.3f}".format(val_accuracy))

        self.train_losses.append(self.logs["loss"])
        self.val_losses.append(self.logs["val_loss"])
        self.val_accuracies.append(val_accuracy)

    def plot_figures(self):
        plt.figure(1)
        plt.clf()
        plt.plot(self.train_losses, color="r", label="training loss")
        plt.plot(self.val_losses, color="b", label="validation loss")
        plt.xlabel("loss")
        plt.legend(loc="upper left")

        plt.figure(2)
        plt.clf()
        plt.plot(self.val_accuracies, color="b", label="validation accuracy")
        plt.xlabel("accuracy")
        plt.legend(loc="upper left")

        plt.show()
        plt.pause(0.01)


def rt_to_index(rt):
    rt = rt_preprocessing(rt)
    index = (rt - args.rt_min_threshold) / (args.rt_max_threshold - args.rt_min_threshold) * 10
    index = int(index)
    return index


def feature_preprocessing(f):
    mean = np.mean(f)
    std = np.std(f)
    return (f - mean) / std


def rt_preprocessing(rt):
    rt = max(rt, args.rt_min_threshold)
    rt = min(rt, args.rt_max_threshold)
    return rt


def apply_shuffle(x_original, y_original, seed=888):
    num_samples = x_original.shape[0]
    random.seed(seed)
    random_indexes = random.sample(range(num_samples), num_samples)
    x_shuffled = x_original[random_indexes]
    y_shuffled = y_original[random_indexes]
    return x_shuffled, y_shuffled


parser = argparse.ArgumentParser()
parser.add_argument("--rt_min_threshold", type=float, default=0.45)
parser.add_argument("--rt_max_threshold", type=float, default=1.3)
# model parameters
parser.add_argument("--dropout_rate", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=18)
parser.add_argument("--loss_function", type=str, default="mae")
parser.add_argument("--k_fold_cross_validation", type=int, default=10)
args = parser.parse_args()

mat_data = loadmat("pw_data_sub_of.mat")["pw_cell_sub"][:, 0]
data = []
label = []

for subject_data in mat_data:
    if subject_data.shape[0] != 720:
        continue
    for single_trial in subject_data:
        correct = bool(single_trial[802])
        if correct:
            feature = single_trial[:800]
            response_time = single_trial[801]
            # feature = feature_preprocessing(feature)
            response_time = rt_preprocessing(response_time)
            data.append(feature)
            label.append(response_time)

data = np.array(data)
data = feature_preprocessing(data)
label = np.array(label)
print("shape of data : {}".format(data.shape))
print("shape of label : {}".format(label.shape))

inputs = Input(shape=800)
x = Dense(800)(inputs)
x = ReLU()(x)
x = Dense(800)(x)
x = ReLU()(x)
x = Dense(400)(x)
x = Dropout(args.dropout_rate)(x)
x = Dense(200)(x)
x = Dropout(args.dropout_rate)(x)
x = Dense(200)(x)
x = LeakyReLU()(x)
x = Dense(50)(x)
x = Dense(1)(x)
model = Model(inputs=inputs, outputs=x)

# inputs = Input(shape=(800, 1))
# x = Conv1D(filters=16, kernel_size=8, activation="relu")(inputs)
# x = MaxPool1D()(x)
# x = Conv1D(filters=32, kernel_size=4, activation="relu")(x)
# x = MaxPool1D()(x)
# x = Conv1D(filters=64, kernel_size=2, activation="relu")(x)
# x = MaxPool1D()(x)
# x = Flatten()(x)
# x = Dense(1)(x)
# model = Model(inputs=inputs, outputs=x)

model.summary()
optimizer = Adam(lr=args.learning_rate)
model.compile(optimizer=optimizer, loss=args.loss_function)
model.save_weights("initial_weights.h5")

data, label = apply_shuffle(data, label)
data_num = data.shape[0]
split_indexes = np.arange(0, data_num + 1, data_num / args.k_fold_cross_validation)
for i in range(args.k_fold_cross_validation):
    valid_start = int(split_indexes[i])
    valid_end = int(split_indexes[i + 1])

    train_indexes = list(range(0, valid_start)) + list(range(valid_end, data_num))
    valid_indexes = list(range(valid_start, valid_end))

    train_x = data[train_indexes]
    train_y = label[train_indexes]
    valid_x = data[valid_indexes]
    valid_y = label[valid_indexes]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=2),
        CustomCallback(),
    ]

    model.load_weights("initial_weights.h5")
    model.fit(train_x, train_y, batch_size=args.batch_size, epochs=20000, validation_data=(valid_x, valid_y), callbacks=callbacks)
