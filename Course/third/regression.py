import os
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------- dataset --------------------------------------
# dataset_path = tf.keras.utils.get_file(
#     'housing.data', 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data')

dataset_path = '/Users/clay/.keras/datasets/housing.data'

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATION', 'B', 'LSTAT', 'MEDV']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
# print(dataset.tail(10))

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# fig, ax = plt.subplots()
# x = train_dataset['RM']
# y = train_dataset['MEDV']
# ax.scatter(x, y, edgecolors=(0, 0, 0))
# ax.set_xlabel('RM')
# ax.set_ylabel('MEDV')
# plt.show()

train_input = train_dataset['RM']
train_target = train_dataset['MEDV']
test_input = test_dataset['RM']
test_target = test_dataset['MEDV']

# # ----------------------------- model --------------------------------------
model = tf.keras.Sequential([tf.keras.layers.Dense(1, use_bias=True, input_shape=(1, ), name='layer1')])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99,
                                     epsilon=1e-05, amsgrad=False, name='Adam')
model.compile(loss=tf.keras.losses.mse, optimizer=optimizer, metrics=['mae', 'mse'])
model.summary()

# early stop callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100)


# custom callback print logger
class EpochLogger(tf.keras.callbacks.Callback):

    def __init__(self, per_epoch):
        super().__init__()
        self.per_epoch = per_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.per_epoch == 0:
            print('Epoch {0}, loss {1:.2f}, val_loss {2:.2f}, mae {3:.2f}, val_mae {4:.2f}, mse {5:.2f}, '
                  'val_mse {6:.2f}'.format(epoch, logs['loss'], logs['val_loss'], logs['mae'],
                                           logs['val_mae'], logs['mse'], logs['val_mse']))


num_epoch_log = 200
epoch_logger = EpochLogger(num_epoch_log)

# TensorBoard callback
base_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

#  checkpoint callback save weights
checkpoint_dir = os.path.join(base_dir, 'training')
checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:05d}.ckpt')
model.save_weights(checkpoint_path.format(epoch=0))
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=num_epoch_log
)

# Trains the model for a fixed number of epochs (iterations on a dataset)
history = model.fit(train_input, train_target, batch_size=256, epochs=4000, verbose=0,
                    callbacks=[early_stop, epoch_logger, tensorboard_callback, checkpoint_callback],
                    validation_split=0.1)

# ----------------------------- loss --------------------------------------
# mae = np.asarray(history.history['mae'])
# val_mae = np.asarray(history.history['val_mae'])
# num_values = len(mae)
# values = np.zeros(shape=(num_values, 2), dtype=float)
# values[:, 0] = mae
# values[:, 1] = val_mae
#
# steps = pd.RangeIndex(start=0, stop=num_values)
# mae_data = pd.DataFrame(values, steps, columns=['training-mae', 'val-mae'])
#
# sns.set(style='whitegrid')
# sns.lineplot(data=mae_data, palette="tab10", linewidth=2.5)
# plt.show()

# ----------------------------- predict --------------------------------------
# predictions = model.predict(test_input).flatten()
# plt.axes(aspect='equal')
# plt.scatter(predictions, test_input, edgecolors=(0, 0, 0))
# plt.xlabel('target')
# plt.ylabel('predict')
# lims = [0, 50]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.plot(lims, lims)
# plt.show()

# ----------------------------- predict --------------------------------------
layer = model.get_layer('layer1')
w1, w0 = layer.get_weights()
w1 = float(w1[0])
w0 = float(w0[0])

fig, ax = plt.subplots()
x = test_input
y = test_target
ax.scatter(x, y, edgecolors=(0, 0, 0))
ax.set_xlabel('RM')
ax.set_ylabel('MEDV')
y_hat = x * w1 + w0
plt.plot(x, y_hat, '-r')
plt.show()

