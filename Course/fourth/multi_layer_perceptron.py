import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# im_list = [x_train[i] for i in range(4*4)]
# fig = plt.figure(figsize=(4., 4.))
# grid = ImageGrid(fig, rect=111, nrows_ncols=(4, 4), axes_pad=0.2)
# for ax, im in zip(grid, im_list):
#     ax.imshow(im, 'gray')
# plt.show()

num_classes = 10
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.sigmoid)
])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=10)

eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=1)
print(eval_loss)
print(eval_acc)
