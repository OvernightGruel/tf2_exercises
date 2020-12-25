import tensorflow as tf
import timeit


# -------------------- tf.function -----------------------
# @tf.function
# def matmul_fn(a, b):
#     return tf.matmul(a, b)
#
#
# def add_fn(a, b):
#     return tf.add(a, b)
#
#
# a = tf.constant([[0.5, 1.0]])
# b = tf.constant([[10.0], [1.0]])
#
# print(matmul_fn(a, b).numpy())
# print(add_fn(a, b).numpy())


# -------------------- speedup -----------------------
class ModelShallow(tf.keras.Model):

    def __init__(self):
        super(ModelShallow, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(30, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

    def get_config(self):
        pass


class ModelDeep(tf.keras.Model):

    def __init__(self):
        super(ModelDeep, self).__init__()
        self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(2000, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(3000, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(rate=0.5)

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

    def get_config(self):
        pass


sample_input = tf.random.uniform([60, 28, 28])
# shallow_model_with_eager = ModelShallow()
# shallow_model_on_graph = tf.function(ModelShallow())
deep_model_with_eager = ModelDeep()
# deep_model_on_graph = tf.function(ModelDeep())


# print('ShallowModel Eager execution time: {0}'.format(
#     timeit.timeit(lambda: shallow_model_with_eager(sample_input), number=200)))
# print('ShallowModel Graph-based execution time: {0}'.format(
#     timeit.timeit(lambda: shallow_model_on_graph(sample_input), number=200)))
print('DeepModel Eager execution time: {0}'.format(
    timeit.timeit(lambda: deep_model_with_eager(sample_input), number=200)))
# print('DeepModel Graph-based execution time: {0}'.format(
#     timeit.timeit(lambda: deep_model_on_graph(sample_input), number=200)))
