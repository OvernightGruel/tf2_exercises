import tensorflow as tf
import numpy as np


# ----------------------- layer -------------------------------
# class SimpleLayer(tf.Module):
#
#     def __init__(self, *args, **kwargs):
#         super(SimpleLayer, self).__init__(*args, **kwargs)
#
#         self.x = tf.Variable([[1.0, 3.0]], name='x')
#         self.y = tf.Variable(2.0, trainable=False, name='y')
#
#     def __call__(self, inputs):
#         return self.x * inputs + self.y


# simple_layer = SimpleLayer(name='layer1')
# output = simple_layer(tf.constant(1.0))
# print(output)
# print(simple_layer.name)
# print(simple_layer.trainable_variables)

# ----------------------- model -------------------------------
# class Model(tf.Module):
#
#     def __init__(self, *args, **kwargs):
#         super(Model, self).__init__(*args, **kwargs)
#
#         self.layer_1 = SimpleLayer(name='layer1')
#         self.layer_2 = SimpleLayer(name='layer2')
#
#     def __call__(self, x):
#         x = self.layer_1(x)
#         output = self.layer_2(x)
#         return output
#
#
# custom_model = Model(name='model1')
# output = custom_model(tf.constant(1.0))
# print(output)
# print(custom_model.name)
# print(custom_model.trainable_variables)


# ----------------------- keras model -------------------------------
class CustomModel(tf.keras.Model):

    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer_1 = tf.keras.layers.Dense(16, activation=tf.nn.relu)
        self.layer_2 = tf.keras.layers.Dense(32, activation=None)

    def call(self, inputs, training=None, mask=None):
        x = self.layer_1(inputs)
        out = self.layer_2(x)
        return out

    def get_config(self):
        pass


custom_model = CustomModel()
output = custom_model(tf.constant([[1.0, 2.0, 3.0]]))
# print(output)
# print(output.shape)
# print(custom_model.name)
print([tf.size(var).numpy() for var in custom_model.trainable_variables])