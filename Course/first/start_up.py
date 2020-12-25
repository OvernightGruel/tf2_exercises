import tensorflow as tf

# Version
print('TF Version: {0}'.format(tf.__version__))

# cuda availibility?
print('cuda availibility: {0}'.format(tf.test.is_built_with_cuda()))

# devices
print('All devices: {0}'.format(tf.config.list_physical_devices()))
print('GPU devices: {0}'.format(tf.config.list_physical_devices(device_type='GPU')))
print('CPU devices: {0}'.format(tf.config.list_physical_devices(device_type='CPU')))

# tensor
print(tf.math.reduce_sum(tf.random.normal([2, 10])))
