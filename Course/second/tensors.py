# 张量
import tensorflow as tf

# tensor: Muti-dimentional多维数组
# rank: 向量维度
# tensor = tf.constant(0)
# print('constant tensor {0} of rank {1}'.format(tensor, tf.rank(tensor)))
# print('show full tensor: {0}'.format(tensor))

# transform tf.tensor to numpy
# tensor2 = tf.constant([1, 2, 3])
# print('tensor2: {0}'.format(tensor2))
# print('numpy: {0}'.format(tensor2.numpy()))  # numpy.ndarray

# tensor operations
x = tf.constant([
    [2, 3],
    [4, 5]
])

y = tf.constant([
    [2, 4],
    [6, 8]
])
print(tf.reduce_sum(x))
print(tf.add(x, y))  # 矩阵加法
print(tf.matmul(x, y))  # 矩阵乘法
print(tf.multiply(x, y))  # 矩阵点乘

# 多维tensor
# tensor3 = tf.ones(shape=[2, 3, 10], dtype=tf.float32)
# print('tensor3: {0}'.format(tensor3))
# print('tensor3 rank: {0}'.format(tf.rank(tensor3).numpy()))
# print('tensor3 shape: {0}'.format(tensor3.shape))
# print('tensor3 dimensions: {0}'.format(tensor3.ndim))
# print('tensor3 elements type: {0}'.format(tensor3.dtype))
# print('tensor3 elements number: {0}'.format(tf.size(tensor3).numpy()))

# indexing
# x = tf.constant(
#     [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]])
# print(x[0, 2].numpy())  # print(x[0][2].numpy())
# print(x[0].numpy())
# print(x[0: 2].numpy())
# print(x[0: 2, 2].numpy())

# data type
# ori_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
# print('ori_tensor elements type: {0}'.format(ori_tensor.dtype))
# casted_tensor = tf.cast(ori_tensor, dtype=tf.float32)
# print('casted_tensor elements type: {0}'.format(casted_tensor.dtype))
