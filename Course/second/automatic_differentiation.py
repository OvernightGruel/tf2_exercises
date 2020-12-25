# 自动微分、
import tensorflow as tf

# x = tf.constant([[1, 2],
#                  [3, 4]], dtype=tf.float32)
#
# with tf.GradientTape(persistent=True, watch_accessed_variables=True) as grad:  # watch_accessed_variables自动监控可训练变量
#     grad.watch(x)  # GradientTape默认只监控由tf.Variable创建的traiable=True属性（默认）的变量
#     y = tf.reduce_sum(x)   # y = x1 + x2 + x3 + x4  dy/dx = 1
#     z = tf.multiply(y, y)  # z = y * Y  dz/dy = 2y
#
# dz_dy = grad.gradient(z, y)
# dz_dx = grad.gradient(z, x)  # dz/dx = dz/dy * dy/dx = 2y = 2 * (1+2+3+4)
# print(dz_dy)
# print(dz_dx)
# del grad

# 高阶导数
x = tf.Variable(1.0, trainable=True)
with tf.GradientTape() as grad:
    with tf.GradientTape() as grad2:
        y = x * x * x
    dy_dx = grad2.gradient(y, x)
dy2_dx2 = grad.gradient(dy_dx, x)

print(dy_dx)
print(dy2_dx2)

