import tensorflow as tf
import numpy as np 

# All building functions support NCHW (for many tf functions only works on GPU's) 
# and NHWC (also for CPU's)
#----------------------------------------------------------------------------
# Get Weights.

def get_weight(shape):
    return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(0, 0.02))

#----------------------------------------------------------------------------
# Convolutional layer.

def conv2d(x, fmaps, kernel, cf = True):
    assert kernel >= 1 and kernel % 2 == 1
    if cf: shape_ = x.shape[1]
    else: shape_ = x.shape[3]
    w = get_weight([kernel, kernel, shape_.value, fmaps])
    w = tf.cast(w, x.dtype)
    if cf: return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
    else: return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NHWC')

#----------------------------------------------------------------------------
# Dense layer.

def dense(x, cf = True):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    #if cf: shape_ = x.shape[1]
    #else: shape_  = x.shape[3]
    w = get_weight([x.shape[1].value, 1])
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Apply biases to given activation tensor.

def apply_bias(x, cf = True):
    if cf: 
        shape_ = x.shape[1]
        reshape_ = [1,-1,1,1]
    else: 
        shape_ = x.shape[3]
        reshape_ = [1,1,1,-1]

    b = tf.get_variable('bias', shape=[shape_], initializer=tf.zeros_initializer())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, shape=reshape_)


#----------------------------------------------------------------------------
# Apply biases to final layer.

def apply_dense_bias(x):
    b = tf.get_variable('bias', shape=1, initializer=tf.zeros_initializer())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b

#----------------------------------------------------------------------------
# Leaky ReLU activation. Same as tf.nn.leaky_relu, but supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)

#----------------------------------------------------------------------------
# Nearest-neighbor upscaling layer.

def upscale2d(x, cf = True,  factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        if cf:
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
            return x
        else: 
            size_ = [factor*int(x.shape[1]),factor*int(x.shape[2])]
            return tf.image.resize_nearest_neighbor(x, size=size_)

#----------------------------------------------------------------------------
# Average-pooling downcaling layer.

def downscale2d(x, cf = True, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        if cf:
            ksize = [1, 1, factor, factor]
            return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')

        else:
            ksize = [1, factor, factor, 1]
            return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NHWC')

#----------------------------------------------------------------------------
# Batchnormalization along feature channels.

def batchnorm(x, cf = True):
    with tf.variable_scope('BatchNorm'):
        if cf: axis_ = 1
        else: axis_ = 3
        return tf.layers.batch_normalization(x, axis=axis_, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

#----------------------------------------------------------------------------
# Interpolation Clipping.

def lerp_clip(a, b, t): 
    return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

