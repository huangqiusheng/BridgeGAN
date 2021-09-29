from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflib as tl
import sys

conv = partial(slim.conv2d, activation_fn=None)
dconv = partial(slim.conv2d_transpose, activation_fn=None)
fc = partial(tl.flatten_fully_connected, activation_fn=None)
relu = tf.nn.relu
lrelu = tf.nn.leaky_relu
sigmoid = tf.nn.sigmoid
tanh = tf.nn.tanh
batch_norm = partial(slim.batch_norm, scale=True, updates_collections=None)
instance_norm = slim.instance_norm

MAX_DIM = 64 * 16

def inner_product(x, y, axis = -1):
    return tf.reduce_sum(x * y, axis = axis)

def _concat(z, z_, _a):
    feats = [z]
    if z_ is not None:
        feats.append(z_)
    if _a is not None:
        if len(tl.shape(_a)) == 2:
            _a = tf.reshape(_a, [-1, 1, 1, tl.shape(_a)[-1]])
            _a = tf.tile(_a, [1, tl.shape(z)[1], tl.shape(z)[2], 1])
        else:
            _a = tf.image.resize_nearest_neighbor(_a, [tl.shape(z)[1], tl.shape(z)[2]])
        feats.append(_a)
    return tf.concat(feats, axis=3)

def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(a, range(a.shape.ndims))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
def gradient_penalty_L(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(a, range(a.shape.ndims))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty_L'):
        x = _interpolate(real, fake)
        pred = f(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
def gradient_penalty_Xreal(f, real, labels, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            if b is None:   # interpolation in DRAGAN
                beta = tf.random_uniform(shape=tf.shape(a), minval=0., maxval=1.)
                _, variance = tf.nn.moments(a, list(range(a.shape.ndims)))
                b = a + 0.5 * tf.sqrt(variance) * beta
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x, l=labels)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = tf.norm(slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        if grad_and_vars[0][0] is None:
            continue
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

#######################################################

#           old architecture

#######################################################

def Genc(x, dim=64, n_layers=5, multi_inputs=1, is_training=True, norm = 'in'):
    if norm == 'bn':
        norm_fn = partial(batch_norm, is_training=is_training)
    elif norm == 'in':
        norm_fn = instance_norm
    else:
        norm_fn = None
    conv_norm_lrelu = partial(conv, normalizer_fn=norm_fn, activation_fn=lrelu)

    with tf.variable_scope('Genc_c', reuse=tf.AUTO_REUSE):
        h, w = x.shape[1:3]
        z = x
        zs = []
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            if multi_inputs > i and i > 0:
                z = tf.concat([z, tf.image.resize_bicubic(x, (h//(2**i), w//(2**i)))], 3)
            z = conv_norm_lrelu(z, d, 4, 2)
            zs.append(z)
        return zs[:-1], zs[-1]
def Gdec(zs, _a, res, zc, zdec_src = None, zdec_tag = None, dim=64, n_layers=5, inject_layers=0, is_training=True, one_more_conv=0, norm = 'in'):
    if norm == 'bn':
        norm_fn = partial(batch_norm, is_training=is_training)
    elif norm == 'in':
        norm_fn = instance_norm
    else:
        norm_fn = None
    dconv_norm_relu = partial(dconv, normalizer_fn=norm_fn, activation_fn=relu)

    inject_layers = min(inject_layers, n_layers - 1)

    zdec = []
    with tf.variable_scope('Gdec', reuse=tf.AUTO_REUSE):
        z = FM(zc, zs, zc, zs, _a)
        z = _concat(z, None, _a)
        for i in range(n_layers):
            if i < n_layers - 1:
                d = min(dim * 2 ** (n_layers - 2 - i), MAX_DIM)
                z = dconv_norm_relu(z, d, 4, 2)
                zdec.append(z)
                if i < 2:
                    if not zdec_src and not zdec_tag:
                        z = FM(res[n_layers - 2 - i], z, z, z, _a)
                    else:
                        z = FM(res[n_layers - 2 - i], z, zdec_src[i], zdec_tag[i], _a)
                if inject_layers > i:
                    z = _concat(z, None, _a)
            else:
                if one_more_conv: # add one more conv after the decoder
                    z = dconv_norm_relu(z, dim//4, 4, 2)
                    x = tf.nn.tanh(dconv(z, 3, one_more_conv))
                else:
                    x = tf.nn.tanh(dconv(z, 3, 4, 2))
        return x, zdec

def FM(encf, decf, decf_src, decf_tag, attdiff, norm = 'in'):
    def SNLM(decf_src, decf_tag, spatialm):
        z_src = conv(decf_src, 1, 4,1)
        z_src = tf.reshape(z_src, (tl.shape(z_src)[0], tl.shape(z_src)[1]*tl.shape(z_src)[2], tl.shape(z_src)[3]))
        z_tag = conv(decf_tag, 1,4,1)
        z_tag = tf.reshape(z_tag, (tl.shape(z_tag)[0], tl.shape(z_tag)[1]*tl.shape(z_tag)[2], tl.shape(z_tag)[3]))
        z_tag = tf.transpose(z_tag, (0,2,1))
        covar = tf.nn.softmax(tf.matmul(z_src, z_tag))
        spatialm = tf.expand_dims(tf.layers.flatten(spatialm), axis=-1)
        spatialm = tf.matmul(covar, spatialm)
        spatialm = tf.reshape(spatialm, (tl.shape(decf_src)[0], tl.shape(decf_src)[1], tl.shape(decf_src)[2], 1))
        return  spatialm

    def CNLM(decf_src, decf_tag, channelm):
        z_src = tf.layers.average_pooling2d(decf_src, tl.shape(decf_src)[1], 1)
        z_src = tf.layers.flatten(z_src)
        z_src = fc(z_src, tl.shape(z_src)[-1])
        z_src = tf.expand_dims(z_src, axis=-1)

        z_tag = tf.layers.average_pooling2d(decf_tag, tl.shape(decf_tag)[1],1)
        z_tag = tf.layers.flatten(z_tag)
        z_tag = fc(z_tag, tl.shape(z_tag)[-1])
        z_tag = tf.expand_dims(z_tag, 1)

        covar = tf.nn.softmax(tf.matmul(z_src, z_tag))
        channelm = tf.squeeze(tf.matmul(covar, tf.expand_dims(channelm, axis=-1)))
        return channelm

    if norm == 'in':
        norm_fn = instance_norm
    else:
        norm_fn = None
    conv_norm_lrelu = partial(conv, normalizer_fn=norm_fn, activation_fn=lrelu)
    encz = conv_norm_lrelu(_concat(encf, None, attdiff), tl.shape(encf)[-1], 4, 1)
    enc_spatialm = conv(encz, 1, 4, 1)
    enc_channelm = tf.layers.average_pooling2d(encz, tl.shape(encz)[1], 1)
    enc_channelm = tf.layers.flatten(enc_channelm)
    enc_channelm = fc(enc_channelm, tl.shape(enc_channelm)[-1])

    decz = conv_norm_lrelu(_concat(decf, None, attdiff), tl.shape(decf)[-1], 4, 1)
    dec_spatialm = conv(decz, 1, 4, 1)
    dec_spatialm = SNLM(decf_src, decf_tag, dec_spatialm)

    dec_channelm = tf.layers.average_pooling2d(decz, tl.shape(decz)[1],1)
    dec_channelm = tf.layers.flatten(dec_channelm)
    dec_channelm = fc(dec_channelm, tl.shape(dec_channelm)[-1])
    dec_channelm = CNLM(decf_src, decf_tag, dec_channelm)

    spatialm = tf.nn.softmax(tf.concat([enc_spatialm, dec_spatialm], axis = -1), axis = -1)
    fus_spatial = tf.expand_dims(spatialm[:, :, :, 0], axis=-1) * encf + tf.expand_dims(spatialm[:, :, :, 1], axis=-1) * decf
    channelm = tf.nn.softmax(tf.stack([enc_channelm, dec_channelm], axis = 2), axis = -1)
    fus_channel = tf.expand_dims(tf.expand_dims(channelm[:, :, 0], axis=1),axis=1) * encf + \
                    tf.expand_dims(tf.expand_dims(channelm[:, :, 1], axis=1),axis=1) * decf
    fus = conv(tf.concat([fus_spatial, fus_channel], axis = -1), tl.shape(encf)[-1], 1, 1)
    return fus


def D(x, n_att, dim=64, fc_dim=MAX_DIM, n_layers=5):
    conv_in_lrelu = partial(conv, normalizer_fn=instance_norm, activation_fn=lrelu)

    with tf.variable_scope('D', reuse=tf.AUTO_REUSE):
        y = x
        for i in range(n_layers):
            d = min(dim * 2**i, MAX_DIM)
            y = conv_in_lrelu(y, d, 4, 2)

        logit_gan = lrelu(fc(y, fc_dim))
        logit_gan = fc(logit_gan, 1)

        logit_att = lrelu(fc(y, fc_dim))
        logit_att = fc(logit_att, n_att)

        return logit_gan, logit_att

#######################################################

#           arch1 (generator, mapping network and discriminator are same as StarGAN v2)

#######################################################
# def ResBlk(x_init, out_c, upsampling = False, downsampling = False, activation = relu, norm = 'in', s_rand = None, s_targ = None, norm_name = None):
#     def _upsample(var):
#         height, width = tl.shape(var)[1:3]
#         return tf.image.resize_images(var, (height * 2, width * 2),
#                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#
#     def _downsample(var):
#         return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')
#
#     def adaptive_instance_norm(var, name, dlatent):
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             var = instance_norm(var, center = False, scale = False)
#             style = fc(dlatent, tl.shape(var)[-1] * 2)
#             style = tf.reshape(style, [-1, 2] + [1] * (len(tl.shape(var)) - 2) + [tl.shape(var)[-1]])
#             return var * (style[:, 0] + 1) + style[:, 1]
#
#     def spatially_adaptive_norm(var, name, dlatent):
#         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#             var = instance_norm(var, center=False, scale=False)
#             height, width, channels = tl.shape(var)[1:]
#             style = tf.image.resize_images(dlatent, (height, width),
#                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#             style = relu(conv(style, 128, 3, 1))
#             gamma = conv(style, channels, 3, 1)
#             beta = conv(style, channels, 3, 1)
#             return var * (1+gamma) + beta
#
#     if norm == 'adain' and s_rand is not None:
#         norm_fn = partial(adaptive_instance_norm, dlatent = s_rand)
#     elif norm == 'in':
#         norm_fn = instance_norm
#     elif norm == 'spade' and s_targ is not None:
#         norm_fn = partial(spatially_adaptive_norm, dlatent = s_targ)
#     else:
#         norm_fn = None
#     in_c = tl.shape(x_init)[-1]
#     is_shortcut_learn = (in_c != out_c) or upsampling or downsampling
#     if norm == 'in':
#         x = norm_fn(x_init)
#     elif norm == 'adain' or norm == 'spade':
#         x = norm_fn(x_init, norm_name + '_1')
#     else:
#         x = x_init
#     x = activation(x)
#     if upsampling:
#         x = _upsample(x)
#     x = conv(x, out_c, 3, 1)
#     if norm == 'in':
#         x = norm_fn(x)
#     elif norm == 'adain' or norm == 'spade':
#         x = norm_fn(x, norm_name + '_2')
#     x = conv(activation(x), out_c, 3, 1)
#     if downsampling:
#         x = _downsample(x)
#     if is_shortcut_learn:
#         if upsampling:
#             x_shortcut = conv(_upsample(x_init), out_c, 1, 1)
#         elif downsampling:
#             x_shortcut = _downsample(conv(x_init, out_c, 1, 1))
#         else:
#             x_shortcut = conv(x_init, out_c, 1, 1)
#     else:
#         x_shortcut = x_init
#     return x + x_shortcut
def conv_adain(x, out_c, kernel_size, stride, s):
    def adaptive_instance_norm(var, dlatent):
        var = instance_norm(var, center = False, scale = False)
        style = fc(dlatent, tl.shape(var)[-1] * 2)
        style = tf.reshape(style, [-1, 2] + [1] * (len(tl.shape(var)) - 2) + [tl.shape(var)[-1]])
        return var * (style[:, 0] + 1) + style[:, 1]
    x = conv(x, out_c, kernel_size, stride)
    x = adaptive_instance_norm(x, s)
    return x

def dconv_adain(x, out_c, kernel_size, stride, s):
    def adaptive_instance_norm(var, dlatent):
        var = instance_norm(var, center = False, scale = False)
        style = fc(dlatent, tl.shape(var)[-1] * 2)
        style = tf.reshape(style, [-1, 2] + [1] * (len(tl.shape(var)) - 2) + [tl.shape(var)[-1]])
        return var * (style[:, 0] + 1) + style[:, 1]
    x = dconv(x, out_c, kernel_size, stride)
    x = adaptive_instance_norm(x, s)
    return x

def ResBlk(x_init, out_c, upsampling = False, downsampling = False, activation = relu, norm = 'in', s_rand = None, s_targ = None, alpha = None, norm_name = None):
    def _upsample(var):
        height, width = tl.shape(var)[1:3]
        return tf.image.resize_images(var, (height * 2, width * 2),
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _downsample(var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')

    def adaptive_instance_norm(var, name, dlatent):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = instance_norm(var, center = False, scale = False)
            style = fc(dlatent, tl.shape(var)[-1] * 2)
            style = tf.reshape(style, [-1, 2] + [1] * (len(tl.shape(var)) - 2) + [tl.shape(var)[-1]])
            return var * (style[:, 0] + 1) + style[:, 1]

    def spatially_adaptive_norm(var, name, dlatent):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = instance_norm(var, center=False, scale=False)
            height, width, channels = tl.shape(var)[1:]
            style = tf.image.resize_images(dlatent, (height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            style = relu(conv(style, 128, 3, 1))
            gamma = conv(style, channels, 3, 1)
            beta = conv(style, channels, 3, 1)
            return var * (1+gamma) + beta

    if norm == 'adain' and s_rand is not None:
        norm_fn = partial(adaptive_instance_norm, dlatent = s_rand)
    elif norm == 'in':
        norm_fn = instance_norm
    elif norm == 'spade' and s_targ is not None:
        norm_fn = partial(spatially_adaptive_norm, dlatent = s_targ)
    else:
        norm_fn = None
    in_c = tl.shape(x_init)[-1]
    is_shortcut_learn = (in_c != out_c) or upsampling or downsampling
    if norm == 'in':
        x = norm_fn(x_init)
    elif norm == 'adain' or norm == 'spade':
        x = norm_fn(x_init, norm_name + '_1')
    elif norm == 'interp':
        x = alpha * spatially_adaptive_norm(x_init, norm_name[0] + '_1', dlatent = s_rand) + \
            (1-alpha) * spatially_adaptive_norm(x_init, norm_name[1] + '_1', dlatent = s_targ)
    else:
        x = x_init
    x = activation(x)
    if upsampling:
        x = _upsample(x)
    x = conv(x, out_c, 3, 1)
    if norm == 'in':
        x = norm_fn(x)
    elif norm == 'adain' or norm == 'spade':
        x = norm_fn(x, norm_name + '_2')
    elif norm == 'interp':
        x = alpha * spatially_adaptive_norm(x, norm_name[0] + '_2', dlatent = s_rand) + \
            (1-alpha) * spatially_adaptive_norm(x, norm_name[1] + '_2', dlatent = s_targ)

    x = conv(activation(x), out_c, 3, 1)
    if downsampling:
        x = _downsample(x)
    if is_shortcut_learn:
        if upsampling:
            x_shortcut = conv(_upsample(x_init), out_c, 1, 1)
        elif downsampling:
            x_shortcut = _downsample(conv(x_init, out_c, 1, 1))
        else:
            x_shortcut = conv(x_init, out_c, 1, 1)
    else:
        x_shortcut = x_init
    return x + x_shortcut
def ResBlk_interp(x_init, out_c, upsampling = False, downsampling = False, activation = relu, norm = 'in', s_rand = None, s_targ = None, s_targ_t = None, alpha = None, norm_name = None):
    def _upsample(var):
        height, width = tl.shape(var)[1:3]
        return tf.image.resize_images(var, (height * 2, width * 2),
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def _downsample(var):
        return tf.layers.average_pooling2d(var, (2, 2), (2, 2), padding='SAME')

    def adaptive_instance_norm(var, name, dlatent):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = instance_norm(var, center = False, scale = False)
            style = fc(dlatent, tl.shape(var)[-1] * 2)
            style = tf.reshape(style, [-1, 2] + [1] * (len(tl.shape(var)) - 2) + [tl.shape(var)[-1]])
            return var * (style[:, 0] + 1) + style[:, 1]

    def spatially_adaptive_norm(var, name, dlatent):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            var = instance_norm(var, center=False, scale=False)
            height, width, channels = tl.shape(var)[1:]
            style = tf.image.resize_images(dlatent, (height, width),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            style = relu(conv(style, 128, 3, 1))
            gamma = conv(style, channels, 3, 1)
            beta = conv(style, channels, 3, 1) #var * (1+gamma) + beta
            return var * (1+gamma), beta

    if norm == 'adain' and s_rand is not None:
        norm_fn = partial(adaptive_instance_norm, dlatent = s_rand)
    elif norm == 'in':
        norm_fn = instance_norm
    elif norm == 'spade' and s_targ is not None:
        norm_fn = partial(spatially_adaptive_norm, dlatent = s_targ)#, dlatent_t=s_targ_t
        # norm_fn_2 = partial(spatially_adaptive_norm, dlatent=s_targ_t)
    else:
        norm_fn = None
    in_c = tl.shape(x_init)[-1]
    is_shortcut_learn = (in_c != out_c) or upsampling or downsampling
    if norm == 'in':
        x = norm_fn(x_init)
    elif norm == 'adain' or norm == 'spade':
        v, _ = norm_fn(x_init, norm_name + '_1', dlatent=s_targ_t)
        _, b = norm_fn(x_init, norm_name + '_1', dlatent=s_targ)
        x = v + b
    elif norm == 'interp':
        x = alpha * spatially_adaptive_norm(x_init, norm_name[0] + '_1', dlatent = s_rand) + \
            (1-alpha) * spatially_adaptive_norm(x_init, norm_name[1] + '_1', dlatent = s_targ)
    else:
        x = x_init
    x = activation(x)
    if upsampling:
        x = _upsample(x)
    x = conv(x, out_c, 3, 1)
    if norm == 'in':
        x = norm_fn(x)
    elif norm == 'adain' or norm == 'spade':
        # x = norm_fn(x, norm_name + '_2')
        v, _ = norm_fn(x, norm_name + '_2', dlatent=s_targ_t)
        _, b = norm_fn(x, norm_name + '_2', dlatent=s_targ)
        x = v + b
    elif norm == 'interp':
        x = alpha * spatially_adaptive_norm(x, norm_name[0] + '_2', dlatent = s_rand) + \
            (1-alpha) * spatially_adaptive_norm(x, norm_name[1] + '_2', dlatent = s_targ)

    x = conv(activation(x), out_c, 3, 1)
    if downsampling:
        x = _downsample(x)
    if is_shortcut_learn:
        if upsampling:
            x_shortcut = conv(_upsample(x_init), out_c, 1, 1)
        elif downsampling:
            x_shortcut = _downsample(conv(x_init, out_c, 1, 1))
        else:
            x_shortcut = conv(x_init, out_c, 1, 1)
    else:
        x_shortcut = x_init
    return x + x_shortcut

def Generator(x, s_rand = None, s_targ = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            if i < n_intermedblks // 2:
                z = ResBlk(z, ch, norm='in')
            else:
                if s_rand is not None:
                    z = ResBlk(z, ch, norm='adain', s_rand = s_rand, norm_name='adain%d_inter'%i)
                elif s_targ is not None:
                    z = ResBlk(z, ch, norm='spade', s_targ=s_targ, norm_name='spade%d_inter'%i)
                else:
                    print('Please input s_rand or s_targ\n')
                    sys.exit()
        for i in range(n_upblks):
            ch /= 2
            if s_rand is not None:
                z = ResBlk(z, ch, upsampling=True, norm='adain', s_rand = s_rand, norm_name='adain%d_up'%i)
            elif s_targ is not None:
                z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ=s_targ, norm_name='spade%d_up'%i)
            else:
                print('Please input s_rand or s_targ\n')
                sys.exit()
        z = tanh(conv(z, 3, 1, 1))
        return z

# def Generator2(x, s_rand = None, s_targ = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
#     with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
#         z = conv(x, ch, 1, 1)
#         for i in range(n_downblks):
#             ch *= 2
#             z = ResBlk(z, ch, downsampling=True, norm = 'in')
#         for i in range(n_intermedblks):
#             if i < n_intermedblks // 2:
#                 z = ResBlk(z, ch, norm='in')
#             else:
#                 if s_rand is not None:
#                     z = ResBlk(z, ch, norm='spade', s_targ = s_rand, norm_name='rand%d_inter'%i)
#                 elif s_targ is not None:
#                     z = ResBlk(z, ch, norm='spade', s_targ=s_targ, norm_name='targ%d_inter'%i)
#                 else:
#                     print('Please input s_rand or s_targ\n')
#                     sys.exit()
#         for i in range(n_upblks):
#             ch /= 2
#             if s_rand is not None:
#                 z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ = s_rand, norm_name='rand%d_up'%i)
#             elif s_targ is not None:
#                 z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ=s_targ, norm_name='targ%d_up'%i)
#             else:
#                 print('Please input s_rand or s_targ\n')
#                 sys.exit()
#         z = tanh(conv(z, 3, 1, 1))
#         return z

def Generator2(x, s_rand = None, s_targ = None, alpha = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            if i < n_intermedblks // 2:
                z = ResBlk(z, ch, norm='in')
            else:
                if s_rand is not None and s_targ is None:
                    z = ResBlk(z, ch, norm='spade', s_targ = s_rand, norm_name='rand%d_inter'%i)
                elif s_targ is not None and s_rand is None:
                    if isinstance (s_targ,list):
                        z = ResBlk(z, ch, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_inter' % i, 'targ%d_inter' % i])
                    else:
                        z = ResBlk(z, ch, norm='spade', s_targ=s_targ, norm_name='targ%d_inter'%i)
                elif s_rand is not None and s_targ is not None:
                    z = ResBlk(z, ch, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_inter'%i, 'targ%d_inter'%i])
                else:
                    print('Please input s_rand or s_targ\n')
                    sys.exit()
        for i in range(n_upblks):
            ch /= 2
            if s_rand is not None and s_targ is None:
                z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ = s_rand, norm_name='rand%d_up'%i)
            elif s_targ is not None and s_rand is None:
                if isinstance(s_targ, list):
                    z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_up' % i, 'targ%d_up' % i])
                else:
                    z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ=s_targ, norm_name='targ%d_up'%i)
            elif s_rand is not None and s_targ is not None:
                z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_up'%i, 'targ%d_up'%i])
            else:
                print('Please input s_rand or s_targ\n')
                sys.exit()
        z = tanh(conv(z, 3, 1, 1))
        return z
def Generator2_wRelu(x, s_rand = None, s_targ = None, s_targ_t=None, alpha = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            if i < n_intermedblks // 2:
                z = ResBlk(z, ch, norm='in')
            else:
                if s_rand is not None and s_targ is None:
                    z = ResBlk(z, ch, norm='spade', s_targ = s_rand, norm_name='rand%d_inter'%i)
                elif s_targ is not None and s_rand is None:
                    if isinstance (s_targ, list):
                        z = ResBlk(z, ch, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_inter' % i, 'targ%d_inter' % i])
                    else:
                        z = ResBlk(z, ch, norm='spade', s_targ=s_targ, norm_name='targ%d_inter'%i)
                        # z = ResBlk_interp(z, ch, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_inter'%i)
                elif s_rand is not None and s_targ is not None:
                    z = ResBlk(z, ch, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_inter'%i, 'targ%d_inter'%i])
                else:
                    print('Please input s_rand or s_targ\n')
                    sys.exit()
        for i in range(n_upblks):
            ch /= 2
            if s_rand is not None and s_targ is None:
                z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ = s_rand, norm_name='rand%d_up'%i)
            elif s_targ is not None and s_rand is None:
                if isinstance(s_targ, list):
                    z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_up' % i, 'targ%d_up' % i])
                else:
                    z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ=s_targ, norm_name='targ%d_up'%i)
                    # z = ResBlk_interp(z, ch, upsampling=True, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_up' % i)
            elif s_rand is not None and s_targ is not None:
                z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_up'%i, 'targ%d_up'%i])
            else:
                print('Please input s_rand or s_targ\n')
                sys.exit()
        z = relu(z)
        z = tanh(conv(z, 3, 1, 1))
        return z
def Generator2_wRelu_imation(x, s_rand = None, s_targ = None, s_targ_t=None, alpha = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            if i < n_intermedblks // 2:
                z = ResBlk(z, ch, norm='in')
            else:
                if s_rand is not None and s_targ is None:
                    z = ResBlk(z, ch, norm='spade', s_targ = s_rand, norm_name='rand%d_inter'%i)
                elif s_targ is not None and s_rand is None:
                    if isinstance (s_targ, list):
                        z = ResBlk(z, ch, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_inter' % i, 'targ%d_inter' % i])
                    else:
                        z = ResBlk(z, ch, norm='spade', s_targ=s_targ, norm_name='targ%d_inter'%i)
                        # z = ResBlk_interp(z, ch, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_inter'%i)
                elif s_rand is not None and s_targ is not None:
                    z = ResBlk(z, ch, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_inter'%i, 'targ%d_inter'%i])
                else:
                    print('Please input s_rand or s_targ\n')
                    sys.exit()
        for i in range(n_upblks):
            ch /= 2
            if s_rand is not None and s_targ is None:
                z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ = s_rand, norm_name='rand%d_up'%i)
            elif s_targ is not None and s_rand is None:
                if isinstance(s_targ, list):
                    z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_up' % i, 'targ%d_up' % i])
                else:
                    z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ=s_targ, norm_name='targ%d_up'%i)
                    # z = ResBlk_interp(z, ch, upsampling=True, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_up' % i)
            elif s_rand is not None and s_targ is not None:
                z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_up'%i, 'targ%d_up'%i])
            else:
                print('Please input s_rand or s_targ\n')
                sys.exit()
        z = relu(z)
        mask_img = tf.nn.sigmoid(conv(z, 1, 1, 1))
        fake_img = tanh(conv(z, 3, 1, 1))
        fake_img = mask_img * x + (1 - mask_img) * fake_img
        return fake_img, mask_img
def Generator2_wRelu_adain(x, s_rand = None, s_targ = None, s_targ_t=None, alpha = None, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            if i < n_intermedblks // 2:
                z = ResBlk(z, ch, norm='in')
            else:
                if s_rand is not None and s_targ is None:
                    z = ResBlk(z, ch, norm='spade', s_targ = s_rand, norm_name='rand%d_inter'%i)
                elif s_targ is not None and s_rand is None:
                    if isinstance (s_targ,list):
                        z = ResBlk(z, ch, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_inter' % i, 'targ%d_inter' % i])
                    else:
                        z = ResBlk(z, ch, norm='adain', s_rand=s_targ, norm_name='rand%d_inter'%i)
                        # z = ResBlk_interp(z, ch, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_inter'%i)
                elif s_rand is not None and s_targ is not None:
                    z = ResBlk(z, ch, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_inter'%i, 'targ%d_inter'%i])
                else:
                    print('Please input s_rand or s_targ\n')
                    sys.exit()
        for i in range(n_upblks):
            ch /= 2
            if s_rand is not None and s_targ is None:
                z = ResBlk(z, ch, upsampling=True, norm='spade', s_targ = s_rand, norm_name='rand%d_up'%i)
            elif s_targ is not None and s_rand is None:
                if isinstance(s_targ, list):
                    z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand=s_targ[0], s_targ=s_targ[1], alpha=alpha, norm_name=['targ%d_up' % i, 'targ%d_up' % i])
                else:
                    z = ResBlk(z, ch, upsampling=True, norm='adain', s_rand=s_targ, norm_name='rand%d_up'%i)
                    # z = ResBlk_interp(z, ch, upsampling=True, norm='spade', s_targ=s_targ, s_targ_t=s_targ_t, norm_name='targ%d_up' % i)
            elif s_rand is not None and s_targ is not None:
                z = ResBlk(z, ch, upsampling=True, norm='interp', s_rand = s_rand, s_targ=s_targ, alpha = alpha, norm_name = ['rand%d_up'%i, 'targ%d_up'%i])
            else:
                print('Please input s_rand or s_targ\n')
                sys.exit()
        z = relu(z)
        z = tanh(conv(z, 3, 1, 1))
        return z

def Generator_stargan(x,  c, n_downblks = 4, n_intermedblks = 4, n_upblks = 4, ch = 32):
    with tf.variable_scope('Ggenerator', reuse=tf.AUTO_REUSE):
        c = tf.reshape(c, [-1, 1, 1, tl.shape(c)[-1]])
        c = tf.tile(c, [1, tl.shape(x)[1], tl.shape(x)[2], 1])
        z = tf.concat([x, c], axis=-1)
        z = conv(z, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, norm='in')

        for i in range(n_upblks):
            ch /= 2
            z = ResBlk(z, ch, upsampling=True, norm='in')
        z = relu(z)
        z = tanh(conv(z, 3, 1, 1))
        return z

def MappingNet(r, attdiff, n_layers = 6, fc_dim = 64):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        z = tf.concat([r, attdiff], axis = -1)
        for i in range(n_layers):
            z = relu(fc(z, fc_dim))
        z = fc(z, 64)
        return z

def MappingNet_deconv(r, attdiff, n_layers = 6, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        z = tf.concat([r, attdiff], axis = -1)
        z = fc(z, 2 * 2 * ch)
        z = tf.reshape(z, (-1, 2, 2, ch))
        for i in range(n_layers):
            ch *= 2
            z = ResBlk(z, ch, upsampling=True, activation=relu, norm='in')
        z = relu(z)
        z = conv(z, ch, 1, 1)
        return z

def MappingNet_deconv2(r, attdiff, n_mlp, fc_dim, n_layers = 6, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            n_dim_per_stream = tl.shape(r)[-1] // n_stream
            z_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                ri = r[:, s*n_dim_per_stream:(s+1)*n_dim_per_stream]
                z = tf.concat([a, ri], axis=-1)
                for i in range(n_mlp):
                    z = relu(fc(z, fc_dim))
                z_s.append(z)
            z = tf.add_n(z_s)

        z = fc(z, 2 * 2 * ch)
        z = tf.reshape(z, (-1, 2, 2, ch))
        for i in range(n_layers):
            ch *= 2
            z = ResBlk(z, ch, upsampling=True, activation=relu, norm='in')
        z = relu(z)
        z = conv(z, ch, 1, 1)
        return z

def MappingNet_multiStream(r, attdiff, n_layers = 6, fc_dim = 64):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        n_stream = tl.shape(attdiff)[-1]
        n_r = tl.shape(r)[-1] // n_stream
        zs = []
        for s in range(n_stream):
            attdiffs = attdiff[:, s:s+1]
            rs = r[:, s * n_r: (s+1)*n_r]
            z = tf.concat([rs, attdiffs], axis = -1)
            for i in range(n_layers):
                z = relu(fc(z, fc_dim))
            z = fc(z, 64)
            zs.append(z)
        return tf.add_n(zs)

def MappingNet_multiStream_deconv(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = fc(r, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        z = conv(z, ch, 1, 1)
        return z

def MappingNet_multiStream_deconv_wIN_wRelu(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = fc(r, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        z = relu(instance_norm(conv(z, ch, 1, 1)))
        return z

def MappingNet_multiStream_deconv_wIN_wRelu_concat(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        z = tf.concat([r, attdiff], axis = -1)
        z = fc(z, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'in')
        z = relu(z)
        z = relu(instance_norm(conv(z, ch, 1, 1))) #
        return z
def MappingNet_multiStream_deconv_wIN_concat(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        z = tf.concat([r, attdiff], axis = -1)
        z = fc(z, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'in')
        if n_layers == 3:
            z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z)
        z = instance_norm(conv(z, ch, 1, 1)) # relu
        return z
def MappingNet_multiStream_deconv_wIN_concat_C(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        z = tf.concat([r, attdiff], axis = -1)
        z = fc(z, 4*4*ch*4*4)
        z = tf.reshape(z, (-1, 4, 4, ch*4*4)) #256*4
        ch = ch*4*4
        for i in range(n_layers):
            ch = ch//2
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'in')
        # z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z)
        z = instance_norm(conv(z, ch, 1, 1)) # relu
        return z
def MappingNet_multiStream_deconv_shareLastConv(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = fc(r, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)

    with tf.variable_scope('GshareConv', reuse=tf.AUTO_REUSE):
        z = relu(instance_norm(conv(z, ch, 1, 1)))
    return z

def MappingNet_multiStream_deconv_wIN(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = fc(r, 4*4*ch*2)
        z = tf.reshape(z, (-1, 4, 4, ch*2))
        for i in range(n_layers):
            z = ResBlk(z, ch, upsampling = True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        z = instance_norm(conv(z, ch, 1, 1))
        return z

def MappingNet_multiStream_deconv1_wIN(r, attdiff, n_mlp = 3, n_layers = 4, fc_dim = 64, ch = 32):
    with tf.variable_scope('Gmappingnet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = relu(fc(r, 4*4*ch))
        z = tf.reshape(z, (-1, 4, 4, ch))
        for i in range(n_layers):
            ch *= 2
            z = relu(dconv_adain(z, ch, 3, 2, s = a))
        z = instance_norm(conv(z, ch, 1, 1))
        return z

def Encoder(x, attdiff_abs, n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope('Gencoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            a = attdiff_abs
            for i in range(n_mlp):
                a = relu(fc(a, 4*ch))

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        z = lrelu(z)
        z = conv(z, ch, 1, 1)
        return z

def Encoder2(x, attdiff_abs, n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: masked s
    '''
    with tf.variable_scope('Gencoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            a = attdiff_abs
            for i in range(n_mlp):
                a = relu(fc(a, 4*ch))

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        ch *= 2
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)

        s = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adains')
        s = relu(s)
        s = conv(s, ch, 1, 1)

        mask = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adainm')
        mask = relu(mask)
        mask = conv(mask, 1, 1, 1) #sigmoid(conv(mask, 1, 1, 1))
        return mask * s

def Encoder3(x, attdiff_abs, n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: masked s
    '''
    with tf.variable_scope('Gencoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            a = attdiff_abs
            for i in range(n_mlp):
                a = relu(fc(a, 4*ch))

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        ch_ori = ch
        #### style code ####
        s = z
        for i in range(n_intermedblks):
            ch *= 2
            s = ResBlk(s, ch, downsampling=True, activation=relu, norm='in')
        s = relu(s)
        h = tl.shape(s)[1]
        s = relu(conv(s, ch, h, 1, padding='VALID'))
        s = tf.reshape(fc(s, 64), (-1,1,1,64))
        #### mask ####
        ch = 2 * ch_ori
        mask = z
        for i in range(n_intermedblks):
            mask = ResBlk(mask, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        mask = relu(mask)
        mask = conv(mask, 1, 1, 1)
        return mask * s, mask

def Encoder4(x, attdiff_abs, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff_abs)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff_abs[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, 4*ch))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z) #lrelu(z)
        z = conv(z, ch, 1, 1)
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))
def Encoder4_wIN_wRelu(x, attdiff_abs, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff_abs)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff_abs[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, 4*ch))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z) #lrelu(z)
        z = relu(instance_norm(conv(z, ch, 1, 1)))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))

def Encoder4_wIN_wRelu_concat(x, attdiff, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        attdiff = tf.reshape(attdiff, [-1, 1, 1, tl.shape(attdiff)[-1]])
        attdiff = tf.tile(attdiff, [1, tl.shape(x)[1], tl.shape(x)[2], 1])
        z = tf.concat([x, attdiff], axis = -1)
        z = conv(z, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z) #lrelu(z)
        z = relu(instance_norm(conv(z, ch, 1, 1)))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))
def Encoder4_wIN_concat(x, attdiff, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        attdiff = tf.reshape(attdiff, [-1, 1, 1, tl.shape(attdiff)[-1]])
        attdiff = tf.tile(attdiff, [1, tl.shape(x)[1], tl.shape(x)[2], 1])
        z = tf.concat([x, attdiff], axis = -1)
        z = conv(z, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z) #lrelu(z)
        z = instance_norm(conv(z, ch, 1, 1))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))
def Encoder4_wIN_concat_wolabel(x, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        z = x
        z = conv(z, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z) #lrelu(z)
        z = instance_norm(conv(z, ch, 1, 1))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))
def Encoder4_wIN_concat_C(x, attdiff, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        attdiff = tf.reshape(attdiff, [-1, 1, 1, tl.shape(attdiff)[-1]])
        attdiff = tf.tile(attdiff, [1, tl.shape(x)[1], tl.shape(x)[2], 1])
        z = tf.concat([x, attdiff], axis = -1)
        z = conv(z, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            ch *= 2
            z = ResBlk(z, ch, activation=relu, norm='in')
        z = relu(z) #lrelu(z)
        z = instance_norm(conv(z, ch, 1, 1))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))

def Encoder4_shareLastConv(x, attdiff_abs, name='Gencoder',n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff_abs)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff_abs[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, 4*ch))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z) #lrelu(z)
    with tf.variable_scope('GshareConv', reuse=tf.AUTO_REUSE):
        z = relu(instance_norm(conv(z, ch, 1, 1)))
    return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))

def IN_forEncoder(x):
    with tf.variable_scope('Gencoder_IN', reuse=tf.AUTO_REUSE):
        x = instance_norm(x)
        return x

def SharedLayer_forEncoder(x):
    with tf.variable_scope('Gsharelayer', reuse=tf.AUTO_REUSE):
        channel = tl.shape(x)[-1]
        # x = conv(x, channel//2, 3,1, normalizer_fn=instance_norm, activation_fn=relu) #_kernel3
        # x = conv(x, channel // 2, 1, 1, normalizer_fn=instance_norm, activation_fn=relu) _kernel1
        # x = conv(x, channel // 2, 1, 1)  #kernel1_woIN_woRelu
        # weight1 = tf.get_variable('w1', shape = (channel//2,), initializer=tf.constant_initializer(0.5))  #2channelweight
        # weight2 = tf.get_variable('w2', shape=(channel // 2,), initializer=tf.constant_initializer(0.5))
        # x = weight1 * x[:,:,:,:channel//2] + weight2 * x[:,:,:,channel//2:]
        # weight = tf.get_variable('w', shape=(channel // 2,), initializer=tf.constant_initializer(0))  # channelweight
        # weight =sigmoid(weight)
        # x = weight * x[:, :, :, :channel // 2] + (1-weight) * x[:, :, :, channel // 2:]
        # weight1 = tf.get_variable('w1', shape = (1,), initializer=tf.constant_initializer(0.5))  #2weight
        # weight2 = tf.get_variable('w2', shape=(1,), initializer=tf.constant_initializer(0.5))
        # x = weight1 * x[:,:,:,:channel//2] + weight2 * x[:,:,:,channel//2:]
        x = conv(x, channel, 1, 1, normalizer_fn=instance_norm, activation_fn=relu)
        return x



def Encoder4_wIN(x, attdiff_abs, n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope('Gencoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff_abs)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff_abs[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, 4*ch))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = conv(x, ch, 1, 1)
        for i in range(n_downblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        for i in range(n_intermedblks):
            z = ResBlk(z, ch, activation=relu, norm='adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        z = instance_norm(conv(z, ch, 1, 1))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))

def Encoder5_wIN(x, attdiff_abs, name, n_downblks = 4, n_intermedblks = 2, n_mlp = 4, ch = 16):
    '''

    :param x:
    :param attdiff_abs:
    :param n_downblks:
    :param n_intermedblks:
    :param n_mlp:
    :param ch:
    :return: z
    '''
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE): #'Gencoder'
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff_abs)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff_abs[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, 4*ch))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = relu(conv(x, ch, 1, 1))
        for i in range(n_downblks):
            ch *= 2
            z = relu(instance_norm(conv(z, ch, 3, 2)))
        for i in range(n_intermedblks):
            ch *= 2
            z = relu(conv_adain(z, ch, 3, 1, s = a))

        z = instance_norm(conv(z, ch, 1, 1))
        return z, tf.layers.flatten(tf.layers.average_pooling2d(z, tl.shape(z)[1],1))

def Discriminator(x, n_att, n_resblks = 6, ch = 16):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_gan = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_gan = fc(logit_gan, 1)

        logit_att_feature = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_att = fc(logit_att_feature, n_att)

        # return logit_gan, logit_att
        return logit_gan, logit_att, logit_att_feature

def Discriminator_starganV2(x, n_att, l, n_resblks = 6, ch = 16):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):  # , regularizer=l2_reg
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            if ch >= 512:
                ch = 512
            else:
                ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_att_feature = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_att = fc(logit_att_feature, n_att)
        logit_att = tf.reduce_sum(logit_att * l, axis=1) # consider [N,1]
        logit_att = tf.reshape(logit_att, [logit_att.shape[0], 1])

        logit_gender_feature = lrelu(conv(z, ch, h, 1, padding='VALID'))
        logit_gender = fc(logit_gender_feature, 1) # [N,1]

        # return logit_gan, logit_att
        return logit_att, logit_gender

def LatentDiscriminator(x, n_layers = 4, ch = 16):
    conv_lrelu = partial(conv, activation_fn=lrelu)
    with tf.variable_scope('DiscriminatorLatent', reuse=tf.AUTO_REUSE):
        z = lrelu(conv(x, ch, 1, 1))
        for i in range(n_layers):
            ch *= 2
            z = conv_lrelu(z, ch, 3,2)
        h = tl.shape(z)[1]

        logit_gan = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_gan = fc(logit_gan, 1)

        return logit_gan

def Discriminator_multiTask(x, n_att, n_resblks = 6, ch = 16):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_gan = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_gan = fc(logit_gan, 1)

        logit_att_list = []
        logit_att_feature_list = []
        for i in range(n_att):
            logit_att_feature = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
            logit_att_feature = lrelu(fc(logit_att_feature, 64))
            logit_att = fc(logit_att_feature, 1)
            logit_att_feature_list.append(logit_att_feature)
            logit_att_list.append(logit_att)

        logit_att_feature = tf.concat(logit_att_feature_list, axis = -1)
        logit_att = tf.concat(logit_att_list, axis=-1)

        return logit_gan, logit_att, logit_att_feature

def PatchDiscriminator(x, n_att, n_resblks = 6, ch = 16):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_gan = lrelu(conv(z, ch, 1, 1))
        logit_gan = conv(logit_gan, 1, 1, 1)

        logit_att = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_att = fc(logit_att, n_att)

        return logit_gan, logit_att

def Classifier(x, n_att, n_resblks = 6, ch = 16):
    final_out = []
    final_vec = []
    with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_att_feature = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        data = fc(logit_att_feature, 2048)
        for i in range (n_att):
            data_fc = fc(data, 1024, scope='fc_%d' % i)
            data_i = fc(data_fc, 2, scope='final_fc_%d' % i)
            final_out.append(data_i)
            final_vec.append(data_fc)
        # logit_att = tf.concat(final_out, 1)

        return final_out, final_vec

def Discriminator_withMatch(x, n_att, output_dim = 16*2, n_resblks = 6, ch = 16):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=lrelu, norm = 'none')
        z = lrelu(z)
        h = tl.shape(z)[1]

        logit_gan = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_gan = fc(logit_gan, 1)

        logit_att = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_att = fc(logit_att, n_att)

        logit_match = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_match = fc(logit_match, output_dim)

        return logit_gan, logit_att, logit_match

def EncoderR(x, output_dim, n_resblks = 6, ch = 16):
    # with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('DencoderR', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'none')
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out = fc(logit_out, output_dim)
        return logit_out

def EncoderR_2(x, style, output_dim, n_resblks = 6, ch = 16):
    # with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('DencoderR', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'adain', s_rand = style, norm_name= 'adain%d' %i)
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out = fc(logit_out, output_dim)
        return logit_out

def EncoderR_3(x, x_src, attdiff, output_dim, n_mlp, fc_dim, n_resblks = 6, ch = 16):
    # with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('DencoderR', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = tf.concat([x, x_src], axis = -1)
        z = conv(z, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out = fc(logit_out, output_dim)
        return logit_out

def EncoderR_4(x, x_src, attdiff, output_dim, n_mlp, fc_dim, n_resblks = 6, ch = 16):
    with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)

        z = tf.concat([x, x_src], axis = -1)
        z = conv(z, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out_mu = fc(logit_out, output_dim)
        logit_out_logvar = fc(logit_out, output_dim)
        logit_out = logit_out_mu + tf.random_normal(shape=tf.shape(output_dim)) * tf.exp(logit_out_logvar)
        return logit_out, logit_out_mu, logit_out_logvar

def EncoderR_VAE(x, output_dim, n_resblks = 6, ch = 16):
    with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out_mu = fc(logit_out, output_dim)
        logit_out_logvar = fc(logit_out, output_dim)
        logit_out = logit_out_mu + tf.random_normal(shape=tf.shape(logit_out_mu)) * tf.exp(logit_out_logvar / 2)
        return logit_out, logit_out_mu, logit_out_logvar

# def EncoderR_NoVAE(x, output_dim, n_resblks = 6, ch = 16):
#     with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
#         z = conv(x, ch, 1, 1)
#         for i in range(n_resblks):
#             ch *= 2
#             z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'in')
#         z = relu(z)
#         h = tl.shape(z)[1]
#
#         logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
#         logit_out = fc(logit_out, output_dim)
#         return logit_out
def EncoderR_NoVAE(x, attdiff, output_dim, n_mlp, fc_dim, n_resblks = 6, ch = 16):
    with tf.variable_scope('GencoderR', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out= fc(logit_out, output_dim)
        return logit_out
def EncoderR_NoVAE_ER(x, attdiff, output_dim, n_mlp, fc_dim, n_resblks = 6, ch = 16):
    with tf.variable_scope('ER', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('mlp', reuse=tf.AUTO_REUSE):
            n_stream = tl.shape(attdiff)[-1]
            a_s = []
            for s in range(n_stream):
                a = attdiff[:,s:s+1]
                for i in range(n_mlp):
                    a = relu(fc(a, fc_dim))
                a_s.append(a)
            a = tf.add_n(a_s)
        z = conv(x, ch, 1, 1)
        for i in range(n_resblks):
            ch *= 2
            z = ResBlk(z, ch, downsampling=True, activation=relu, norm = 'adain', s_rand = a, norm_name= 'adain%d' %i)
        z = relu(z)
        h = tl.shape(z)[1]

        logit_out = lrelu(conv(z, ch, h, 1, padding = 'VALID'))
        logit_out= fc(logit_out, output_dim)
        return logit_out

def attention_HW(F_s, F_t):
    with tf.variable_scope('Gattention', reuse=tf.AUTO_REUSE):
        inputs_shape = F_s.get_shape().as_list()
        batchsize, height, width, C = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
        # height = height // 2
        # width = width // 2
        # C = C * 4
        query_conv = conv(F_s, C, 1, 1, padding='VALID')
        key_conv = conv(F_t, C, 1, 1, padding='VALID')
        value_conv = conv(F_t, C, 1, 1, padding='VALID')

        proj_key = tf.reshape(key_conv, [batchsize, width * height, -1])
        proj_query = tf.transpose((tf.reshape(query_conv, [batchsize, width * height, -1])), perm=[0, 2, 1])
        energy = tf.transpose(tf.matmul(proj_key, proj_query), perm=[0, 2, 1])

        attention = tf.nn.softmax(energy)/8.0
        proj_value = tf.reshape(value_conv, [batchsize, width * height, -1])

        out = tf.matmul(attention, proj_value)
        out = tf.reshape(out, [batchsize, height, width, C])
        return out
def attention_C(F_s, F_t):
    with tf.variable_scope('Gattention', reuse=tf.AUTO_REUSE):
        inputs_shape = F_s.get_shape().as_list()
        batchsize, height, width, C = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]
        query_conv = conv(F_s, C, 1, 1, padding='VALID')
        key_conv = conv(F_t, C, 1, 1, padding='VALID')
        value_conv = conv(F_t, C, 1, 1, padding='VALID')

        proj_key = tf.transpose(tf.reshape(key_conv, [batchsize, width * height, -1]), perm=[0, 2, 1])
        proj_query = tf.reshape(query_conv, [batchsize, width * height, -1])
        energy = tf.transpose(tf.matmul(proj_key, proj_query), perm=[0, 2, 1])

        attention = tf.nn.softmax(energy)/64.0
        proj_value = tf.transpose(tf.reshape(value_conv, [batchsize, width * height, -1]), perm=[0, 2, 1])

        out = tf.transpose(tf.matmul(attention, proj_value), perm=[0, 2, 1])
        out = tf.reshape(out, [batchsize, height, width, C])
        return out