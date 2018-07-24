import tensorflow as tf
import numpy as np

def allen_net(x, variables_dict, training=True):
    """
    x: input tensor
    variables_dict: dict
        A dictionary into which variables and tensors are saved as the graph is being defined. 
        For debugging purposes
    training: bool
        whether we are in training mode
    """

    C_in = 1
    C_out = 1
    C_mid, C_mul = 32, 32
    assert C_mul * C_in == C_mid
    blocks = 4

    with tf.variable_scope('u_net'):
        x = allen_sub_u_net(x, variables_dict, C_in, C_mult=C_mid // C_in, blocks=blocks, training=training)
    with tf.variable_scope('after_u'):
        x = conv3_br(x, C=C_mid, L=C_out, variables_dict=variables_dict, training=training)
    return x

def set_dict_no_overwrite(d, k, v):
    """
    d: dict
    k: key
    v: value
    """
    assert k not in d
    d[k] = v

def allen_sub_u_net(x, variables_dict, C, C_mult, down_stride=2, blocks=4, training=True):
    """
    Subnet portion of a u-net copied from the pytorch system. Calls itself recursively
    C_mult is hardcoded to 2 in recursive calls
    x: input tensor
    variables_dict: dict
        A dictionary into which variables and tensors are saved as the graph is being defined. 
        For debugging purposes
    C: channels
    C_mult: int
        channel count multiplier in recursive call
    down_stride: int
        spatial stride used before increasing number of channels in recursive call
    blocks: int
        number of blocks to recurse for
    training: bool
        whether we are in trining
    """
    assert blocks <= 10
    i = blocks
    in_shape = tf.shape(x)
    L = C_mult * C # number of output channels -- L for latent
    if i <= 0:
        with tf.variable_scope('conv_inner_a'):
            set_dict_no_overwrite(variables_dict, 'conv_inner_a', dict())
            x = conv3_br(x, C=C, L=L, variables_dict=variables_dict['conv_inner_a'], training=training)
        with tf.variable_scope('conv_inner_b'):
            set_dict_no_overwrite(variables_dict, 'conv_inner_b', dict())
            x = conv3_br(x, C=L, variables_dict=variables_dict['conv_inner_b'], training=training)
        return x
    

    key = 'conv_more%0.2d' % i # C -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=C, L=L, variables_dict=variables_dict[key], training=training)

    key = 'conv_into%0.2d' % i # L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, variables_dict=variables_dict[key], training=training)

    x_into = x

    key = 'conv_down%0.2d' % i # L -> L, reduce spatial resolution
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, kernel_size=down_stride, strides=(1, down_stride, down_stride, down_stride, 1),
                     variables_dict=variables_dict[key], training=training)

    # L -> 2 * L
    x = allen_sub_u_net(x, variables_dict, C=L, C_mult=2, down_stride=down_stride, blocks=blocks-1, training=training)
    # output of allen_sub_u_net call has C_mult * L channels

    key = 'conv_up%0.2d' % i # 2 * L -> L, recover spatial resolution
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())

        up_shape = tf.concat([in_shape[:4], [L]], axis=0)
        x = conv3t_br(x, C=L, L=2*L, 
                      output_shape=up_shape,
                      kernel_size=down_stride,
                      strides=(1, down_stride, down_stride, down_stride, 1),
                      variables_dict=variables_dict[key], training=training)
    x_up = x
    x = tf.concat([x_into, x_up], axis=4) # NDHWC, results in in 2*L channels

    key = 'conv_less%0.2d' % i # 2 * L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=2*L, L=L, variables_dict=variables_dict.get(key, dict()), training=training)

    key = 'conv_outof%0.2d' % i # L -> L
    with tf.variable_scope(key):
        set_dict_no_overwrite(variables_dict, key, dict())
        x = conv3_br(x, C=L, L=L, variables_dict=variables_dict.get(key, dict()), training=training)
    return x

def conv3t_br(x, 
              C, # C is the output size, so as to pair up with conv
              L, # L is the input size, so as to pair up with conv
              output_shape,
              kernel_size,
              strides,
              padding='SAME',
              data_format='NDHWC',
              dilations=(1,1,1,1,1),
              variables_dict=None,
              training=True):
    """
    Wrapper around conv3d_transpose, batch_normalization, relu
    """
    
    assert data_format == 'NDHWC'
    ks = kernel_size
    
    w_xavier = tf.initializers.truncated_normal(0, stddev=np.sqrt(2. / (ks*ks*ks*L)), seed=0, dtype=tf.float32) # notice slight difference from conv3_br
    w = tf.get_variable(name='w', shape=[ks, ks, ks, C, L], dtype=tf.float32, initializer=w_xavier)
    b = tf.get_variable(name='b', shape=[L], dtype=tf.float32, initializer=tf.initializers.zeros(tf.float32))

    x_convt = tf.nn.conv3d_transpose(x + b, w, output_shape=output_shape, strides=strides, data_format=data_format, name=None)
    bn_layer = tf.layers.BatchNormalization() # container object that allows us to access moving mean, moving variance
    x_bn = bn_layer.apply(x_convt, training=training)
    x_relu = tf.nn.relu(x_bn)

    if variables_dict is not None:
        variables_dict['w'] = w
        variables_dict['b'] = b
        variables_dict['bn_layer'] = bn_layer
        variables_dict['gamma'] = bn_layer.gamma
        variables_dict['beta'] = bn_layer.beta
        variables_dict['moving_mean'] = bn_layer.moving_mean
        variables_dict['moving_variance'] = bn_layer.moving_variance
    return x_relu

def conv3_br(x,
             C=None,
             L=None,
             kernel_size=3,
             strides=(1,1,1,1,1),
             padding='SAME',
             data_format='NDHWC',
             dilations=(1,1,1,1,1),
             variables_dict=None,
             training=True,
             do_bn=True,
             do_relu=True):
    """
    Wrapper around conv3d_transpose, batch_normalization, relu
    """
    
    assert data_format == 'NDHWC'
    ks = kernel_size
    if C is None:
        C = tf.get_shape(x).as_list()[data_format.index('C')]
    if L is None:
        L = C

    w_xavier = tf.initializers.truncated_normal(0, stddev=np.sqrt(2. / (ks*ks*ks*C)), seed=0, dtype=tf.float32)
    w = tf.get_variable(name='w', shape=[ks, ks, ks, C, L], dtype=tf.float32, initializer=w_xavier)
    b = tf.get_variable(name='b', shape=[L], dtype=tf.float32, initializer=tf.initializers.zeros(tf.float32))
    print(w, b)
    x_conv = tf.nn.conv3d(x, w, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=None) + b
    if do_bn:
        bn_layer = tf.layers.BatchNormalization() # container object that allows us to access moving mean, moving variance
        x_bn = bn_layer.apply(x_conv, training=training)
    if do_relu:
        x_relu = tf.nn.relu(x_bn)

    if variables_dict is not None:
        variables_dict['w'] = w
        variables_dict['b'] = b
        if do_bn:
            variables_dict['bn_layer'] = bn_layer
            variables_dict['gamma'] = bn_layer.gamma
            variables_dict['beta'] = bn_layer.beta
            variables_dict['moving_mean'] = bn_layer.moving_mean
            variables_dict['moving_variance'] = bn_layer.moving_variance
    return x_relu



"""
import torch
import pdb

class Net(torch.nn.Module):
    def __init__(self,
                 depth=4,
                 mult_chan=32,
                 in_channels=1,
                 out_channels=1,
    ):
        super().__init__()
        self.depth = depth
        self.mult_chan = mult_chan
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.net_recurse = _Net_recurse(n_in_channels=self.in_channels, mult_chan=self.mult_chan, depth=self.depth)
        self.conv_out = torch.nn.Conv3d(self.mult_chan, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_rec = self.net_recurse(x)
        return self.conv_out(x_rec)


class _Net_recurse(torch.nn.Module):
    def __init__(self, n_in_channels, mult_chan=2, depth=0):

        super().__init__()
        self.depth = depth
        n_out_channels = n_in_channels*mult_chan
        self.sub_2conv_more = SubNet2Conv(n_in_channels, n_out_channels)
        
        if depth > 0:
            self.sub_2conv_less = SubNet2Conv(2*n_out_channels, n_out_channels)
            self.conv_down = torch.nn.Conv3d(n_out_channels, n_out_channels, 2, stride=2)
            self.bn0 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu0 = torch.nn.ReLU()
            
            self.convt = torch.nn.ConvTranspose3d(2*n_out_channels, n_out_channels, kernel_size=2, stride=2)
            self.bn1 = torch.nn.BatchNorm3d(n_out_channels)
            self.relu1 = torch.nn.ReLU()
            self.sub_u = _Net_recurse(n_out_channels, mult_chan=2, depth=(depth - 1))
            
    def forward(self, x):
        if self.depth == 0:
            return self.sub_2conv_more(x)
        else:  # depth > 0
            x_2conv_more = self.sub_2conv_more(x)
            x_conv_down = self.conv_down(x_2conv_more)
            x_bn0 = self.bn0(x_conv_down)
            x_relu0 = self.relu0(x_bn0)
            x_sub_u = self.sub_u(x_relu0)
            x_convt = self.convt(x_sub_u)
            x_bn1 = self.bn1(x_convt)
            x_relu1 = self.relu1(x_bn1)
            x_cat = torch.cat((x_2conv_more, x_relu1), 1)  # concatenate
            x_2conv_less = self.sub_2conv_less(x_cat)
        return x_2conv_less

class SubNet2Conv(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(n_in,  n_out, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm3d(n_out)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm3d(n_out)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
"""