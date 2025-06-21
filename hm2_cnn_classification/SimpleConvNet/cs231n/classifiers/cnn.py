from builtins import object
import numpy as np
import cupy as cp

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class MultiLayerConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    {conv - relu} x M - 2x2 max pool - {affine - relu} x (L-1) - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    
    The {conv-relu} block is repeated M times and the {affine-relu} block is repeated L-1 times.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        nums_filters = [16,32],
        filter_size=5,
        hidden_dims = [500,100],
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - nums_filters: A list of integers giving the numbers of filters to use in the convolutional layers. 
                        M = len(nums_filters)
        - filter_size: Width/height of filters to use in the convolutional layer.
        - hidden_dims: A list of integers giving the numbers of units to use in the fully-connected hidden layers.
                        L-1 = len(hidden_dims)
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filter_size = filter_size
        self.nums_filters = nums_filters
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.weight_scale = weight_scale

        ############################################################################
        # TODO: Initialize weights and biases for the multi-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the first layer    #
        # in W1 and b1; for the second layer use W2 and b2, etc.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the convolutional layers are chosen so that                #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        C, H, W = input_dim
        M = len(nums_filters)
        L = len(hidden_dims) + 1  # 全连接层数 = 隐藏层数 + 输出层
        prev_C = C
        prev_H, prev_W = H, W
        # 卷积层参数初始化
        for i, num_filter in enumerate(nums_filters):
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(num_filter, prev_C, filter_size, filter_size)
            self.params[f'b{i+1}'] = np.zeros(num_filter)
            prev_C = num_filter
            # 只在最后一个卷积层后做池化
            if i == len(nums_filters) - 1:
                prev_H = prev_H // 2
                prev_W = prev_W // 2
        # 全连接层参数初始化
        fc_input_dim = prev_C * prev_H * prev_W
        dims = [fc_input_dim] + hidden_dims + [num_classes]
        for i in range(len(dims)-1):
            self.params[f'W{M+i+1}'] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params[f'b{M+i+1}'] = np.zeros(dims[i+1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
            


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the multi-layer convolutional network.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """

        X = X.astype(self.dtype)
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size

        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the multi-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = []
        out = X
        M = len(self.filter_size if isinstance(self.filter_size, list) else self.params) // 2
        M = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 4])
        L = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 2])
        # 卷积层部分
        for i in range(M):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            out, cache = conv_relu_forward(out, W, b, conv_param)
            caches.append(cache)
            # 只在最后一个卷积层后做池化
            if i == M-1:
                out, cache = max_pool_forward_fast(out, pool_param)
                caches.append(cache)
        # 全连接层部分
        for i in range(L):
            W = self.params[f'W{M+i+1}']
            b = self.params[f'b{M+i+1}']
            if i == L-1:
                out, cache = affine_forward(out, W, b)
            else:
                out, cache = affine_relu_forward(out, W, b)
            caches.append(cache)
        scores = out

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the multi-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # 反向传播
        grads = {}
        reg = self.reg
        loss, dscores = softmax_loss(scores, y)
        # L2 正则化项
        for k in self.params:
            if k.startswith('W'):
                loss += 0.5 * reg * np.sum(self.params[k] ** 2)
        # 反向传播
        M = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 4])
        L = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 2])
        dout = dscores
        cache_idx = len(caches) - 1
        # 全连接层部分
        for i in reversed(range(L)):
            W_key = f'W{M+i+1}'
            b_key = f'b{M+i+1}'
            if i == L-1:
                dout, dW, db = affine_backward(dout, caches[cache_idx])
            else:
                dout, dW, db = affine_relu_backward(dout, caches[cache_idx])
            grads[W_key] = dW + reg * self.params[W_key]
            grads[b_key] = db
            cache_idx -= 1
        # 卷积层部分
        if M > 0:
            # 池化层反向
            dout = max_pool_backward_fast(dout, caches[cache_idx])
            cache_idx -= 1
            for i in reversed(range(M)):
                W_key = f'W{i+1}'
                b_key = f'b{i+1}'
                dout, dW, db = conv_relu_backward(dout, caches[cache_idx])
                grads[W_key] = dW + reg * self.params[W_key]
                grads[b_key] = db
                cache_idx -= 1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class MultiLayerConvNetCupy(MultiLayerConvNet):

    def __init__(self,
                 input_dim=(3, 32, 32),
                 nums_filters = [16,32],
                 filter_size=5,
                 hidden_dims = [500,100],
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=np.float32,
                 ):
        super().__init__(input_dim, nums_filters, filter_size, hidden_dims, num_classes, weight_scale, reg, dtype)

        for k, v in self.params.items():
            self.params[k] = cp.asarray(v)
            self.params[k] = self.params[k].astype(dtype)

    """
    支持cupy的多层卷积网络，前后向自动调用cupy实现。
    """
    def loss(self, X, y=None):
        # 若输入为numpy则转为cupy
        if isinstance(X, np.ndarray):
            X = cp.asarray(X)
        if y is not None and isinstance(y, np.ndarray):
            y = cp.asarray(y)
        X = X.astype(self.dtype)
        conv_param = {'stride': 1, 'pad': (self.filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        caches = []
        out = X
        M = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 4])
        L = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 2])
        # 卷积层部分
        for i in range(M):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            W = cp.asarray(W) if isinstance(W, np.ndarray) else W
            b = cp.asarray(b) if isinstance(b, np.ndarray) else b
            out, cache = conv_relu_forward_cupy(out, W, b, conv_param)
            caches.append(cache)
            if i == M-1:
                out, cache = max_pool_forward_cupy(out, pool_param)
                caches.append(cache)
        # 全连接层部分
        for i in range(L):
            W = self.params[f'W{M+i+1}']
            b = self.params[f'b{M+i+1}']
            W = cp.asarray(W) if isinstance(W, np.ndarray) else W
            b = cp.asarray(b) if isinstance(b, np.ndarray) else b
            if i == L-1:
                out, cache = affine_forward(out, W, b)
            else:
                out, cache = affine_relu_forward_cupy(out, W, b)
            caches.append(cache)
        scores = out
        if y is None:
            return scores
        grads = {}
        reg = self.reg
        # softmax_loss需cupy实现，或转回numpy
        if isinstance(scores, cp.ndarray):
            scores_cpu = cp.asnumpy(scores)
            y_cpu = cp.asnumpy(y)
        else:
            scores_cpu = scores
            y_cpu = y
        loss, dscores = softmax_loss(scores_cpu, y_cpu)
        for k in self.params:
            if k.startswith('W'):
                W = self.params[k]
                if isinstance(W, np.ndarray):
                    loss += 0.5 * reg * np.sum(W ** 2)
                else:
                    loss += 0.5 * reg * float(cp.sum(W ** 2))
        # 反向传播
        M = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 4])
        L = len([k for k in self.params if k.startswith('W') and self.params[k].ndim == 2])
        dout = cp.asarray(dscores) if not isinstance(dscores, cp.ndarray) else dscores
        cache_idx = len(caches) - 1
        # 全连接层部分
        for i in reversed(range(L)):
            W_key = f'W{M+i+1}'
            b_key = f'b{M+i+1}'
            if i == L-1:
                dout, dW, db = affine_backward(dout, caches[cache_idx])  # affine_backward需cupy版本可替换
            else:
                dout, dW, db = affine_relu_backward_cupy(dout, caches[cache_idx])
            grads[W_key] = dW + reg * (cp.asarray(self.params[W_key]) if isinstance(self.params[W_key], np.ndarray) else self.params[W_key])
            grads[b_key] = db
            cache_idx -= 1
        # 卷积层部分
        if M > 0:
            dout = max_pool_backward_cupy(dout, caches[cache_idx])
            cache_idx -= 1
            for i in reversed(range(M)):
                W_key = f'W{i+1}'
                b_key = f'b{i+1}'
                dout, dW, db = conv_relu_backward_cupy(dout, caches[cache_idx])
                grads[W_key] = dW + reg * (cp.asarray(self.params[W_key]) if isinstance(self.params[W_key], np.ndarray) else self.params[W_key])
                grads[b_key] = db
                cache_idx -= 1
        # grads转为numpy，便于外部兼容
        for k in grads:
            if isinstance(grads[k], cp.ndarray):
                grads[k] = cp.asnumpy(grads[k])
        return loss, grads