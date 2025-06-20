from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class MultiLayerConvNet(object):
    """
    A multi-layer convolutional network with the following architecture:

    {conv - relu} x (M) - 2x2 max pool - {affine - relu} x (L-1) - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    
    The {conv-relu} block is repeated M times and the {affine-relu} block is repeated L-1 times
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

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        C, H, W = input_dim
        num_conv_layers = len(nums_filters)
        self.num_conv_layers = num_conv_layers
        self.num_affine_layers = len(hidden_dims)
        self.num_layers = self.num_conv_layers + self.num_affine_layers + 1
        
        for i, num_filters in enumerate(nums_filters):
            self.params[f'W{i+1}'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
            self.params[f'b{i+1}'] = np.zeros(num_filters)
            C = num_filters

        if num_conv_layers>0:
            layer_input_dim = (H // 2) * (W // 2) * num_filters
        else:
            layer_input_dim = C*H*W
            
        for i, hidden_dim in enumerate(hidden_dims):
            self.params[f'W{num_conv_layers+i+1}'] = np.random.normal(0, weight_scale, (layer_input_dim, hidden_dim))
            self.params[f'b{num_conv_layers+i+1}'] = np.zeros(hidden_dim)
            layer_input_dim = hidden_dim

        # Last layer: from last hidden layer to output
        self.params[f'W{self.num_layers}'] = np.random.normal(0, weight_scale, (layer_input_dim, num_classes))
        self.params[f'b{self.num_layers}'] = np.zeros(num_classes)

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

        #for param_name in sorted(self.params):
            #print("%s", param_name)
            #print(type(self.params[param_name]), self.params[param_name].shape)

        cache = {}
        layer_input = X
        
        if self.num_conv_layers>0:
          for i in range(1, self.num_conv_layers):
            W, b = self.params[f'W{i}'], self.params[f'b{i}']
            
            # Forward pass: Conv - ReLU 
            out_conv, cache_conv = conv_relu_forward(layer_input, W, b, conv_param)
            
            cache[f'conv_cache{i}'] = cache_conv
            
            layer_input = out_conv
        
          W, b = self.params[f'W{self.num_conv_layers}'], self.params[f'b{self.num_conv_layers}']    
          out_pool, cache_pool = conv_relu_pool_forward(layer_input, W, b, conv_param, pool_param)   
            
          cache['conv_pool_cache'] = cache_pool
        
          layer_input = out_pool
        
        for i in range(1, self.num_affine_layers+1):
            W, b = self.params[f'W{self.num_conv_layers+i}'], self.params[f'b{self.num_conv_layers+i}']
            # Forward pass: Affine - ReLU
            out_affine, cache_affine = affine_relu_forward(layer_input, W, b)
            cache[f'affine_cache{i}'] = cache_affine
            layer_input = out_affine

        W, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, scores_cache = affine_forward(layer_input, W, b)
        cache['scores_cache'] = scores_cache

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

       # 计算损失（Softmax + 正则化）
        loss,dscores=softmax_loss(scores,y)

        for i in range(1, self.num_layers+1):
            W = self.params[f'W{i}']
            # L2 正则化
            loss += 0.5 * self.reg * np.sum(W * W)  

        # 开始反向传播
        dout, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(dscores, cache['scores_cache'])

        # 加上正则化梯度
        grads[f'W{self.num_layers}'] += self.reg * self.params[f'W{self.num_layers}']  

        # 从倒数第二层向前
        for i in range(self.num_affine_layers,0,-1):
            
            # Affine-Relu backward
            dout, grads[f'W{self.num_conv_layers+i}'], grads[f'b{self.num_conv_layers+i}'] = affine_relu_backward(dout, cache[f'affine_cache{i}'])
            
            # Regularization gradient
            grads[f'W{self.num_conv_layers+i}'] += self.reg * self.params[f'W{self.num_conv_layers+i}']  

        if self.num_conv_layers>0:
          dout, grads[f'W{self.num_conv_layers}'], grads[f'b{self.num_conv_layers}'] = conv_relu_pool_backward(dout, cache['conv_pool_cache'])
          grads[f'W{self.num_conv_layers}'] += self.reg * self.params[f'W{self.num_conv_layers}']
        
          for i in range(self.num_conv_layers-1, 0, -1):
            dout, grads[f'W{i}'], grads[f'b{i}'] = conv_relu_backward(dout, cache[f'conv_cache{i}'])
            
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        #for param_name in sorted(grads):
            #print("%s", param_name)
            #print(type(grads[param_name]), self.params[param_name].shape)

    
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads