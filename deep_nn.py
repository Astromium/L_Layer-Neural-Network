# Python implementation of an L-layer deep neural network
# 4 helper functions
# 1- initialize_params : to initialize the parameters w, b according to the dimentions of the layers
# 2- forward_propagation : implementing the forward propagation step
# 3- backward_propagation: implementing the backward propagation step
# 4- copute cost : to compute the cost of the logistic regression 
# 5- update_params : to update the parameters w and b using gradient descent

import numpy as np

def ReLU(Z):
    A = np.maximum(0,Z)
    cache = 
    return a, cache
    

def sigmoid(Z):
    A =  1 / (1 + np.exp(-Z))
    cache = Z
    return a, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ    

def initialize_params(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters    

"""
Forward Propagation
divided into 3 steps 
1) linear_forward : computes Z[l] = W[l].A[l-1] + b[l]
    input: A[l-1], b[l], W[l]
    output: Z[l], cache = (A[l-1], b[l], W[l])

2) linear_activation_forward : computes the activation function of a neuron  
    input: A_prev, W, b, activation:String (Relu or sigmoid)
    output: A, cache = (linear_cache = (A_prev, W, b), activation_cache = (Z))

3) L_model_forward: does the forward propagation for L layers
    input: X, parameters
    output: AL, caches    
"""

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b 
    cache = (A, W, b)

    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    L = len(parameters) // 2 #number of layers
    caches = []
    A = X

    # Relu activation for L-1 first layers
    for l in range(1, L-1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu') 
        caches.append(cache)  

    # sigmoid activation for the last layer
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, 'sigmoid')
    caches.append(cache)

    return AL, cache 

"""
Backward propagation
3 steps:
1) linear_backward: computes dW[l], db[l] and dA[l-1] using dZ[l]
    input: dZ[l]
    output: dW[l], db[l], dA[l-1]   

2) linear_activation_backward: computes dZ[l] using dA[l] and Z[l] -> dZ[l] = dA[l] * g'(Z[l])
    input: dA, cache, activation: String
    output: dA_prev, dW, db

3) L_model_backward: does the backward propagation for L layers
    input: AL, Y, caches
    output: gradients     
"""

def linear_backward(dZ, cache):
    A_prev, W, b = cache 
    m = A_prev.shape[1]

    dW = 1/m*np.dot(dZ, A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)    

    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache) 

    return dA_prev, dW, db  

def L_model_backward(AL, Y, caches):
    L = len(caches) # number of layers
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads = {}

    current_cache = caches[L-1]
    dA_prev, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, 'sigmoid')
    grads['dW' + str(L)] = dW_temp
    grads['db' + str(L)] = db_temp
    grads['dA' + str(L-1)] = dA_prev

    for l in revesed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)], current_cache, 'relu')
        grads['dW' + str(l)] = dW_temp
        grads['db' + str(l)] = db_temp
        grads['dA' + str(l-1)] = dA_prev_temp

    return grads

def compute_cost(AL, Y):
    m = Y.shape[1]

    C1 = Y * np.log(AL)
    C2 = (1-Y) * np.log(1-AL)
    cost = -1/m*np.sum(C1+C2)    

    return cost

def update_params(parameters, grads, learning_rate):
    params = parameters.copy()
    L = len(params) // 2

    for l in range(L):
        params['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        params['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]

    return params    

"""
combine all the functions in the final model

"""

def model(X, Y, layers_dims, n_iters, learning_rate):
    parameters = initialize_params(layers_dims)

    for i in range(0, n_iters):
        AL, caches = linear_activation_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = linear_activation_backward(AL, Y, caches)

        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs        