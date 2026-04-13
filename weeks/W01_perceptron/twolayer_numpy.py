#Architecture:
#Input layer: 2 neurons (X1, X2)
#Hidden layer: 4 neurons, sigmoid activation
#Output layer: 1 neuron, sigmoid activation
#Requirements:
#Implement forward pass through both layers
#Implement backpropagation through both layers manually (chain rule through the hidden layer)
#Use sigmoid activation and binary cross-entropy loss
#Train with gradient descent, learning rate 0.5, 10000 epochs
#Print the loss every 500 epochs
#After training, print predicted probabilities and whether XOR passes

import numpy as np

# four samples (4,2) 
X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)

# initialize w and b with random values
W1 = np.random.randn(2,4)  # two weights each for four neurons of hidden layer
b1 = np.random.randn(4)    # one bias each for four neurons of hidden layer
W2 = np.random.randn(4,1)  # four weights for one output neuron's four inputs
b2 = np.random.randn(1)    # one bias for output neuron

# XOR expected output (4,)
y = np.array([[0],[1],[1],[0]],dtype=float)

def forward_pass(X,w,b):
    z = np.dot(X,w) + b
    return z

def sigmoid(z):
    y_hat = (1 / (1 + np.exp(-z)))
    return y_hat

def sigmoid_deriv(a):
    d = a * (1-a)
    return d

def loss_function(y,y_hat):
    loss = (-y * np.log(y_hat) - (1-y) * np.log(1-y_hat)) # returns a (4,) loss array
    avg_loss = np.mean(loss)
    return avg_loss

def backward_pass(X,y_hat,y):
    dw = X.T @ (y_hat - y)
    db = np.mean(y_hat - y)
    return dw,db

lr = 0.5
for epoch in range(10000):
    z1 = forward_pass(X,W1,b1)
    a1 = sigmoid(z1)
    z2 = forward_pass(a1,W2,b2)
    a2 = sigmoid(z2)
    loss = loss_function(y,a2)
    
    # backpropagation
    dw2,db2 = backward_pass(a1,a2,y)
    W2 = W2 - lr * dw2
    b2 = b2 - lr * db2

    dz1 = (a2 - y) @ W2.T * sigmoid_deriv(a1)   # (4,4) * (4,4) = (4,4)
    dW1 = X.T @ dz1                             # (2,4) & (4,4) = (2,4) - same as W1
    db1 = np.mean(dz1,axis=0)                   # (4,) - Same shape as b1
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    

    if epoch%500==0:
        print("loss:", loss)

z1 = forward_pass(X,W1,b1)
a1 = sigmoid(z1)
z2 = forward_pass(a1,W2,b2)
y_hat_final = sigmoid(z2)
predictions = (y_hat_final >= 0.5).astype(int)  #converts sigmoid predictions to 0 and 1
passed = np.array_equal(predictions, y.astype(int))
print(f"\n{'PASS' if passed else 'FAIL'}")
print(f"Predicted: {np.round(y_hat_final, 4)}")
print(f"Expected:  {y.astype(int)}")





