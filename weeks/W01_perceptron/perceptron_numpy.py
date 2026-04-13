import numpy as np
import math

# four samples each with two inputs, x1 and x2
X = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
# w,b are randomly chosen, no need for definition

# expected outputs, `y` for each case
gates = {
    'AND':np.array([0,0,0,1],dtype=float),
    'OR':np.array([0,1,1,1],dtype=float),
    'XOR':np.array([0,1,1,0],dtype=float)
}


def forward(X,w,b):
    z = np.dot(X,w) + b
    return z

def sigmoid(z):
    y_hat = 1 / (1+np.exp(-z))
    return y_hat

def bce_loss(y,y_hat):
    loss = (- y * np.log(y_hat) - (1-y) * np.log(1-y_hat)) # returns a (4,) array
    avg_loss = np.mean(loss)   # returns a scalar
    return avg_loss

def backward_pass(X,y_hat,y):
    # pd1 = X    # sensitivity of linear output to weights
    # pd2 = y_hat * (1-y_hat) # sensitivity of prediction to linear input
    # pd3 = ((-y/y_hat) + ((1-y)/(1-y_hat))) # sensitivity of loss to prediction output
    dw = X.T @ (y_hat - y) / len(y) # gradient for each weight
    db = np.mean(y_hat - y)  # the pd of the last term dz/db is 1, so the chain rule only needs the first two terms
    return dw,db
    

def train(X,y,epochs=5000,lr=0.5):
    # pre-requisite, random initialization of w and b
    w = np.random.randn(2)
    b = np.random.randn()
    for epoch in range(epochs):
        # implements a single perceptron, single layer neural network
        # first forward pass produces `z`
        z = forward(X,w,b)
        # next, sigmoid function calculates `y_hat`, predicted output
        y_hat = sigmoid(z)
        # then, loss function is used to calc difference between `y` (one of the gates) and `y_hat`
        loss = bce_loss(y,y_hat)
        # backward pass is performed to calculate sensitivity of loss to each weight
        dw,db = backward_pass(X,y_hat,y)
        w = w - lr * dw
        b = b - lr * db
        if epoch%100==0:
            print("loss: ", round(loss,4))
    return w,b

for key,value in gates.items():
    w,b = train(X,value)
    y_hat_final = sigmoid(forward(X,w,b))
    predictions = (y_hat_final >= 0.5).astype(int)  #converts sigmoid predictions to 0 and 1
    passed = np.array_equal(predictions, value.astype(int))
    print(f"\n{key} gate — {'PASS' if passed else 'FAIL'}")
    print(f"Predicted: {np.round(y_hat_final, 4)}")
    print(f"Expected:  {value.astype(int)}")


    