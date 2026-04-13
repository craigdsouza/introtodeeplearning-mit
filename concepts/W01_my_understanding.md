classification problem
The classification problem here is to identify/segment which pixels fall into different categories "person", "car", "traffic light" , "road lane marking" etc. Individual pixels by themselves are not useful as features to help identify class. You need to identify higher level "features". For instance edges are a higher order feature. Layers of a neural network identify these higher order features. The higher order features are pieced together to identify an object level class. Mathematically, a higher level feature is approximated with a non-linear function. Linear classifiers are the basis of deep learning perceptrons. To make these linear classifiers recognise Non linear functions (shapes in images), requires non linear activation functions.

neural networks
the simplest neural network is just a single perceptron, which is a linear classifier function, with weights (w0,w1,...,wi) given to a set of inputs (x1,x2,...,xi) and a bias (w0) added to the weighted sum of the inputs. 

set of inputs
inputs in a 2D classification problem are simply pixels on a grid, a small number of which are selected for training and the rest for testing. 

forward pass (z) + activation function (y_hat)
The learning starts in a forward pass, with a random set of weights and bias , the linear output is calculated by multiplying the set of weights to the set of inputs (training data) and then an activation function is applied to it to make it non-linear, since most real world phenomenon are non linear , we need to learn have a way to capture non linearities. 

This forward pass is completed with a batch of inputs from your training data. i.e. a set of 32-256 sample inputs (x0,x1,...,xi) all multiplied with the same ONE set of random weights. The end result is an array of predictions 

loss function
The output of the activation function (AF) is then compared to the actual known correct value for each sample of inputs. The comparison is done with something called a loss function. The average loss across all samples is calculated. 

Next, comes the step where the actual learning happens: Gradient descent. The sensitivity of the Loss to each weight is individually calculated using the chain rule -> then average of all samples in batch (sensitivity of the loss to the output * sensitivity of the AF output to its input * sensitivity of the AF input to the weights. )

Using this sensitivity the individual weights are updated by multiplying by the learning rate and summing with the prior value of the weight. Once all weights are updated a second forward pass happens. After each pass we see the loss and once this loss (training loss) falls below the testing loss (loss calculated over a set of sample inputs not used in the training process) we stop the training to avoid overfitting in a process called regularization.



