# cnn-experimentation
Experimentation with tuning, hyperparameters, loss functions, optimizers, and architecture on simple FashionMNIST datset using Pytorch. Goal is to learn what changes do what, and why they do that.

<h2>Baseline Model</h2>

First implemntation of CNN for Fashion MNIST dataset. I used a simple model and hyperparameters for a baseline. Two convolutional layers with 5x5 kernels producing 5, then 15 feature maps respectively. Each convolutional layer is followed by a max pooling layers with a 2x2 kernel. Ending with 15 7x7 feature maps flattened in a 735 dimensional vector into a 128, 32, 10 fully connected multilayer perceptron, with ReLu activation functions. I used cross entropy loss for my loss function and stochastic gradient descent for my optimizer, with a learing rate of 0.01. Realistically a learning rate of 0.001 or 0.005 would have been better but I wanted a quick baseline testing model, and with 15 epochs it reached a validation accuracy of ~87%.

<h4>Potential Changes</h4>

 - Deeper Architecture 
 - Different pooling/activation functions
 - Data transformations
 - Effect of batch size on accuracy
 - Different optimizers; Adam, SGH with momentum, etc.
 - Experiment with different loss functions
 - Normalization/Dropout
 - Weight Initialization


