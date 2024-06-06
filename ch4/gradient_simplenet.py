import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch3 import softmax
from error import cross_entropy_error
from differentiation import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax.softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])
net.loss(x, t)

def f(W):
    return net.loss
# f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)