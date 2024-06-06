from differentiation import numerical_gradient
import numpy as np

def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function(x):
    return x[0]**2 + x[1]**2

# example
init_x = np.array([-3.0, 4.0])
array = gradient_descent(function, init_x = init_x, lr=0.1, step_num=100)
print(array)