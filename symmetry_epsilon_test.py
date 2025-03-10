import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.ma.extras import ndenumerate
from Simple_Elliptic_PINN_2 import train_PINN, generate_training_points

num_points = 100
num_test_points = 10000
x_test = torch.linspace(0, 1, num_test_points).reshape(-1, 1)

def get_assymetry(model):
    forwards = model(x_test).detach().numpy()
    backwards = forwards[::-1]

    # calculating l2 norm of difference between forward and backward pass of function
    return (np.mean((forwards - backwards) ** 2)) ** 0.5


if __name__ == '__main__':
    # Getting Collocation and Weights
    uniform, uniform_weights = generate_training_points(num_points=num_points)
    gauss_100, gauss_100_weights = generate_training_points(method='gauss_legendre', num_points=num_points)
    gauss_101, gauss_101_weights = generate_training_points(method='gauss_legendre', num_points=101)
    lobatto_100, lobatto_100_weights = generate_training_points(method='gauss_lobatto', num_points=num_points)
    lobatto_101, lobatto_101_weights = generate_training_points(method='gauss_lobatto', num_points=101)

    one_to_ten = [i for i in range(1, 10)]
    ten_to_one = one_to_ten[::-1]

    epsilon_list = []

    for j in range(1, 4):
        for i in ten_to_one:
            epsilon_list.append(i * (10 ** -j))

    epsilon_array = np.array(epsilon_list)
    uniform_assymetry_array = np.zeros_like(epsilon_array)
    gauss_100_assymetry_array = np.zeros_like(epsilon_array)
    # gauss_101_assymetry_array = np.zeros_like(epsilon_array)
    lobatto_100_assymetry_array = np.zeros_like(epsilon_array)
    # lobatto_101_assymetry_array = np.zeros_like(epsilon_array)

    for i, epsilon in ndenumerate(epsilon_array):
        print("Epsilon: ", epsilon)
        # training on uniform collocation
        print("Uniform")
        model = train_PINN(uniform, uniform_weights,epsilon)
        uniform_assymetry_array[i] = get_assymetry(model)
        # training on gauss collocation (100 points)
        print("Gauss")
        model = train_PINN(gauss_100, gauss_100_weights,epsilon)
        gauss_100_assymetry_array[i] = get_assymetry(model)
        # training on lobatto collocation (100 points)
        print("Lobatto")
        model = train_PINN(lobatto_100, lobatto_100_weights,epsilon)
        lobatto_100_assymetry_array[i] = get_assymetry(model)

    plt.scatter(epsilon_array, uniform_assymetry_array, color='red', label='Uniform')
    plt.scatter(epsilon_array, gauss_100_assymetry_array, color='blue', label='Gauss')
    plt.scatter(epsilon_array, lobatto_100_assymetry_array, color='green', label='Lobatto')
    plt.xscale('log')
    plt.legend()
    plt.title("l2 norm of assymetry for various collocations")
    plt.xlabel('epsilon')
    plt.ylabel('assymetry')
    plt.show()
