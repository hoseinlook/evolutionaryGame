import numpy as np
from random import random
import random

class NeuralNetwork():

    def __init__(self, layer_sizes, hidden_layer_weights=None, output_layer_weights=None):
        # TODO
        # layer_sizes example: [4, 10, 2]
        self._input_layer_size = layer_sizes[0]
        self._hidden_layer_size = layer_sizes[1]
        self._output_layer_size = layer_sizes[2]

        if output_layer_weights is None or hidden_layer_weights is None:
            self.hidden_layer_weights = np.random.randn(self._hidden_layer_size, self._input_layer_size)
            self.hidden_layer_B = np.zeros((self._hidden_layer_size, 1))

            self.output_layer_weights = np.random.randn(self._output_layer_size, self._hidden_layer_size)
            self.output_layer_B = np.zeros((self._output_layer_size, 1))
        # print('CREATED ')
        else:
            self.hidden_layer_weights = hidden_layer_weights
            self.output_layer_weights = output_layer_weights

    def set_weights(self, hidden_layer_weights, output_layer_weights):
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        pass

    def _activation(self, input):
        # TODO
        z = 1 / (1 + np.exp(-input))
        return z

    def forward(self, input):
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])

        A1 = self.hidden_layer_weights @ input + self.hidden_layer_B
        Z1 = self._activation(A1)
        A2 = self.output_layer_weights @ Z1 + self.output_layer_B

        return self._activation(A2)

        pass

    def flat_weights(self):
        flat_hidden = self.hidden_layer_weights.reshape(1, self._hidden_layer_size * self._input_layer_size)[0]
        flat_out = self.output_layer_weights.reshape(1, self._hidden_layer_size * self._output_layer_size)[0]

        return list(flat_out), list(flat_hidden)

    def reshape_weights_from_flat(self, hidden_weights, out_weights):
        self.hidden_layer_weights = np.array([hidden_weights]).reshape(self._hidden_layer_size, self._input_layer_size)
        self.output_layer_weights = np.array([out_weights]).reshape(self._output_layer_size, self._hidden_layer_size)

    def cross_over(self,other_nn,NO_points=2):
        other_nn:NeuralNetwork

        random.choices()


    def mutation_weights_with_a_probability(self, probability=0.6):

        # o_weights_list, h_weights_list = self.flat_weights()
        # for i in range(len(o_weights_list)):
        #     o_weights_list[i] = np.random.randn() if random() < probability else o_weights_list[i]
        # for i in range(len(h_weights_list)):
        #     h_weights_list[i] = np.random.randn() if random() < probability else h_weights_list[i]
        #
        # self.reshape_weights_from_flat(h_weights_list,o_weights_list)

        if probability > random.random():
            self.hidden_layer_weights += np.random.normal(0, 0.5, self.hidden_layer_weights.shape)
            self.hidden_layer_B += np.random.normal(0, 0.3, self.hidden_layer_B.shape)

            self.output_layer_weights += np.random.normal(0, 0.5, self.output_layer_weights.shape)
            self.output_layer_B += np.random.normal(0, 0.3, self.output_layer_B.shape)


if __name__ == '__main__':
    # print(NeuralNetwork([1,1,1]).activation(np.array([[1],[1]])))
    # np.random.seed(1)
    # print(np.random.randn(1, 10))

    x = NeuralNetwork([10, 10, 1])
    inputtt = np.random.randn(10, 1)
    print(x.forward(inputtt))
    print(x.hidden_layer_weights)
    print('************************************************************')
    print('************************************************************')
    print('************************************************************')
    # print(x.hidden_layer_weights.reshape(1, 100).reshape(10, 10))
    print(x.mutation_weights_with_a_probability(0.9))
    print(x.hidden_layer_weights)
    print("RESSS ", x.forward(inputtt))
    print(x.mutation_weights_with_a_probability(0.9))
    print(x.hidden_layer_weights)
    print("RESSS ", x.forward(inputtt))
    print(x.mutation_weights_with_a_probability(0.9))
    print(x.hidden_layer_weights)
    print("RESSS ", x.forward(inputtt))
    print(x.mutation_weights_with_a_probability(0.9))
    print(x.hidden_layer_weights)
    print("RESSS ", x.forward(inputtt))
    # print(x.hidden_layer_weights.shape)
    # print((x.reshape_weights()))
