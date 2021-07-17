import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):
        # TODO
        # layer_sizes example: [4, 10, 2]
        self._input_layer_size = layer_sizes[0]
        self._hidden_layer_size = layer_sizes[1]
        self._output_layer_size = layer_sizes[2]

        self.hidden_layer_weights = np.random.randn(self._hidden_layer_size, self._input_layer_size)
        self.output_layer_weights = np.random.randn(self._output_layer_size, self._hidden_layer_size)

        # self.hidden_layer_weights=hidden_layer_weights
        # self.output_layer_weights=output_layer_weights
        pass

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

        A1 = self.hidden_layer_weights @ input
        Z1 = self._activation(A1)
        A2 = self.output_layer_weights @ Z1

        return self._activation(A2)

        pass

    def reshape_weights(self):
        flat_hidden = self.hidden_layer_weights.reshape(1, self._hidden_layer_size * self._input_layer_size)[0]
        flat_out = self.output_layer_weights.reshape(1, self._hidden_layer_size * self._output_layer_size)[0]

        return list(flat_out), list(flat_hidden)


if __name__ == '__main__':
    # print(NeuralNetwork([1,1,1]).activation(np.array([[1],[1]])))
    # np.random.seed(1)
    # print(np.random.randn(1, 10))

    x = NeuralNetwork([10, 10, 1])
    print(x.forward(np.random.randn(10, 1)))
    print(x.hidden_layer_weights)
    print('************************************************************')
    print('************************************************************')
    print('************************************************************')
    print(x.hidden_layer_weights.reshape(1, 100).reshape(10, 10))
    # print(x.hidden_layer_weights.shape)
    # print((x.reshape_weights()))
