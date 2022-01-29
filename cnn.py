from tensorflow.keras import datasets
import numpy as np


def load_mnist():
    (x_train, y_train), (x_eval, y_eval) = datasets.mnist.load_data()
    x_valid = x_eval[:5000, :, :]
    y_valid = y_eval[:5000]
    x_test = x_eval[5000:, :, :]
    y_test = y_eval[5000:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

class LayerConvolution:
    def __init__(self, dim_input, num_filters, dim_filters, stride, padding):
        self.dim_input = dim_input.astype(int)
        self.num_filters = num_filters
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.num_filters, self.dim_input[2], *self.dim_filters)
        self.biases = np.random.rand(self.num_filters,1)

        self.dim_output = (self.dim_input[:-1] - self.dim_filters + 2*self.padding) / self.stride + 1
        self.dim_output = self.dim_output.astype(int)
        self.output = np.zeros((*self.dim_output, self.num_filters))

    def convolve(self, input_neurons):
        self.output = self.output.reshape((np.prod(self.dim_output), self.num_filters))
        for j in range(self.num_filters):
            col = 0
            row = 0
            for i in range(self.output.shape[0]):
                self.output[i][j] = np.sum(
                    input_neurons[row:self.dim_filters[0]+row,
                                  col:self.dim_filters[1]+col,
                                  :] * self.weights[j]) + self.biases[j]
                col += self.stride
                if col + self.dim_filters[0] > self.dim_input[1]:
                    col = 0
                    row += self.stride
        self.output = self.output.reshape((*self.dim_output, self.num_filters))


class LayerMaxPooling:
    def __init__(self, dim_input, dim_filters, stride):
        self.dim_input = dim_input.astype(int)
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride

        self.dim_output = (self.dim_input[:-1] - self.dim_filters) / self.stride + 1
        self.dim_output = self.dim_output.astype(int)
        self.output = np.zeros((*self.dim_output, self.dim_input[:-1]))

    def pool(self, input_image):
        self.output = self.output.reshape((np.prod(self.dim_output), self.dim_input[:-1]))
        for j in range(self.dim_input[2]):
            row = 0
            col = 0
            for i in range(self.output.shape[0]):
                slide = input_image[row:self.dim_filters[0]+row,
                                     col:self.dim_filters[0]+col][j]
                self.output[i][j] = np.amax(slide)
                col += self.dim_filters[1]
                if col + dim_filters[1] > self.dim_input[1]:
                    col = 0
                    row += self.dim_filters[0]
        self.output = self.output.reshape((*self.dim_output, self.dim_input[:-1]))


class LayerFullyConnected:
    def __init__(self, dim_input, dim_output):
        self.dim_input = dim_input.astype(int)
        self.dim_output = dim_output.astype(int)

        self.weights = np.random.randn(self.num_filters, self.dim_input[2], *self.dim_filters)
        self.biases = np.random.rand(self.num_filters,1)

        self.weights = np.random.randn(self.dim_output, *self.dim_input)
        self.biases = np.random.randn(*self.dim_output,1)

    def forward(self, input_data):
        self.weights = self.weights.reshape((self.dim_output, np.prod(self.dim_input)))
        self.input_data = input_data.reshape((np.prod(dim_input), 1))
        self.output = np.dot(self.weights, input_data) + self.biases




if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    dim_input = np.append(np.array(x_train[0].shape), 1)
    layer1 = LayerConvolution(dim_input, 6, np.array((5,5)), 1, 2)
