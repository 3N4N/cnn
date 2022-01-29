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

    def convolve(self, input_neurons):
        self.dim_output = (self.dim_input[:-1] - self.dim_filters + 2*self.padding) / self.stride + 1
        self.dim_output = self.dim_output.astype(int)
        self.output = np.zeros((*self.dim_output, self.num_filters))
        del self.dim_output

    def convolve(self, input_neurons):
        self.output = self.output.reshape((np.prod(self.dim_output), self.num_filters))

        for j in range(self.num_filters):
            slide = 0
            row = 0

            for i in range(self.output.shape[0]):
                self.output[i][j] = np.sum(
                    input_neurons[row:self.dim_filters[0]+row,
                                  slide:self.dim_filters[1]+slide,
                                  :] * self.weights[j]) + self.biases[j]
                slide += self.stride

                if self.dim_filters[0] + slide - self.stride >= self.width_in:
                    slide = 0
                    row += self.stride

        self.output = self.output.reshape((*self.output.shape, self.num_filters))






if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    dim_input = np.append(np.array(x_train[0].shape), 1)
    layer1 = LayerConvolution(dim_input, 6, np.array((5,5)), 1, 2)
