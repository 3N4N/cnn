import numpy as np

from tensorflow.keras import datasets


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

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0,0)))
    return X_pad




class LayerMaxPooling:
    def __init__(self, dim_input, dim_filters, stride):
        self.dim_input = dim_input.astype(int)
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride

        self.dim_output = (self.dim_input[:-1] - self.dim_filters) / self.stride + 1
        self.dim_output = self.dim_output.astype(int)
        output = np.zeros((*self.dim_output, self.dim_input[:-1]))

    def forward(self, input_image):
        output = output.reshape((np.prod(self.dim_output), self.dim_input[:-1]))
        for j in range(self.dim_input[2]):
            row = 0
            col = 0
            for i in range(output.shape[0]):
                slide = input_image[row:self.dim_filters[0]+row,
                                     col:self.dim_filters[0]+col][j]
                output[i][j] = np.amax(slide)
                col += self.dim_filters[1]
                if col + dim_filters[1] > self.dim_input[1]:
                    col = 0
                    row += self.dim_filters[0]
        output = output.reshape((*self.dim_output, self.dim_input[:-1]))


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
        output = np.dot(self.weights, input_data) + self.biases

class LayerFlattening:
    def __init__(self, dim_input):
        self.dim_input = dim_input.astype(int)

    def forward(self, input_data):
        output = np.ravel(input_data)


class LayerConvolution:
    def __init__(self, dim_input, num_filters, dim_filters, stride, padding):
        self.dim_input = dim_input.astype(int)
        self.num_filters = num_filters
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride
        self.padding = padding

        self.weights = np.random.randn(self.num_filters, self.dim_input[2], *self.dim_filters)
        self.biases = np.random.rand(self.num_filters,1)

        self.dim_output = (self.dim_input[1:-1] - self.dim_filters + 2*self.padding) / self.stride + 1
        self.dim_output = self.dim_output.astype(int)

    def forward(self, input_neurons):
        input_neurons = zero_pad(input_neurons, self.padding)
        print(input_neurons.shape)
        num_images = input_neurons.shape[0]
        output = np.zeros((num_images, *self.dim_output, self.num_filters))
        output = output.reshape((num_images, np.prod(self.dim_output), self.num_filters))
        for k in range(num_images):
            for j in range(self.num_filters):
                col = 0
                row = 0
                for i in range(np.prod(self.dim_output)):
                    output[k][i][j] = np.sum(
                        np.multiply(input_neurons[k,
                                                  row:self.dim_filters[0]+row,
                                                  col:self.dim_filters[1]+col,
                                                  :], self.weights[j])) + self.biases[j]
                    col += self.stride
                    if col + self.dim_filters[0] > self.dim_input[1]:
                        col = 0
                        row += self.stride
        output = output.reshape((num_images, *self.dim_output, self.num_filters))
        return output



if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    dim_input = np.append(np.array(x_train[0].shape), 1)
    z = np.expand_dims(x_train, 3)
    layer1 = LayerConvolution(dim_input, 6, np.array((5,5)), 1, 2)
    o = layer1.forward(z[:1000])
