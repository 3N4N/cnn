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





class LayerConvolution:
    def __init__(self, num_filters, dim_filters, stride, padding):
        self.num_filters = num_filters
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride
        self.padding = padding


    def forward(self, input_neurons):
        print(input_neurons.shape)
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]
        input_neurons = zero_pad(input_neurons, self.padding)

        weights = np.random.randn(self.num_filters, dim_input[-1], *self.dim_filters)
        biases = np.random.rand(self.num_filters,1)

        dim_output = (dim_input[:-1] - self.dim_filters + 2*self.padding) / self.stride + 1
        dim_output = dim_output.astype(int)
        output = np.zeros((num_neurons, *dim_output, self.num_filters))
        output = output.reshape((num_neurons, np.prod(dim_output), self.num_filters))

        for k in range(num_neurons):
            for j in range(self.num_filters):
                col = 0
                row = 0
                for i in range(np.prod(dim_output)):
                    output[k][i][j] = np.sum(
                        np.multiply(input_neurons[k,
                                                  row:self.dim_filters[0]+row,
                                                  col:self.dim_filters[1]+col,
                                                  :], weights[j])) + biases[j]
                    col += self.stride
                    if col + self.dim_filters[0] > dim_input[1]:
                        col = 0
                        row += self.stride
        output = output.reshape((num_neurons, *dim_output, self.num_filters))
        return output


class LayerMaxPooling:
    def __init__(self, dim_filters, stride):
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride

    def forward(self, input_neurons):
        print(input_neurons.shape)
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]

        dim_output = (dim_input[:-1] - self.dim_filters) / self.stride + 1
        dim_output = dim_output.astype(int)

        output = np.zeros((num_neurons, *dim_output, dim_input[-1]))
        output = output.reshape((num_neurons, np.prod(dim_output), dim_input[-1]))

        for k in range(num_neurons):
            for j in range(dim_input[-1]):
                row = 0
                col = 0
                for i in range(np.prod(dim_output)):
                    slide = input_neurons[k, row:self.dim_filters[0]+row,
                                          col:self.dim_filters[0]+col, j]
                    output[k][i][j] = np.amax(slide)
                    col += self.dim_filters[1]
                    if col + self.dim_filters[1] > dim_input[1]:
                        col = 0
                        row += self.dim_filters[0]
        output = output.reshape((num_neurons, *dim_output, dim_input[-1]))
        return output


class LayerFullyConnected:
    def __init__(self, num_output):
        self.num_output = num_output

    def forward(self, input_neurons):
        print(input_neurons.shape)
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]

        weights = np.random.randn(np.prod(dim_input), self.num_output)
        biases = np.random.randn(self.num_output,)
        input_neurons = input_neurons.reshape(num_neurons, -1)
        print(input_neurons.shape)
        output = np.dot(input_neurons, weights) + biases
        return output


class LayerFlatten:
    def __init__(self):
        self.shape = ()
    def forward(self, input_data):
        shape = input_data.shape
        output = np.ravel(input_data).reshape(shape[0], -1)
        self.shape = shape
        return output

class Softmax:
    def __init__(self):
        pass
    def forward(self, input_data):
        x = input_data - np.max(input_data, axis=1, keepdims=True)
        e = np.exp(x)
        p = e / np.sum(e, axis=1, keepdims=True)
        return p



if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    dim_image = tuple(np.append(np.array(x_train[0].shape), 1))
    z = np.expand_dims(x_train, 3)
    l1 = LayerConvolution(6, np.array((5,5)), 1, 2)
    o1 = l1.forward(z[:32])
    l2 = LayerMaxPooling(np.array((2,2)), 2)
    o2 = l2.forward(o1)
    l3 = LayerFullyConnected(10)
    o3 = l3.forward(o2)
    l4 = LayerFlatten()
    o4 = l4.forward(o3)
    l5 = Softmax()
    o5 = l5.forward(o4)
