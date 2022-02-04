import numpy as np

from tensorflow.keras import datasets


def load_mnist():
    (x_train, y_train), (x_eval, y_eval) = datasets.mnist.load_data()
    x_train = x_train.astype(float) / 255
    x_eval  = x_eval.astype(float)  / 255
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
        self.weights = None
        self.biases = None
        self.neurons = None

    def forward(self, input_neurons):
        self.neurons = input_neurons
        print(input_neurons.shape)
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]

        self.weights = np.random.randn(np.prod(dim_input), self.num_output)
        self.biases = np.random.randn(self.num_output,)
        input_neurons = input_neurons.reshape(num_neurons, -1)
        print(input_neurons.shape)
        output = np.dot(input_neurons, self.weights) + self.biases
        return output

    def backward(self, input_data):
        N = self.neurons.shape[0]
        x = self.neurons.reshape(N, -1)

        dx = np.dot(input_data, self.weights.T).reshape(self.neurons.shape)
        dw = np.dot(x.T, input_data)
        db = np.sum(input_data.T, axis=1)

        return dx, dw, db



class LayerFlatten:
    def __init__(self):
        self.shape = ()
    def forward(self, input_data):
        shape = input_data.shape
        output = np.ravel(input_data).reshape(shape[0], -1)
        self.shape = shape
        return output
    def backward(self, input_data):
        return input_data.reshape(self.shape)


class Softmax:
    def __init__(self):
        pass
    def forward(self, input_data):
        x = input_data - np.max(input_data, axis=1, keepdims=True)
        e = np.exp(x)
        p = e / np.sum(e, axis=1, keepdims=True)
        return p
    def loss(self, probs, y, epsilon=1e-8):
        N = probs.shape[0]
        probs = np.clip(probs, epsilon, 1. - epsilon)
        correct_logprobs = -np.log(probs[range(N),y])
        loss = np.sum(correct_logprobs) / N
        return loss
    def backward(self, probs, y):
        N = probs.shape[0]
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        return dscores



if __name__ == "__main__":
    N = 32
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    dim_image = tuple(np.append(np.array(x_train[0].shape), 1))
    z = np.expand_dims(x_train, 3)
    l1 = LayerConvolution(6, np.array((5,5)), 1, 2)
    f1 = l1.forward(z[:N])
    l2 = LayerMaxPooling(np.array((2,2)), 2)
    f2 = l2.forward(f1)
    l3 = LayerFullyConnected(10)
    f3 = l3.forward(f2)
    l4 = LayerFlatten()
    f4 = l4.forward(f3)
    l5 = Softmax()
    f5 = l5.forward(f4)
    loss = l5.loss(f5, y_train[N])
    b5 = l5.backward(f5, y_train[N])
    b4 = l4.backward(b5)
    b3, w3, a3 = l3.backward(b4)
