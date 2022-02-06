import numpy as np
from tqdm.auto import tqdm
from tensorflow.keras import datasets


def dbgprn(text):
    print("- [DBG]", text)
    return

def load_mnist():
    (x_train, y_train), (x_eval, y_eval) = datasets.mnist.load_data()
    x_train = x_train.astype(float) / 255
    x_eval  = x_eval.astype(float)  / 255
    x_valid = x_eval[:5000, :, :]
    y_valid = y_eval[:5000]
    x_test = x_eval[5000:, :, :]
    y_test = y_eval[5000:]
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_valid = np.expand_dims(x_valid, axis=3)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_cifar():
    # cp cifar-10-batches-py.tar.gz ~/.keras/datasets/
    (x_train, y_train), (x_eval, y_eval) = datasets.cifar10.load_data()
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

def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0,0)))
    return X_pad


class LayerReLU:
    def __init__(self):
        self.neurons = None
    def forward(self, x):
        # print(x.shape)
        self.neurons = x
        return np.maximum(x, 0)
    def backward(self, dout):
        # print(dout.shape)
        return dout * (self.neurons > 0)



class LayerConvolution:
    def __init__(self, num_filters, dim_filters, stride, padding):
        self.num_filters = num_filters
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride
        self.padding = padding
        self.neurons = None
        self.weights = None
        self.biases = None

    def forward(self, input_neurons):
        # print(input_neurons.shape)
        self.neurons = input_neurons
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]
        # dbgprn(input_neurons.shape)
        neurons_padded = zero_pad(input_neurons, self.padding)

        self.weights = np.random.randn(self.num_filters, *self.dim_filters, dim_input[-1]) * 0.00004
        self.biases = np.random.rand(self.num_filters,1)

        dim_output = (dim_input[:-1] - self.dim_filters + 2*self.padding) / self.stride + 1
        dim_output = dim_output.astype(int)
        output = np.zeros((num_neurons, *dim_output, self.num_filters))

        for n in range(num_neurons):
            for h in range(dim_output[0]):
                for w in range(dim_output[1]):
                    rows = slice(h * self.stride, h * self.stride + self.dim_filters[0])
                    cols = slice(w * self.stride, w * self.stride + self.dim_filters[1])
                    for c in range(self.num_filters):
                        slide = neurons_padded[n, rows, cols, :]
                        output[n, h, w, c] = np.sum(slide * self.weights[c,:,:,:]) + self.biases[c]
        return output

    def backward(self, din):
        num_neurons = self.neurons.shape[0]

        dout = np.zeros_like(self.neurons)
        neurons_padded = zero_pad(self.neurons, self.padding)
        dout_padded = zero_pad(dout, self.padding)

        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)

        dim_neurons = neurons_padded.shape[1:]

        for n in range(num_neurons):
            for h in range(din.shape[1]):
                for w in range(din.shape[2]):
                    rows = slice(h * self.stride, h * self.stride + self.dim_filters[0])
                    cols = slice(w * self.stride, w * self.stride + self.dim_filters[1])

                    for c in range(self.num_filters):
                        slide = neurons_padded[n, rows, cols, :]
                        dout_padded[n, rows, cols, :] += self.weights[c,:,:,:] * din[n,h,w,c]
                        dw[c,:,:,:] += slide * din[n,h,w,c]
                        db[c] += din[n,h,w,c]

            if self.padding != 0:
                dout[n,:,:,:] = dout_padded[n, self.padding:-self.padding, self.padding:-self.padding, :]
            self.weights = dw
            self.biases = db
        return dout



class LayerMaxPooling:
    def __init__(self, dim_filters, stride):
        self.dim_filters = dim_filters.astype(int)
        self.stride = stride
        self.neurons = None

    def forward(self, input_neurons):
        # print(input_neurons.shape)
        self.neurons = input_neurons
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]

        dim_output = (dim_input[:-1] - self.dim_filters) / self.stride + 1
        dim_output = dim_output.astype(int)

        output = np.zeros((num_neurons, *dim_output, dim_input[-1]))

        for n in range(num_neurons):
            for h in range(dim_output[0]):
                for w in range(dim_output[1]):
                    rows = slice(h * self.stride, h * self.stride + self.dim_filters[0])
                    cols = slice(w * self.stride, w * self.stride + self.dim_filters[1])
                    for c in range(dim_input[-1]):
                        slide = input_neurons[n, rows, cols, c]
                        output[n, h, w, c] = np.max(slide)
        return output

    def create_mask(self, x):
        return x == np.max(x)

    def backward(self, din):
        # print(din.shape)
        num_neurons = self.neurons.shape[0]
        dout = np.zeros_like(self.neurons)

        for n in range(num_neurons):
            for h in range(din.shape[1]):
                for w in range(din.shape[2]):
                    rows = slice(h * self.stride, h * self.stride + self.dim_filters[0])
                    cols = slice(w * self.stride, w * self.stride + self.dim_filters[1])
                    for c in range(din.shape[-1]):
                        slide = self.neurons[n, rows, cols, c]
                        mask = self.create_mask(slide)
                        dout[n, rows, cols, c] += din[n, h, w, c] * mask
        return dout


class LayerDense:
    def __init__(self, num_output):
        self.num_output = num_output
        self.weights = None
        self.biases = None
        self.neurons = None

    def forward(self, input_neurons):
        # print(input_neurons.shape)
        self.neurons = input_neurons
        num_neurons = input_neurons.shape[0]
        dim_input = input_neurons.shape[1:]

        self.weights = np.random.randn(np.prod(dim_input), self.num_output) * 0.00004
        self.biases = np.random.randn(self.num_output,)
        input_neurons = input_neurons.reshape(num_neurons, -1)
        output = np.dot(input_neurons, self.weights) + self.biases
        return output

    def backward(self, din):
        # print(din.shape)
        N = self.neurons.shape[0]
        x = self.neurons.reshape(N, -1)

        dx = np.dot(din, self.weights.T).reshape(self.neurons.shape)
        dw = np.dot(x.T, din)
        db = np.sum(din.T, axis=1)

        self.weights = dw
        self.biases = db

        return dx



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



def loss(y, yhat, epsilon=1e-8):
    N = yhat.shape[0]
    # yhat = np.clip(yhat, epsilon, 1. - epsilon)
    logprobs = -np.log(yhat[range(N),y])
    loss = np.sum(logprobs) / N
    return loss

def softmax(X):
    x = X - np.max(X, axis=1, keepdims=True)
    e = np.exp(x)
    p = e / np.sum(e, axis=1, keepdims=True)
    return p

def delta_cross_entropy(y, yhat):
    m = y.shape[0]
    grad = yhat.copy()
    grad[range(m),y] -= 1
    grad = grad/m
    return grad



class ConvNet:
    def __init__(self):
        self.layers = []
        return

    def addlayer(self, layer):
        self.layers.append(layer)
        return

    def forward(self, neurons):
        for layer in self.layers:
            neurons = layer.forward(neurons)
        return neurons

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def create_batches(self, x, y, batch_size=128):
        batches = []
        num_examples = x.shape[0]
        num_batches = num_examples // batch_size
        for i in range(num_batches):
            _x = x[i*batch_size:(i+1)*batch_size, :]
            _y = y[i*batch_size:(i+1)*batch_size]
            batches.append((_x,_y))
        if num_examples % batch_size != 0:
            _x = x[(num_batches-1)*batch_size:, :]
            _y = y[(num_batches-1)*batch_size:]
            batches.append((_x,_y))
        return batches

    def optimize(self, lr):
        for layer in self.layers:
            if isinstance(layer, LayerConvolution) or isinstance(layer, LayerDense):
                layer.weights -= layer.weights * lr
                layer.biases -= layer.biases * lr
        return

    def train(self, x, y, epochs=3, batch_size=32, lr=1e-4):
        batches = self.create_batches(x, y, batch_size)
        losses = []
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(batches, leave=False):
                x, y = batch
                out = self.forward(x)
                yhat = softmax(out)
                grad = delta_cross_entropy(y, yhat)
                dout = self.backward(grad)
                losses.append(loss(y, yhat))
                self.optimize(lr)

        return losses

    def eval(self, x, y):
        out = self.forward(x)
        yhat = softmax(out)

def model_mnist(N=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    if N is None:
        N = x_train.shape[0]

    cnn = ConvNet()
    cnn.addlayer(LayerConvolution(6, np.array((5,5)), 1, 2))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))
    cnn.addlayer(LayerConvolution(12, np.array((5,5)), 1, 0))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))
    cnn.addlayer(LayerConvolution(100, np.array((5,5)), 1, 0))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerDense(10))
    losses = cnn.train(x_train[:N], y_train[:N], epochs=1)

def model_cifar(N=None):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_cifar()
    if N is None:
        N = x_train.shape[0]

    cnn = ConvNet()
    cnn.addlayer(LayerConvolution(6, np.array((5,5)), 1, 2))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))
    cnn.addlayer(LayerConvolution(12, np.array((5,5)), 1, 0))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))
    cnn.addlayer(LayerConvolution(100, np.array((5,5)), 1, 0))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerDense(10))
    losses = cnn.train(x_train[:N], y_train[:N], epochs=1)

if __name__ == "__main__":
    # model_mnist(N=8)
    model_cifar(N=8)
