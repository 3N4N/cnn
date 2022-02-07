import numpy as np
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tensorflow.keras import datasets
from sklearn import metrics
import pickle


tqdmcols = 80
np.random.seed(111)

def dbgprn(*args):
    print("- [DBG]", *args)
    return

def load_mnist():
    (x_train, y_train), (x_eval, y_eval) = datasets.mnist.load_data()
    x_train = x_train.astype(float) / 255
    x_eval  = x_eval.astype(float)  / 255
    x_valid = x_eval[:5000, :, :]
    y_valid = y_eval[:5000]
    x_test = x_eval[5000:, :, :]
    y_test = y_eval[5000:]
    # x_train = np.expand_dims(x_train, axis=3)
    # x_test = np.expand_dims(x_test, axis=3)
    # x_valid = np.expand_dims(x_valid, axis=3)
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    x_valid = np.expand_dims(x_valid, axis=1)
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_cifar():
    # cp cifar-10-batches-py.tar.gz ~/.keras/datasets/
    (x_train, y_train), (x_eval, y_eval) = datasets.cifar10.load_data()
    x_train = np.moveaxis(x_train, 1, 3)
    x_eval = np.moveaxis(x_eval, 1, 3)
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
    # X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0,0)))
    X_pad = np.pad(X, ((0, 0), (0,0), (pad, pad), (pad, pad)))
    return X_pad


class LayerReLU:
    def __init__(self):
        self.neurons = None
    def forward(self, x):
        self.neurons = x
        return np.maximum(x, 0)
    def backward(self, dout):
        return dout * (self.neurons > 0)



class LayerConvolution:
    def __init__(self, nfilters, dfilters, stride, padding, weight_scale=0.01, lr=0.02):
        self.nfilters = nfilters
        self.dfilters = dfilters.astype(int)
        self.stride = stride
        self.padding = padding
        self.neurons = None
        self.weights = None
        self.biases = None
        self.ws = weight_scale
        self.lr = lr

    def forward(self, x):
        N, C, H, W = x.shape
        stride = self.stride
        padding =  self.padding

        if self.weights is None:
            # dbgprn('conv')
            w = np.random.normal(0.0, self.ws, (self.nfilters, C, *self.dfilters)) #* 1e-5
            b = np.zeros((self.nfilters,))
        else:
            w = self.weights
            b = self.biases

        F, _, FH, FW = w.shape
        oH = int((H - FH + 2*padding)/stride + 1)
        oW = int((W - FW + 2*padding)/stride + 1)

        out = np.zeros((N, F, oH, oW))

        padded_x = zero_pad(x, padding)
        _, _, padded_H, padded_W = padded_x.shape

        w_row = w.reshape(F, C * FH * FW)
        x_col = np.zeros((C * FH * FW, oH * oW))
        for i in range(N):
            c = 0
            for j in range(0, padded_H - FH + 1, stride):
                for k in range(0, padded_W - FW + 1, stride):
                    x_col[:, c] = padded_x[i, :, j:j+FH, k:k+FW].reshape(C * FH * FW)
                    c += 1
            out[i,:,:,:] = (np.dot(w_row, x_col) + b.reshape(-1, 1)).reshape(F, oH, oW)

        self.weights = w
        self.biases = b
        self.neurons = x

        return out

    def backward(self, dout):
        w = self.weights
        b = self.biases
        x = self.neurons
        stride = self.stride
        padding =  self.padding

        N, C, H, W = x.shape
        F, _, FH, FW = w.shape

        _,_, oH, oW = dout.shape

        padded_x = zero_pad(x, padding)
        _, _, padded_H, padded_W = padded_x.shape

        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        w_row = w.reshape(F, C * FH * FW)
        x_col = np.zeros((C * FH * FW, oH * oW))

        for i in range(N):
            curr_dout = dout[i, :, :, :].reshape(F, oH * oW)
            curr_out = np.dot(w_row.T, curr_dout)
            curr_dpx = np.zeros(padded_x.shape[1:])
            c = 0
            for j in range(0, padded_H - FH + 1, stride):
                for k in range(0, padded_W - FW + 1, stride):
                    curr_dpx[:, j:j+FH, k:k+FW] += curr_out[:, c].reshape(C, FH, FW)
                    x_col[:, c] = padded_x[i, :, j:j+FH, k:k+FW].reshape(C * FH * FW)
                    c += 1
            if padding != 0:
                dx[i] = curr_dpx[:, padding:-padding, padding:-padding]
            else:
                dx[i] = curr_dpx
            dw += np.dot(curr_dout, x_col.T).reshape(F, C, FH, FW)
            db += np.sum(curr_dout, axis=1)

            self.weights -= dw * self.lr
            self.biases -= db * self.lr

        return dx



class LayerMaxPooling:
    def __init__(self, dfilters, stride):
        self.dfilters = dfilters.astype(int)
        self.stride = stride
        self.neurons = None

    def forward(self, x):
        stride = self.stride
        N, C, H, W = x.shape
        pool_height, pool_width = self.dfilters

        out_H = 1 + (H - pool_height) // stride
        out_W = 1 + (W - pool_width) // stride

        out = np.zeros((N, C, out_H, out_W))

        for i in range(N):
            curr_out = np.zeros((C, out_H * out_W))
            c = 0
            for j in range(0, H - pool_height + 1, stride):
                for k in range(0, W - pool_width + 1, stride):
                    curr_region = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height * pool_width)
                    curr_max_pool = np.max(curr_region, axis=1)
                    curr_out[:, c] = curr_max_pool
                    c += 1
            out[i, :, :, :] = curr_out.reshape(C, out_H, out_W)

        self.neurons = x
        return out

    def backward(self, dout):
        x = self.neurons
        pool_height, pool_width = self.dfilters
        stride = self.stride

        N, C, H, W = x.shape
        _, _, out_H, out_W = dout.shape

        dx = np.zeros_like(x)

        for i in range(N):
            curr_dout = dout[i, :].reshape(C, out_H * out_W)
            c = 0
            for j in range(0, H - pool_height + 1, stride):
                for k in range(0, W - pool_width + 1, stride):
                    curr_region = x[i, :, j:j+pool_height, k:k+pool_width].reshape(C, pool_height * pool_width)
                    curr_max_idx = np.argmax(curr_region, axis=1)
                    curr_dout_region = curr_dout[:, c]
                    curr_dpooling = np.zeros_like(curr_region)
                    curr_dpooling[np.arange(C), curr_max_idx] = curr_dout_region
                    dx[i, :, j:j+pool_height, k:k+pool_height] = curr_dpooling.reshape(C, pool_height, pool_width)
                    c += 1

        return dx


class LayerDense:
    def __init__(self, nout, lr=0.02):
        self.nout = nout
        self.weights = None
        self.biases = None
        self.neurons = None
        self.lr = lr
        self.ws=0.01

    def forward(self, x):
        N, C, H, W = x.shape

        if self.weights is None:
            # dbgprn('dense')
            w = np.random.randn(C*H*W, self.nout) * np.sqrt(2.0/(H*W))
            b = np.zeros((self.nout,))
        else:
            w = self.weights
            b = self.biases
        x_new = x.reshape(N, -1)
        out = np.dot(x_new, w) + b

        self.neurons = x
        self.weights = w
        self.biases = b
        return out

    def backward(self, dout):
        x, w, b = self.neurons, self.weights, self.biases
        N = x.shape[0]
        x_new = x.reshape(N, -1)

        dx = np.dot(dout, w.T).reshape(x.shape)
        dw = np.dot(x_new.T, dout)
        db = np.sum(dout.T, axis=1)

        self.weights -= dw * self.lr
        self.biases -= db * self.lr
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



def softmax(X):
    x = X - np.max(X, axis=1, keepdims=True)
    e = np.exp(x)
    p = e / np.sum(e, axis=1, keepdims=True)
    return p

def calculate_loss(y, prob, epsilon=1e-8):
    N = y.shape[0]
    prob = np.clip(prob, epsilon, 1. - epsilon)
    logprob = -np.log(prob[np.arange(N),y])
    loss = np.sum(logprob) / N
    return loss

def delta_cross_entropy(y, prob):
    m = y.shape[0]
    grad = prob.copy()
    grad[np.arange(m),y] -= 1
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
        i = 0
        for layer in self.layers:
            # dbgprn(i, neurons.shape)
            neurons = layer.forward(neurons)
        return neurons

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def create_batches(self, x, y, batch_size):
        batches = []
        num_examples = x.shape[0]
        num_batches = num_examples // batch_size
        # dbgprn(x.shape, batch_size, num_batches, num_examples)
        for i in range(num_batches):
            _x = x[i*batch_size:(i+1)*batch_size, :]
            _y = y[i*batch_size:(i+1)*batch_size]
            batches.append((_x,_y))
        if num_examples % batch_size != 0:
            _x = x[(num_batches-1)*batch_size:, :]
            _y = y[(num_batches-1)*batch_size:]
            batches.append((_x,_y))
        return batches

    def predict(self, x):
        out = self.forward(x)
        prob = softmax(out)
        yhat = np.argmax(prob, axis=1)
        return yhat, prob

    def train(self, x_train, y_train, x_valid, y_valid, epochs=3, batch_size=128, save_cache=True):
        # dbgprn(x_train.shape, batch_size)
        batches = self.create_batches(x_train, y_train, batch_size)
        losses = []
        validation_scores = []

        for epoch in tqdm(range(epochs), ncols=tqdmcols):
            for batch in tqdm(batches, leave=False, ncols=tqdmcols):
                x_batch, y_batch = batch
                out = self.forward(x_batch)
                prob = softmax(out)
                # yhat = np.argmax(prob, axis=1)
                # print(np.unique(yhat, return_counts=True))
                grad = delta_cross_entropy(y_batch, prob)
                dout = self.backward(grad)
                losses.append(calculate_loss(y_batch, prob))
                if save_cache:
                    with open('layers.pkl', 'wb') as f:
                        pickle.dump(self.layers, f)
                    with open('losses.pkl', 'wb') as f:
                        pickle.dump(losses, f)
            loss, accu, f1sc = self.validate(x_valid, y_valid)
            validation_scores.append({"Epoch": epoch+1, "Loss": loss, "Accuracy": accu, "F1-score": f1sc})
            if save_cache:
                with open('scores.pkl', 'wb') as f:
                    pickle.dump(validation_scores, f)
        return losses, validation_scores

    def validate(self, x, y):
        yhat, prob = self.predict(x)
        loss = calculate_loss(y, prob)
        accu = metrics.accuracy_score(y, yhat)
        f1sc = metrics.f1_score(y, yhat, average='macro')
        return loss, accu, f1sc



def main(x_train, y_train, x_valid, y_valid, x_test, y_test, N, M, T,
         lr=1e-3, epochs=1, batch_size=128, save_cache=True):
    cnn = ConvNet()
    cnn.addlayer(LayerConvolution(6, np.array((5,5)), 1, 2, lr=lr)) # 28x28x1
    cnn.addlayer(LayerReLU())                                       # 28x28x6
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))               # 28x28x6
    cnn.addlayer(LayerConvolution(12, np.array((5,5)), 1, 0, lr=lr)) #14x14x6
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerMaxPooling(np.array((2,2)), 2))
    cnn.addlayer(LayerConvolution(100, np.array((5,5)), 1, 0))
    cnn.addlayer(LayerReLU())
    cnn.addlayer(LayerDense(10, lr=lr))
    losses, validation_scores = cnn.train(x_train[:N], y_train[:N], x_train[:M], y_valid[:M],
                                          epochs, batch_size=batch_size, save_cache=save_cache)
    test_scores = cnn.validate(x_test[:T], y_test[:T])
    return losses, validation_scores, test_scores

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist()
    # data_cifar = load_cifar()
    # x_train, y_train, x_valid, y_valid, x_test, y_test = data_cifar

    N = 1; M = 8; T = 8; epochs = 1
    N = 160; M = 8; T = 8; epochs = 5
    N = x_train.shape[0]; M = x_valid.shape[0]; T = x_test.shape[0]; epochs=1; batch_size=128

    losses, validation_scores, test_scores = main(x_train, y_train, x_valid, y_valid, x_test, y_test,
                                                  N, M, T, 1e-2, epochs, batch_size, False)
    print("Losses: {}\nValidation results: {}".format(losses, validation_scores))
    print("Test results: {}".format(test_scores))
