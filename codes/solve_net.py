from utils import onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):

    iter_counter = 0

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)
    return loss_value


def test_net(model, loss, inputs, labels, batch_size):

    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        output = model.forward(input)
        acc_value = calculate_acc(output, label)
    return acc_value
