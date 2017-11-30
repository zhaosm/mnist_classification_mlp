from utils import onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq, current_iter_count):

    iter_counter = 0
    loss_value = 0
    loss_values = []

    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value += loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        if iter_counter % disp_freq == 0:
            loss_values.append({'iter': current_iter_count + iter_counter, 'loss': loss_value / disp_freq})
            loss_value = 0
    current_iter_count += iter_counter
    return current_iter_count, loss_values


def test_net(model, loss, inputs, labels, batch_size):

    acc_value = 0.0
    count = 0
    for input, label in data_iterator(inputs, labels, batch_size, shuffle=False):
        output = model.forward(input)
        acc_value += calculate_acc(output, label)
        count += 1
    return acc_value / count
