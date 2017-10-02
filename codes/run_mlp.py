from network import Network
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss  # , SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
from logpath import logpath
import csv
from datetime import datetime


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 81, 0.01))
model.add(Sigmoid('sigmoid'))
model.add(Linear('fc2', 81, 10, 0.01))
model.add(Sigmoid('sigmoid'))

loss = EuclideanLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.5,
    'batch_size': 100,
    'max_epoch': 300,
    'disp_freq': 50,
    'test_epoch': 5
}

start = datetime.now()
display_start = str(start).split(' ')[1][:-3]
log_list = []
for epoch in range(config['max_epoch']):
    loss_value = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
    log_list.append({'epoch': epoch, 'loss': loss_value})

    if epoch == config['max_epoch'] - 1:
        acc_value = test_net(model, loss, test_data, test_label, config['batch_size'])

now = datetime.now()
display_now = str(now).split(' ')[1][:-3]
logfile = file(logpath, 'wb')
writer = csv.writer(logfile)
data = [(log['epoch'], log['loss']) for log in log_list]
writer.writerows(data)
data = [(display_start, ), (display_now, ), (acc_value, )]
writer.writerows(data)
logfile.close()
