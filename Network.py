import Load_Data as data
import theano
import numpy as np
from theano import tensor as T
import os

def weight_init(shape, choose):
    path = ''
    if choose == 0:
        path = r'./data/input_hidden_weight.txt'
    else:
        path = r'./data/hidden_output_weight.txt'
    if os.path.exists(path):
        w = np.loadtxt(path)
    else:
        print 'file do not exist'
        w = np.random.uniform(-0.1, 0.1, shape)
    return w

def create_batch_target(y):
    target = []
    for t in y:
        t = int(t)
        temp = [0  for x in range(10)]
        temp[t] = 1
        target.append(temp)
    return target

def save_weight():
    np.savetxt(r'./data/input_hidden_weight.txt',np.array(i_h_weight.get_value()))
    np.savetxt(r'./data/hidden_output_weight.txt', np.array(h_o_weight.get_value()))


input_size = 784
sample_num = 50000
step = 0.01
hidden_num = 30
output_num = 10
batch_size = 100

input = T.matrix('hidden_input')
h_output = T.matrix('hidden_output')
o_input = T.matrix('outLayer_input')
o_output = T.matrix('outLayer_output')
target = T.matrix('target')
loss = T.scalar('loss')

i_h_shape = (hidden_num, input_size)
i_h_weight = theano.shared(weight_init(i_h_shape, 0), 'input2hidden_weight')

h_o_shape = (output_num, hidden_num)
h_o_weight = theano.shared(weight_init(h_o_shape, 1), 'hidden2output_weight')

h_sum = T.dot(input, T.transpose(i_h_weight))
h_output = T.nnet.sigmoid(h_sum)

o_input = h_output
o_sum = T.dot(o_input, T.transpose(h_o_weight))
o_output = T.nnet.softmax(o_sum)

loss = T.sum((o_output-target)**2) / output_num #->change to batch_loss

grad_h2o = T.grad(loss, h_o_weight) #->change to batch grad_h2o
grad_i2h = T.grad(loss, i_h_weight) #->change to batch grad_i2h

#prediction function
prediction = theano.function([input], o_output)

train = theano.function([input, target], loss,
                        updates = [(i_h_weight, i_h_weight - step*grad_i2h),
                                   (h_o_weight, h_o_weight - step*grad_h2o)],
                        on_unused_input='ignore')

sum_loss = 0
for i in range(0,500):
    print '%d train'%(i)

    right = 0
    for ip, t in zip(data.test_setX, data.test_setY):
        t = int(t)
        op = np.array(prediction([ip]))
        result = np.argmax(op[0])
        if (result == t):
            right += 1
    print 'right rate %s'%(float(right)/100),'%'

    for j in range(sample_num/batch_size):
        sum_loss = train(data.train_setX[batch_size*j:batch_size*(j+1)], create_batch_target(data.train_setY[batch_size*j:batch_size*(j+1)]))+sum_loss
    print 'sum_loss = ',sum_loss
    sum_loss = 0
    save_weight()
