##Neural Network, TensorFlow
#only run on Python3.5 under Program Files/Python35

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



##5 Learn a simple regression

#create data
x_data = np.random.rand(100).astype(np.float32) #tf use float32
y_data = x_data*0.1+0.3

###create tensorflow structure start###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) #could be matrix, here [1] value start with rand value betw -1,1
biases = tf.Variable(tf.zeros([1])) #start [1] value from 0

y = Weights*x_data+biases

loss = tf.reduce_mean(tf.square(y-y_data)) #different expect to origin
optimizer = tf.train.GradientDescentOptimizer(0.5) #construct a optimizer, use GD with learn rate=0.5
train = optimizer.minimize(loss) #minized loss

init = tf.global_variables_initializer() #not yet given
#change from initialize_all_variables() since 3/2/2017
###create tensorflow structure end###

#given session, start running
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step %20 ==0:
        print(step, sess.run(Weights),sess.run(biases))



##6 Session control
#sess.run(arg) to run any arg

##structure##
matrix1 = tf.constant([[3,3]])  #1x2
matrix2 = tf.constant([[2],[2]]) #2x1
product = tf.matmul(matrix1,matrix2) #matrix multify, as np.dot(m1,m2)
##structure##

#m1
sess = tf.Session()
result = sess.run(product) #use sess.run() to run above structure
print(result)
sess.close() #more systematic, could ignore

#m2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2) #session close directly

print(1)



##7 Variable
#def as Variable, that will consider

state = tf.Variable(0,name='counter') #name as counter
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value) #assign the state by updating it

init = tf.global_variables_initializer() #if define variable, must initial all

with tf.Session() as sess:
    sess.run(init) #run initialize before everything
    for _ in range(3):
        sess.run(update)
        print(sess.run(state)) #point to sess.run(state) that we could see result



##8 Placehoder
#like give a variable, but with empty content, given value below in session
input1 = tf.placeholder(tf.float32) #default only flo32, [1,1] sturcture
input2 = tf.placeholder(tf.float32) #default only flo32

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
#each time feed with different value from a dictionary



##9 Activation function
#for non-linear function
#y = AF(Wx)
#AF: relu, sigmoid, tanh, should be differentiable
#small: any above, cNN: relu, rNN: relu or tanh



##10 Activation function
#eg. AF(x) = x*1(x>0) is relu
#activate after Wx+b
#above: tf.nn.relu()
#classification problem: tf.nn.softplus()
#reduce overfitting: tf.nn.dropout()
#tf.nn.sigmoid(x)
#tf.nn.tanh(x)



##11##12 Add define layer ##13 Plot result
#ex 2-layer
#add a layer from input, act_funct then output
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #random better than all zero
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #hope not zero+0.1
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

#give data
x_data = np.linspace(-1,1,300)[:,np.newaxis] #add new axis
noise = np.random.normal(0,0.05,x_data.shape)#add something different
y_data = np.square(x_data) - 0.5 + noise

#give space for x, y for later session feeding
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1]) 

#input 1x -> hidden10 -> output 1y, 3layers
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) #1 to 2
prediction = add_layer(l1, 10, 1, activation_function = None) #2 to 3

#predict the loss by GD
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                     reduction_indices=[1])) #sum: reduce_sum(), mean: reduce_mean()
train_step = tf.train.AdamOptimizer(0.1).minimize(loss) #how to optimize? give alpha=0.1
#GradientDescentOptimizer()
#AdamOptimizer() #faster

#initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data,y_data)
plt.ion() #continue run, without pause
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data}) #small step more effici
    if i%50 == 0:
        #output show loss, become smaller as more steps
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        
        #plot the prediction y based on x_data, and line as red
        try:
            ax.lines.remove(lines[0]) #可能一開始沒有線 要先try 怕報錯不繼續跑
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value, 'r-',lw=5)        
        plt.pause(0.1)



##15 ##16 Optimizer
#SGD: cut into batch
#other method like: Adadelta, Adagrad, NAG, Momentum, AdamOptimizer, FtrlOptimizer, RMSProp
        
#Momentum
#W += -learning rate*dx, more curve and puting into slope as momentum
#m = b1*m-learning rate*dx
#W += m

#AdaGrad
#put on a bad shoes
#v += dx^2
#W += -leanring rate*dx/sqrt(v)

#RMSProp
#combine Momentum, AdaGrad
#v = b1*v+(1-b1)*dx^2
#W += -learning rate*dx/sqrt(v)

#Adam
#better and fast converge
#m = b1*m+(1-b1)*dx
#v = b2*v+(1-b2)*dx^2
#W += -learning rate*m/sqrt(v)



#17 Tensorboard
#annotate each node, and will be visualized in browser

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        return outputs


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

#after running, switch to current work directory in cmd
#cmd type: tensorboard --logdir=logs
#open the browser as instructed http://192.168.137.1:6006/
#under tag GRAPHS



##18 Tensorboard2
#hist, bias, event: show loss
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
            #give hist layer name weight 1 weights
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# Make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)

# the error between prediciton and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    #visualize the loss in event, pure scalar

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()
#merge all summary in graph

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
#record merged every 50 steps, and add in summary i



##19 Classification in TF
#MNIST input data 28x28=784, output (0100000000)
from tensorflow.examples.tutorials.mnist import input_data

#number 0 to 9
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))



##20 Overfit
#minimize error as possible with train data
#error increase when with test data
#sol: increase data size, L1,L2 regularization, dropout regularization(drop some neuron unit)
#y = Wx
#cost = (Wx - real_y)^2 + abs(W) <<L1




##21 Overfit, dropout
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    #how much percent knots to keep?
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs) #histgram sum
    return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh) #avoid into None
#if into 100 nots, will be too much
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
tf.summary.scalar('loss', cross_entropy) #scala sum
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
# summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    # here to determine the keeping probability
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    #assign how much probability to drop/keep
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

#then shift to log directory
#tensorboard --logdir='logs'



##22 CNN1
#compress length,width in a patch then increase depth in a image to next convolution layer
#stride, how many steps each
#valid padding: smaller size than origin, same padding: same size
#pooling: use pooling to prevent lost info, use stride=1 then pooling to condense to smaller size
#max pooling, average pooling



##23 CNN2,3
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    #positive better
    return tf.Variable(initial)

def conv2d(x, W):
    # 2-dim cnn, input all info=x
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # prevent too many stride, use pooling to prevent
    # stride [1, x_movement, y_movement, 1]
    # ksize:
    # strides: 1,2,2,1 making smaller
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1]) #-1 avoid the axis, 1 as channel b/w, 3 color
#print(x_image.shape) #should be [n_samples,28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) #patch 5x5, insize 1 origin, outsize 32 new convo
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # nolinear, output 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                         # output 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) #patch 5x5, insize 32 origin, outsize 64 new convo
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(x_image,W_conv2) + b_conv2) # nolinear, output 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                         # output 7x7x64

## func1 layer ##

## func2 layer ##


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
mnist.test.images, mnist.test.labels))
        
##

