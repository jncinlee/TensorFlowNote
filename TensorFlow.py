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



##1112 Add define layer ##13 Plot result
#ex 2-layer
#add a layer from input, act_funct then output
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size])) #random better than all zero
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #hope not zero+0.1
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
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
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) #how to optimize? give alpha=0.1

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































