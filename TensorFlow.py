##Neural Network, TensorFlow
#only run on Python3.5 under Program Files/Python35

import tensorflow as tf
import numpy as np



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
        
