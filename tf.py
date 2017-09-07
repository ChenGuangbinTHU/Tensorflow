import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3
#
# print(x_data)
# print(y_data)
#
# Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# bias = tf.Variable(tf.zeros([1]))
#
# print(Weight)
# print(bias)
#
# y = Weight * x_data + bias
# loss = tf.reduce_mean(tf.square(y-y_data))
#
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# sess = tf.Session()
# sess.run(init)
#
# for step in range(200):
#     sess.run(train)
#     if(step % 20 == 0) :
#         print(step,sess.run(Weight),sess.run(bias))
#
#
# print(tf.__version__)


def add_layer(inp,in_size,out_size,active_function):
    Weight = tf.Variable(tf.random_normal([in_size,out_size]))
    bias = tf.Variable(tf.zeros([1,out_size])+0.1)


    Wx_plus_b = tf.matmul(inp,Weight) + bias
    if active_function is None:
        output = Wx_plus_b
    else:
        output = active_function(Wx_plus_b)
    return output

x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data) -0.5 + noise

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,tf.nn.relu)
prediction = add_layer(l1,10,1,None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),axis=1))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0 :
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=2)
        plt.pause(0.1)

