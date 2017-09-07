import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEP = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006

def get_batch():
    global BATCH_START,TIME_STEP

    xs = np.arange(BATCH_START,BATCH_START+TIME_STEP*BATCH_SIZE).reshape((BATCH_SIZE,TIME_STEP))/(10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)

    # print(np.arange(BATCH_START,BATCH_START+TIME_STEP*BATCH_SIZE))
    # print(xs)
    # print(seq)
    # print(seq[:, :, np.newaxis, np.newaxis])
    # print(seq[:, :, np.newaxis])
    # print(res)

    BATCH_START += TIME_STEP
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

class LSTMRNN(object):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size,sess):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.sess = sess

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32,[None,n_steps,input_size],name='xs')
            self.ys = tf.placeholder(tf.float32,[None,n_steps,output_size],name = 'xs')

        with tf.variable_scope('in_hidden'):
            self.add_input_layer()

        with tf.variable_scope('LSTM_cell'):
            self.add_cell()

        with tf.variable_scope('out_hidden'):
            self.add_output_layer()

        with tf.name_scope('cost'):
            self.compute_cost()

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred,[-1],name = 'reshape_pred')],
            [tf.reshape(self.ys,[-1],name = 'reshape_target')],
            [tf.ones([self.batch_size * self.n_steps],dtype=tf.float32)],
            average_across_timesteps = True,
            softmax_loss_function=self.ms_error,
            name = 'losses'
        )

        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses,name = 'losses_sum'),
                self.batch_size,
                name='average_cost'
            )
            tf.summary.scalar('cost',self.cost)

    def ms_error(self,labels,logits):
        return tf.square(tf.subtract(labels,logits))

    def _weight_variable(self,shape,name='weights'):
        initializer = tf.random_normal_initializer(mean=0,stddev=1)
        return tf.get_variable(shape=shape,initializer=initializer,name=name)

    def _bias_variable(selfself,shape,name = 'bias'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name,shape=shape,initializer=initializer)


    def add_input_layer(self):
        l_in_X = tf.reshape(self.xs,[-1,self.input_size],name='2_2D')

        Ws_in = self._weight_variable([self.input_size,self.cell_size])

        bs_in = self._bias_variable([self.cell_size,])

        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_X,Ws_in)+bs_in

        self.l_in_y = tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='2_3D')

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size,forget_bias = 1.0,state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
        self.cell_outputs,self.final_state = tf.nn.dynamic_rnn(
            lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False
        )

    def add_output_layer(self):
        l_out_x = tf.reshape(self.cell_outputs,[-1,self.cell_size],name = '2_2D')

        Ws_out = self._weight_variable([self.cell_size,self.output_size])
        bs_out = self._bias_variable([self.output_size,])

        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x,Ws_out)+bs_out
        # print(sess.run(self.pred))

if __name__ == '__main__':
    # get_batch()
    # print('-'*30)
    # get_batch()
    # print('-' * 30)
    # get_batch()

    sess = tf.Session()
    model = LSTMRNN(TIME_STEP, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, sess)
    sess.run(tf.global_variables_initializer())

    a = 1
    for i in range(200):
        seq,res,xs = get_batch()

        if i == 0:
            feed_dict = {
                model.xs : seq,
                model.ys : res,
            }
        else :
            feed_dict = {
                model.xs : seq,
                model.ys : res,
                model.cell_init_state : state
            }

        _,cost,state ,pred,output,s = sess.run(
            [model.train_op,model.cost,model.final_state,model.pred,model.cell_outputs,tf.reshape(model.pred,[-1],name = 'reshape_pred')],
            feed_dict=feed_dict
        )
        if a:
            print(np.shape(pred))
            print(np.shape(output))
            print(np.shape(s))
            a = 0
        if i%20 == 0:
            print('cost',round(cost,4))