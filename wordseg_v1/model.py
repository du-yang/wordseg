import inspect
import tensorflow as tf
import numpy as np
from read_data import read_data_sets




class lstm_model():
    def __init__(self, config):
        '''
        :param config:config类，一些训练参数设置
        '''
        self.config = config
        self.x = tf.placeholder(tf.int32, shape=(config.batch_size, config.num_steps))
        self.y = tf.placeholder(tf.int32, shape=(config.batch_size, config.num_steps))

        # def lstm_cell():
        #     return tf.contrib.rnn.BasicLSTMCell(
        #             self.config.hidden_size, forget_bias=0.0, state_is_tuple=True)

        attn_cell = tf.contrib.rnn.BasicLSTMCell(
                         self.config.hidden_size, forget_bias=0.0, state_is_tuple=True)

        # attn_cell = lstm_cell
        # if config.keep_prob < 1:
        #     def attn_cell():
        #         return tf.contrib.rnn.DropoutWrapper(
        #             lstm_cell(), output_keep_prob=config.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True) for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(self.config.batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [self.config.vocab_size, self.config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding,self.x)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.config.hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [self.config.hidden_size, self.config.output_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [self.config.output_size], dtype=tf.float32)
        self.logits = tf.matmul(output, softmax_w) + softmax_b

    def loss(self):

        logits = self.logits
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.y, [-1])],
            [tf.ones([self.config.batch_size * self.config.num_steps], dtype=tf.float32)])
        # 交叉熵损失函数，下一篇专门讲下tensorflow中的几个损失函数的实现
        cost = tf.reduce_sum(loss) / self.config.batch_size

        return cost

    def training(self):

        loss = self.loss()

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.config.learningrate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars))
        # global_step = tf.contrib.framework.get_or_create_global_step()

        return train_op

class config():
    '''参数配置类'''
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    hidden_size = 200
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 50
    num_steps = 50
    vocab_size = 3000
    output_size = 3
    learningrate = 0.5

if __name__ == '__main__':

    conf = config()
    lstmer = lstm_model(config())
    jd_data = read_data_sets('jdData.json',conf.num_steps)

    with tf.Session() as sess:

        saver = tf.train.Saver()


        loss_op = lstmer.loss()
        train_op = lstmer.training()
        # tf.global_variables_initializer().run()
        saver.restore(sess,'tmp/model')


        for i in range(100):
            x_data, y_data = jd_data.next_batch(conf.batch_size)

            print('训练前loss：',sess.run(loss_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data}))

            sess.run(train_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data})

            saver.save(sess, './tmp/model')
            # for j in range(10):
            #     sess.run(train_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data})
            #
            #     print('第%s_%i训练后loss：'%(i,j), sess.run(loss_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data}))

            print('训练后loss：',sess.run(loss_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data}))
            print('预测结果：',sess.run(lstmer.logits,feed_dict={lstmer.x: x_data, lstmer.y: y_data}))
            print(y_data)

            # print('loss1:',loss1,'    ','loss2:',loss2)
            print('完成第%s轮' % i)