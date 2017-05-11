import tensorflow as tf
import os

from model import lstm_model,config
from read_data import read_data_sets


def train(config,echoNum):

    lstmer = lstm_model(config)
    jd_data = read_data_sets('data/jdData.json', 'data/w2id.json', config.num_steps)

    with tf.Session() as sess:

        if not os.path.exists('tmp/'):
            os.mkdir('tmp/')

        loss_op = lstmer.loss()
        train_op = lstmer.training()

        saver = tf.train.Saver()
        if os.path.exists('tmp/checkpoint'):  # 判断模型是否存在
            saver.restore(sess, 'tmp/model')  # 存在就从模型中恢复变量
        else:
            init = tf.global_variables_initializer()  # 不存在就初始化变量
            sess.run(init)


        for i in range(echoNum):
            x_data, y_data = jd_data.next_batch(config.batch_size)

            print('训练前loss：',sess.run(loss_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data}))

            sess.run(train_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data})

            saver.save(sess, './tmp/model')

            print('训练后loss：',sess.run(loss_op, feed_dict={lstmer.x: x_data, lstmer.y: y_data}))
            # print('预测结果：',sess.run(lstmer.logits,feed_dict={lstmer.x: x_data, lstmer.y: y_data}))
            # print(y_data)
            print('完成第%s轮' % i)

def main():
    conf = config()
    conf.batch_size = 50
    conf.num_steps = 50
    train(conf,1000)

if __name__ == '__main__':
    main()