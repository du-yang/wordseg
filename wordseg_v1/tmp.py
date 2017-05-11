import tensorflow as tf
from model import config

import json

class lstm_model():
    def __init__(self,config):

        self.config = config
        self.x = tf.placeholder(tf.int32, shape=(1,config.num_steps))


        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, forget_bias=0.0, state_is_tuple=True) for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(1, tf.float32)

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.config.vocab_size, self.config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding,self.x)

        self.outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                self.outputs.append(cell_output)

        output = tf.reshape(tf.concat(axis=1, values=self.outputs), [-1, self.config.hidden_size])
        self.softmax_w = tf.get_variable(
            "softmax_w", [self.config.hidden_size, self.config.output_size], dtype=tf.float32)
        self.softmax_b = tf.get_variable("softmax_b", [self.config.output_size], dtype=tf.float32)
        self.logits = tf.matmul(output, self.softmax_w) + self.softmax_b
        self.index = tf.argmax(self.logits,axis=1)

class cut_word():

    def __init__(self,model_path='tmp/model',w2id_path='data/w2id.json'):

        with open(w2id_path) as f:
            self.w2id = json.load(f)

        conf = config()
        self.num_steps = conf.num_steps

        self.cutModel = lstm_model(conf)

        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)


    def cut(self,sentence):
        sentence = sentence.strip()
        ids = []
        for word in sentence:
            try:
                ids.append(self.w2id[word])
            except:
                ids.append(0)
        if len(ids)<self.num_steps:
            ids.extend([0]*(self.num_steps-len(ids)))
        else:ids=ids[:self.num_steps]

        # with tf.Session() as sess:
        index = self.sess.run(self.cutModel.index,{self.cutModel.x:[ids]})

        result = []
        tmp = []
        for i,word in enumerate(sentence):
            if i==0:tmp.append(word)
            elif i>0:
                if index[i-1]==1 and index[i]==1:
                    result.append(''.join(tmp))
                    tmp=[]
                    tmp.append(word)
                if index[i-1] == 1 and index[i] == 2:
                    tmp.append(word)
                if index[i - 1] == 2 and index[i] == 1:
                    result.append(''.join(tmp))
                    tmp = []
                    tmp.append(word)
                if index[i - 1] == 2 and index[i] == 2:
                    tmp.append(word)
                elif index[i]==0:
                    result.append(''.join(tmp))
                    tmp=[]
                    result.append('0')
        result.append(''.join(tmp))

        return result


if __name__ == '__main__':

    cuter = cut_word()
    print(cuter.cut('京东还是很不错的'))
    print(cuter.cut('我想买这个'))
    print(cuter.cut('帮我推荐一款手机吧'))
    print(cuter.cut('哈哈哈哈哈这里是电子科技大学'))
    print(cuter.cut('谁的青春不放荡'))
    print(cuter.cut('这道题我不会做'))
    print(cuter.cut('优衣库的衣服质量不错'))

