#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 22:08:02 2017

@author: cyrano
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib as contrib

import reader

#flags = tf.flags
logging = tf.logging

#全局配置
model = "small"
path = "input/ptb"

#flags.DEFINE_string("model", "small", "A type of model. Possible options are: small, medium, large.")
#flags.DEFINE_string("data_path", 'input/ptb', "data_path")
#flags.DEFINE_bool("use-fp16", False, "Train using 16-bit floats instead of 32bit floats")
#
#FLAGS = flags.FLAGS

#根据全局配置输出data_type
def data_type():
    return tf.float32;
#    if (FLAGS.use_fp16): 
#        return tf.float16
#    else: 
#        return tf.float32


class PTBModel:
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        
        #单个lstm的隐层数量
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        #输入数据 x
        self._input_data = tf.placeholder(dtype=tf.int32, shape=(batch_size, num_steps))
        #目标数据 y
        self._targets = tf.placeholder(dtype=tf.int32, shape=(batch_size, num_steps))

        #创建单个LSTM，隐匿层的单元数量，遗忘门的初始值可以为1，三向门为开
        lstm_cell = contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
        
        #在训练时以及为输出的保留几率小于1时，为每个lstm的cell加入dropout机制
        if is_training and config.keep_prob < 1:
            lstm_cell = contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        
        #多层的RNN网络，每个layers由一个lstm组成, lstm有hidden_size层
        cell = contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        
        #所有隐层的初始值为0
        self._initial_state = cell.zero_state(batch_size, data_type())


        with tf.device('/cpu:0'):
            #设定embedding变量以及转化输入单词为embedding里的词向量
            embedding = tf.get_variable(name='embedding', 
                                        shape=(vocab_size, hidden_size),
                                        dtype=data_type())
            #（embedding_lookup函数）
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
        
        #对输入进行dropout
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)
        
        #简单调用实现方式
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        # for input_ in tf.split(1, num_steps, inputs)]:
        #   outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)

        outputs = []
        
        state = self._initial_state
        
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                    
                # 从state开始运行RNN架构，输出为cell的输出以及新的state.
                cell_out, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_out)
        
        #输出定义为cell的输出乘以softmax weight w后加上softmax bias b. 即logit
        output = tf.reshape(tf.concat(outputs, axis=1), [-1, hidden_size])
        softmax_w = tf.get_variable('softmax_w', (hidden_size, vocab_size),dtype=data_type())
        softmax_b = tf.get_variable('softmax_b', (vocab_size,), dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        #loss函数是average negative log probability, 函数sequence_loss_by_examples实现
        loss = contrib.legacy_seq2seq.sequence_loss_by_example(logits=[logits], 
                                                              targets=[tf.reshape(self._targets,[-1])],
                                                              weights=[tf.ones((batch_size * num_steps,),dtype=data_type())])        
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return
        
        # learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        
        # 根据张量间的和的norm来clip多个张量
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        
        # 用之前的变量learning rate来起始梯度下降优化器。
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        
        # 一般的minimize为先取compute_gradient,再用apply_gradient
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        self._new_lr = tf.placeholder(dtype=tf.float32, shape=[],name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr)
    
    #更新learning rate
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def get_config():
    if model == "small":
        return SmallConfig()
    elif model == "medium":
        return MediumConfig()
    elif model == "large":
        return LargeConfig()
    elif model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", model)


# 在函数传递入的session里运行rnn图的cost和 fina_state结果，另外也计算eval_op的结果
# 这里eval_op是作为该函数的输入
def run_iter(session, m, data, eval_op, x, y, state, verbose, step,
             epoch_size, costs, iters, start_time):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 feed_dict={m.input_data: x,
                                            m.targets: y,
                                            m.initial_state: state})
    costs += cost
    iters += m.num_steps
    
    # 每一定量运行后输出目前结果
    if verbose and step % (epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / epoch_size, np.exp(costs / iters),
               iters * m.batch_size / (time.time() - start_time)))
    return costs, iters


def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size = (len(data) // m.batch_size - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    state = session.run(fetches=m.initial_state)
    
    #ptb_iterator函数在接受了输入，batch size以及运行的step数后输出
    #步骤数以及每一步骤所对应的一对x和y的batch数据，大小为[batch_size, num_step]
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,m.num_steps)):
        costs, iters = run_iter(session, m, data, eval_op, x, y, state, verbose, step,
                                epoch_size, costs, iters, start_time)
        
    return np.exp(costs / iters)


def main(_):
    t0 = time.time()  # 打开深度学习计时器
    
    if not path:
        raise ValueError("Must set data_path to PTB data directory")
    
    # 读取输入数据并将他们拆分开
    raw_data = reader.ptb_raw_data(path)
    train_data, valid_data, test_data, _ = raw_data
    
    print("train_data: ", np.shape(train_data))
    print("valid_data: ", np.shape(valid_data))
    print("test_data: ", np.shape(test_data))
    
    
    print("取数据耗时: ",(time.time() - t0), "s ..." )

    # 读取用户输入的config，这里用具决定了是小，中还是大模型
    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    
    # 建立了一个default图并开始session
    with tf.Graph().as_default(), tf.Session() as session:
        #先进行initialization
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        # train
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        
        # valid, test
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            m_valid = PTBModel(is_training=False, config=config)
            m_test = PTBModel(is_training=False, config=eval_config)

        session.run(tf.global_variables_initializer())
        
        # 递减learning rate
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            
            #打印出perplexity
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, m_valid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, m_test, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
    tf.app.run()