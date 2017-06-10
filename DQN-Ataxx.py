#coding=utf-8
import gym
from gym.wrappers import Monitor # recording results
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque,namedtuple

if "./" not in sys.path:
  sys.path.append("./")


# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
VALID_ACTIONS=[0,1,2,3]


class Estimator():
    """Q-Value Estimator neural network.
        This network is used for both the Q-Network and the Target Network.
    """
    def __init__(self,scope="estimator",summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir,"summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
    def _build_model(self):
        self.X_pl = tf.placeholder(shape=[None,7,7],dtype=tf.uint8,name="X")
        self.y_pl = tf.placeholder(shape=[None],dtype=tf.uint32,name='y')
        self.actions_pl = tf.placeholder(shape=[None],dtype=tf.uint32,name='actions')

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        conv1 = tf.contrib.layers.conv2d(X,32,8,4,activation_fn=tf.nn.relu) # input,output,kernel size,stride
        conv2 = tf.contrib.layers.conv2d(conv1,64,4,2,activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2,64,3,1,activation_fn=tf.nn.relu)

        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.full_connected(flattened,512)
        self.predictions = tf.contrib.layers.full_connected(fc1,len(VALID_ACTIONS))

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1]+self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions,[-1]),gather_indices)

        self.losses = tf.squared_difference(self.y_pl,self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6)
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.contrib.framework.get_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss',self.loss),
            tf.summary.histogram('lost_hist',self.losses),
            tf.summary.histogram('q_values_hist',self.predictions),
            tf.summary.scalar('max_q_value',tf.reduce_max(self.predictions))
        ])

    def predict(self,sess,s):
        return sess.run(self.predictions,{self.X_pl:s})

    def update(self,sess,s,a,y):
        feed_dict = {self.X_pl:s,self.y_pl:y,self.actions_pl:a}
        summaries, global_step,_,loss = sess.run(
            [self.summaries,tf.contrib.framework.get_global_step(),
             self.train_op,self.loss],feed_dict
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries,global_step)
        return loss

if __name__ == '__main__':
    for item in gym.envs.registry.all():
         print str(item)+'\n'




