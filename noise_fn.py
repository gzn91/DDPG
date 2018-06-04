"""
Parameter space noise, https://arxiv.org/abs/1706.01905
"""

import tensorflow as tf

class AdaptiveParamNoise:
    def __init__(self, initial_stddev=0.05, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = tf.Variable(initial_stddev)
        self.desired_action_stddev = tf.Variable(desired_action_stddev)
        self.adoption_coefficient = tf.Variable(adoption_coefficient)
        self.current_stddev = tf.Variable(initial_stddev)

    def adapt_stddev(self, distance):
        return tf.cond(distance > self.desired_action_stddev,
                       lambda: self.current_stddev.assign(tf.div(self.current_stddev, self.adoption_coefficient)),
                       lambda: self.current_stddev.assign(tf.multiply(self.current_stddev, self.adoption_coefficient)))

    def __repr__(self):
        _repr = 'AdaptiveParamNoise(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return _repr.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)
