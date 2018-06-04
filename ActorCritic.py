"""
DDPG algorithm,
Author: Gustaf Jacobzon, Martin Larsson
Date: 2018 03 20
"""
import tensorflow as tf
import tensorflow.contrib as tc
from noise_fn import AdaptiveParamNoise
from running_mean_std import Stats


def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


class Actor:
    def __init__(self, sess, save_path, state_dim, action_dim, action_bound, normalize_observations=True,
                 use_param_noise=False, tau=0.01, gamma=.99, delta=0.2, learning_rate=0.0001,
                 observation_range=(-5., 5.)):
        self.sess = sess
        self.save_path = save_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.gamma = gamma
        self.delta = delta
        self.learning_rate = learning_rate
        self.observation_range = observation_range
        self.normalize_observations = normalize_observations

        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.drop_prob_ph = tf.placeholder(tf.float32)
        self.critic_gradients = tf.placeholder(tf.float32, [None, self.action_dim])
        self.obs_rms = None

        # Observation normalization.
        if self.normalize_observations:
            with tf.variable_scope('obs_rms'):
                self.obs_rms = Stats(self.sess,shape=self.state_dim)
                # self.obs_rms = RunningMeanStd(shape=self.state_dim)

        with tf.variable_scope('actor'):
            self.actions = self.policy_network(self.state_ph, self.drop_prob_ph)
        self.policy_params = tf.trainable_variables(scope='actor')

        if use_param_noise:
            with tf.variable_scope('perturbed_actor'):
                self.perturbed_actions = self.policy_network(self.state_ph, self.drop_prob_ph)
            self.perturbed_policy_params = tf.trainable_variables(scope='perturbed_actor')

            self.adapt_noise = AdaptiveParamNoise(initial_stddev=.1, desired_action_stddev=self.delta)
            self.stddev_ph = tf.placeholder(tf.float32)

            self.action_distance, self.stddev = self.adaptive_param_noise()

            # with tf.variable_scope('Actor'):
            #     variable_summaries(self.stddev, 'stddev')
            #     variable_summaries(self.action_distance, 'action_distance')

            self.update_perturbed_policy_params = [self.perturbed_policy_params[i].assign(self.policy_params[i] +
                                                                                          tf.random_normal(tf.shape(
                                                                                            self.perturbed_policy_params
                                                                                            [i]), 0., self.stddev_ph, seed=1337))
                                                   for i in range(len(self.perturbed_policy_params)) if
                                                   'LayerNorm' not in self.perturbed_policy_params[i].name]

        with tf.variable_scope('target_actor'):
            self.target_actions = self.policy_network(self.state_ph, self.drop_prob_ph)
        self.target_policy_params = tf.trainable_variables(scope='target_actor')

        self.objective_gradient = \
            tf.gradients(self.actions, self.policy_params, -self.critic_gradients)  # , tf.shape(self.state_ph)[0])

        self.update_policy_params = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.objective_gradient, self.policy_params))

        self.update_target_policy_params = [self.target_policy_params[i].assign(self.tau * self.policy_params[i]
                                                                                + (1 - self.tau) *
                                                                                self.target_policy_params[i])
                                            for i in range(len(self.target_policy_params))]

        self.merge_op = tf.summary.merge_all(scope='Actor')

        self.saver = tf.train.Saver()

        # 64,128,128
    def policy_network(self, state, drop_prob, seed=1337):
        state = tf.clip_by_value(normalize(state, self.obs_rms),
                                 self.observation_range[0], self.observation_range[1])

        # Layer 1
        h1 = tf.layers.dense(state, 128)
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h1 = tf.nn.elu(h1)
        h1 = tf.layers.dropout(h1, drop_prob, seed=seed)

        # Layer 2
        h2 = tf.layers.dense(h1, 128)
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
        h2 = tf.nn.elu(h2)
        h2 = tf.layers.dropout(h2, drop_prob, seed=seed)

        # Layer 3
        h3 = tf.layers.dense(h2, 64)
        h3 = tc.layers.layer_norm(h3, center=True, scale=True)
        h3 = tf.nn.elu(h3)
        h3 = tf.layers.dropout(h3, drop_prob, seed=seed)

        # Layer 4
        h4 = tf.layers.dense(h3, 64)
        h4 = tc.layers.layer_norm(h4, center=True, scale=True)
        h4 = tf.nn.elu(h4)
        h4 = tf.layers.dropout(h4, drop_prob, seed=seed)

        # # Layer 5
        # h5 = tf.layers.dense(h4, 64)
        # h5 = tc.layers.layer_norm(h5, center=True, scale=True)
        # h5 = tf.nn.elu(h5)
        # h5 = tf.layers.dropout(h5, drop_prob, seed=seed)

        # Output layer
        actions = tf.layers.dense(h4, self.action_dim, kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3, seed=seed))
        actions = tf.nn.tanh(actions)

        return actions

    def add_param_noise(self, state):
        adaptive_policy_distance, stddev = self.sess.run([self.action_distance, self.stddev],
                                                         feed_dict={self.state_ph: state, self.drop_prob_ph: 0.0})

        self.sess.run(self.update_perturbed_policy_params, feed_dict={self.stddev_ph: stddev})

        return adaptive_policy_distance, stddev

    def adaptive_param_noise(self):
        adaptive_policy_distance = tf.sqrt(tf.reduce_mean(
            tf.squared_difference(self.actions, self.perturbed_actions)))

        std_dev = self.adapt_noise.adapt_stddev(adaptive_policy_distance)
        # std_dev = tf.Print(std_dev, [std_dev])

        return adaptive_policy_distance, std_dev

    def train(self, state, drop_prob, critic_grads):
        self.sess.run(self.update_policy_params, feed_dict={self.state_ph: state,
                                                            self.drop_prob_ph: drop_prob,
                                                            self.critic_gradients: critic_grads})

    def predict(self, state, drop_prob):
        return self.sess.run(self.actions, feed_dict={self.state_ph: state,
                                                      self.drop_prob_ph: drop_prob})

    def perturbed_predict(self, state, drop_prob):
        return self.sess.run(self.perturbed_actions, feed_dict={self.state_ph: state,
                                                                self.drop_prob_ph: drop_prob})

    def target_predict(self, state, drop_prob):
        return self.sess.run(self.target_actions, feed_dict={self.state_ph: state,
                                                             self.drop_prob_ph: drop_prob})

    def restore_model(self):
        self.saver.restore(self.sess, './' + self.save_path + '/model_actor.ckpt')

    def save_model(self):
        self.saver.save(self.sess, './' + self.save_path + '/model_actor.ckpt')


class Critic:

    def __init__(self, sess, save_path, state_dim, action_dim, normalize_observations=True,
                 tau=0.01, gamma=.99, learning_rate=0.001, l2_reg=0.01, observation_range=(-5., 5.)):
        self.sess = sess
        self.save_path = save_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.normalize_observations = normalize_observations
        self.observation_range = observation_range
        self.obs_rms = None

        self.state_ph = tf.placeholder(tf.float32, [None, self.state_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.future_reward_ph = tf.placeholder(tf.float32, [None, 1])
        self.drop_prob_ph = tf.placeholder(tf.float32)

        if self.normalize_observations:
            with tf.variable_scope('obs_rms',reuse=tf.AUTO_REUSE):
                self.obs_rms = Stats(self.sess, shape=self.state_dim)
                # self.obs_rms = RunningMeanStd(shape=self.state_dim)

        with tf.variable_scope('critic'):
            self.q_value = self.q_value_network(self.state_ph, self.action_ph, self.drop_prob_ph)
            #variable_summaries(self.q_value, 'Q_value')
        self.q_params = tf.trainable_variables(scope='critic')

        with tf.variable_scope('target_critic'):
            self.target_q_value = self.q_value_network(self.state_ph, self.action_ph, self.drop_prob_ph)
        self.target_q_params = tf.trainable_variables(scope='target_critic')

        self.critic_gradients = tf.gradients(self.q_value, self.action_ph)

        # Compute loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_value, self.future_reward_ph)) + \
                    self.l2_reg*tf.losses.get_regularization_loss(scope='critic')

        # Train
        self.update_q_params = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.update_target_q_params = [self.target_q_params[i].assign(self.tau * self.q_params[i]
                                                                      + (1 - self.tau) * self.target_q_params[i]) for i
                                       in range(len(self.target_q_params))]

        self.saver = tf.train.Saver()

        # with tf.variable_scope('critic', reuse=True):
        #     tf.summary.scalar('Q_learning_loss', self.loss)

        self.merge_op = tf.summary.merge_all(scope='critic')

    # 64 128, 128 512
    # 128 512 512 runt 4k 200 e
    # 700,600,200 0.5drop
    def q_value_network(self, state, action, drop_prob, seed=1337):
        state = tf.clip_by_value(normalize(state, self.obs_rms),
                                 self.observation_range[0], self.observation_range[1])

        # Layer 1
        h1 = tf.layers.dense(state, 128, kernel_regularizer=tf.nn.l2_loss)
        h1 = tc.layers.layer_norm(h1, center=True, scale=True)
        h1 = tf.nn.elu(h1)
        h1 = tf.layers.dropout(h1, drop_prob, seed=seed)

        # Layer 2 + add actions
        h2 = tf.concat([h1, action], axis=-1)

        h2 = tf.layers.dense(h2, 128, kernel_regularizer=tf.nn.l2_loss)
        h2 = tc.layers.layer_norm(h2, center=True, scale=True)
        h2 = tf.nn.elu(h2)
        h2 = tf.layers.dropout(h2, drop_prob, seed=seed)

        # Layer 3
        h3 = tf.layers.dense(h2, 64, kernel_regularizer=tf.nn.l2_loss)
        h3 = tc.layers.layer_norm(h3, center=True, scale=True)
        h3 = tf.nn.elu(h3)
        h3 = tf.layers.dropout(h3, drop_prob, seed=seed)

        # Layer 4
        h4 = tf.layers.dense(h3, 64, kernel_regularizer=tf.nn.l2_loss)
        h4 = tc.layers.layer_norm(h4, center=True, scale=True)
        h4 = tf.nn.elu(h4)
        h4 = tf.layers.dropout(h4, drop_prob, seed=seed)

        # Layer 5
        # h5 = tf.layers.dense(h4, 64, kernel_regularizer=tf.nn.l2_loss)
        # h5 = tc.layers.layer_norm(h5, center=True, scale=True)
        # h5 = tf.nn.elu(h5)
        # h5 = tf.layers.dropout(h5, drop_prob, seed=seed)

        # Output q_value
        q_value = tf.layers.dense(h4, 1, kernel_initializer=tf.random_uniform_initializer(-3e-4, 3e-4, seed=seed))

        return q_value

    def compute_grads(self, state, action, drop_prob):
        return self.sess.run(self.critic_gradients, feed_dict={self.state_ph: state,
                                                               self.action_ph: action,
                                                               self.drop_prob_ph: drop_prob})

    def train(self, state, action, drop_prob, future_reward):
        self.sess.run(self.update_q_params, feed_dict={self.state_ph: state,
                                                         self.action_ph: action,
                                                         self.drop_prob_ph: drop_prob,
                                                         self.future_reward_ph: future_reward})

    def predict(self, state, action, drop_prob):
        return self.sess.run(self.q_value, feed_dict={self.state_ph: state,
                                                      self.action_ph: action,
                                                      self.drop_prob_ph: drop_prob})

    def target_predict(self, state, action, drop_prob):
        return self.sess.run(self.target_q_value, feed_dict={self.state_ph: state,
                                                             self.action_ph: action,
                                                             self.drop_prob_ph: drop_prob})

    def restore_model(self):
        self.saver.restore(self.sess, './' + self.save_path + '/model_actor.ckpt')

    def save_model(self):
        self.saver.save(self.sess, './' + self.save_path + '/model_actor.ckpt')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
