"""
DDPG algorithm,
Author: Gustaf Jacobzon, Martin Larsson

GLEW FIX: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so
"""
from ActorCritic import *
import tensorflow as tf
import numpy as np
import gym
import mujoco_py
from datetime import datetime
from gym import wrappers
import argparse
import pprint as pp
from replay_buffer import ReplayBuffer


def preprocess_state(state):
    return np.expand_dims(state, 0)


def build_summaries():
    with tf.variable_scope('total_reward'):
        var = tf.Variable(0.)
        tf.summary.scalar('reward', var)
        tf.summary.histogram('histogram', var)
    return tf.summary.merge_all(scope='total_reward'), var


def restore_model(sess,saver):
    saver.restore(sess, './' + args['env'] + '/model_vars.ckpt')


def save_model(sess,saver):
    saver.save(sess, './' + args['env'] + '/model_vars.ckpt')


def train(sess, env, args, actor, critic, noise, state_dim, action_dim, saver, writer):

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Q' <-- Q, u' <-- u
    sess.run([actor.update_target_policy_params, critic.update_target_q_params])

    # Init replay buffer and noise
    replay_buffer = ReplayBuffer(int(args['buffer_size']), state_dim=state_dim, action_dim=action_dim, random_seed=int(args['random_seed']))

    total_steps = 0
    total_episodes = 0

    if args['restore']:
        restore_model(sess, saver)
        actor.restore_model()
        critic.restore_model()
        print('Model restored!')
        total_steps, total_episodes = sess.run([total_steps_var, total_episodes_var])

    if args['use_adaptive_parameter_noise']:
        agent = actor.perturbed_predict
    else:
        agent = actor.predict

    max_exploration_steps = args['max_exploration_steps']
    p = 1
    drop_prob = args['drop_prob']
    for episode in range(args['max_episodes']):

        _distance = []
        _std_dev = []

        observation = env.reset()

        state = preprocess_state(observation)

        # if episode == 0 and args['use_adaptive_parameter_noise']:
        #     ## Initialize perturbed weights
        #     _, _ = actor.add_param_noise(state)

        total_reward = 0
        _evaluate = False

        for step in range(args['max_episode_len']):

            if total_steps < max_exploration_steps:

                if args['render_env']:
                    env.render()

                actions = agent(state, drop_prob=drop_prob)
                if not args['use_adaptive_parameter_noise']:
                    actions += noise(0, .2, action_dim)

                # if total_steps % 100 == 0:
                #     print(total_steps)

                p -= 1 / max_exploration_steps

            else:
                # actions = actor.predict(state,0.)
                print('### TRAINING COMPLETE ###')
                # key_press = input('Press 1 to eval\n'
                #       'Press 0 to save\n: ')
                # print('')
                # if key_press != '1':
                raise KeyboardInterrupt
                # else:
                #     return

            # else:

            action = actions[0]
            # print(action)

            next_state, reward, terminal, _ = env.step(action)

            replay_buffer.add(state.reshape(state_dim, ), action.reshape(action_dim, ),
                              reward, terminal, next_state.reshape(state_dim, ))
            if actor.normalize_observations:
                actor.obs_rms.update(state)

            state = preprocess_state(next_state)
            total_reward += reward
            total_steps += 1
            if total_steps % 10000 == 0:
                _evaluate = True

            if terminal:
                if _evaluate:
                    evaluate(sess, env, args, actor, critic, saver, writer)
                break

        for train_step in range(args['train_step']):
            if replay_buffer.size >= args['minibatch_size']:
                state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = replay_buffer.sample_batch(
                    args['minibatch_size'])

                # Compute future reward
                y = reward_batch*10 + critic.gamma * \
                    critic.target_predict(next_state_batch,
                                          actor.target_predict(next_state_batch, drop_prob),
                                          drop_prob)

                # Update critic
                critic.train(state_batch, action_batch, drop_prob, y)

                # Update policy
                actor.train(state_batch, drop_prob,
                            critic.compute_grads(state_batch, action_batch, drop_prob)[0])

                # Update target networks
                sess.run(actor.update_target_policy_params)
                sess.run(critic.update_target_q_params)

                # Adapt noise
                if args['use_adaptive_parameter_noise'] and train_step % 20 == 0:
                    distance, std_dev = actor.add_param_noise(state_batch)
                    _distance.append(distance)
                    _std_dev.append(std_dev)

        total_episodes += 1
        distance, std_dev = actor.add_param_noise(state_batch)
        _distance.append(distance)
        _std_dev.append(std_dev)
        # summary = sess.run(merge_op, feed_dict={total_reward_var: total_reward})
        # writer.add_summary(summary, total_steps)
        sess.run([total_steps_var.assign(total_steps), total_episodes_var.assign(episode)])
        # if episode % 10 == 0:
        if args['use_adaptive_parameter_noise']:
            print(
                f'|| Global step: {total_steps} || Episode: {total_episodes} || Reward: {total_reward:.2f} || Distance: {np.array(_distance).mean():.4f} ||')
        else:
            print(
                f'|| Global step: {total_steps} || Episode: {total_episodes} || Reward: {total_reward:.2f} ||')


def evaluate(sess,env,args,actor,critic, saver, writer):
    total_steps = sess.run(total_steps_var)
    _reward = []

    if args['restore'] and not args['train']:
        sess.run(tf.global_variables_initializer())
        restore_model(sess, saver)
        actor.restore_model()
        critic.restore_model()
        print('Model restored!')

    for episode in range(10):

        observation = env.reset()

        state = preprocess_state(observation)

        total_reward = 0

        for step in range(args['max_episode_len']):

            actions = actor.predict(state, drop_prob=0.)

            action = actions[0]

            next_state, reward, terminal, _ = env.step(action)
            if actor.normalize_observations:
                actor.obs_rms.update(state)
            next_state = preprocess_state(next_state)

            state = next_state
            total_reward += reward

            if terminal:
                break
        _reward.append(total_reward)
    total_reward = np.max(np.array(_reward))
    summary = sess.run(merge_op, feed_dict={total_reward_var: total_reward})
    writer.add_summary(summary, total_steps)
    print(
                f'|| Evaluating at step: {total_steps} || Reward: {total_reward:.2f} ||')


def main(args):
    tf.set_random_seed(int(args['random_seed']))
    np.random.seed(int(args['random_seed']))
    with tf.Session() as sess:
        try:
            env = gym.make(args['env'])
            env.seed(int(args['random_seed']))
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            assert (env.action_space.high == -env.action_space.low).all()
            action_bound = env.action_space.high[0]

            global total_steps_var, total_episodes_var
            total_episodes_var = tf.Variable(0)
            total_steps_var = tf.Variable(0)

            if args['use_gym_monitor']:
                if not args['render_env']:
                    env = wrappers.Monitor(
                        env, args['monitor_dir'], video_callable=False, force=True)
                else:
                    env = wrappers.Monitor(env, args['monitor_dir'], force=True)

            actor = Actor(sess, args['env'], state_dim, action_dim, action_bound,
                          use_param_noise=args['use_adaptive_parameter_noise'], tau=args['tau'], gamma=args['gamma'],
                          delta=args['delta'], normalize_observations=args['normalize_obs'])
            critic = Critic(sess, args['env'], state_dim, action_dim, tau=args['tau'], gamma=args['gamma'],
                            learning_rate=args['critic_lr'], normalize_observations=args['normalize_obs'])

            ## Noise
            noise = np.random.normal

            saver = tf.train.Saver()
            writer = tf.summary.FileWriter(args['summary_dir'] + '/' + args['env'] + '/' + str(datetime.now()))

            if args['train']:
                train(sess,env,args,actor,critic,noise,state_dim,action_dim,saver, writer)
            evaluate(sess,env,args,actor,critic,saver, writer)

        except KeyboardInterrupt:
            if args['save']:
                save_model(sess,saver)
                actor.save_model()
                critic.save_model()
                print('Model saved!')


parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

# agent parameters
parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
parser.add_argument('--tau', help='soft target update parameter', default=0.01)
parser.add_argument('--use-adaptive-parameter-noise', help='', default=True)
parser.add_argument('--delta', help='desired action deviation', default=.2)
parser.add_argument('--l2_reg', help='Q loss regularization strength', default=.0)
parser.add_argument('--drop-prob', help='drop out probability', default=.4)
parser.add_argument('--buffer-size', help='max size of the replay buffer', default=100000)
parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=128)
parser.add_argument('--normalize-obs', help='normalizes observations', default=True)

# run parameters
parser.add_argument('--train', help='train the model', default=True)
parser.add_argument('--train_step', help='number of training steps', default=50)
parser.add_argument('--max-exploration-steps', help='', default=1e6)
parser.add_argument('--env', help='choose the gym env', default='HalfCheetah-v2')
parser.add_argument('--save', help='save trained model', default=True)
parser.add_argument('--restore', help='restore prev trained model', default=False)
parser.add_argument('--random-seed', help='random seed for repeatability', default=1337)
parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=500000)
parser.add_argument('--max-episode-len', help='max length of 1 episode', default=5000)
parser.add_argument('--render-env', help='render the gym env', action='store_true')
parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

parser.set_defaults(render_env=False)
parser.set_defaults(use_gym_monitor=False)

args = vars(parser.parse_args())

# with open('./random_seeds_used', 'a+') as f:
#     f.write(f"Time: {str(datetime.now())}, random seed: {args['random_seed']}\n")


pp.pprint(args)
merge_op, total_reward_var = build_summaries()
main(args)
